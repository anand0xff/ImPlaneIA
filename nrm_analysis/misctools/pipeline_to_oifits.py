#! /usr/bin/env python
"""
Given a Calwebb Ami3 pipeline product (i.e. FITS table of interferometric observables),
convert that file into OIFITS format.
Identify a MIRAGE filename used to generate this observation; header contains some info for OIFITS.
Read MIRAGE data file with existing ImPlaneIA tools (InstrumentData class) to access info4oif_dict
Pass info dict along, and read in the FITS table contents in ObservablesFromFITSTable object
Function table2oif makes a dict of ALL the info and saves as OIFITS to specified directory.
"""
import numpy as np
import os
import glob
from scipy.special import comb
from astropy.time.core import Time
from astropy.io import fits
from nrm_analysis import InstrumentData
from nrm_analysis.misctools.implane2oifits import observable2dict, populate_NRM, Plot_observables
import nrm_analysis.misctools.oifits as oifits


class ObservablesFromFITSTable():
    """
    oifinfo is the info4oif_dict made by InstrumentData
    """
    def __init__(self, nh, oifinfo, fitsname,
                 #oifpath=None,
                 observables=("phases", "amplitudes", "CPs", "CAs"),
                 angunit="radians",
                 verbose=True):
        self.fitsname = fitsname
        #self.oifpath = oifpath
        self.verbose = verbose
        self.observables = observables
        self.oifinfo = oifinfo
        # Assume same number of observable output files for each observable.
        # Eg 1 slice or NINT slices cube output from eg implaneia.
        # Each image analyzed has a phases, an amplitudes, ... txt output file in this txtdir.
        # Each file might contain different numbers of individual quantities
        if verbose: print('Example of txt file pattern:  txtpath/{0:s}*.txt'.format(self.observables[0]))
        #   - yes many fringes, more cp's, and so on.
        self.nslices = 1 # for now, assume this is an averaged (2d) file
        self.nh = nh
        self.nbl = int(comb(self.nh, 2))
        self.ncp = int(comb(self.nh, 3))
        self.nca = int(comb(self.nh, 4))
        # arrays of observables, (nslices,nobservables) shape.
        self.fp = np.zeros((self.nslices, self.nbl))
        self.fa = np.zeros((self.nslices, self.nbl))
        self.cp = np.zeros((self.nslices, self.ncp))
        if len(self.observables) == 4:
            self.ca = np.zeros((self.nslices, self.nca))
        self.angunit = angunit
        if verbose:
            print("assumes angles in", angunit)
        if verbose:
            print("angle unit:", angunit)
        if angunit == 'radians':
            self.degree = 180.0 / np.pi
        else:
            self.degree = 1

        self._readfitsdata()
        if self.verbose:
            self._showdata()

    def _makequads_all(self):
        """ returns int array of quad hole indices (0-based),
            and float array of three uvw vectors in all quads
        """
        nholes = self.ctrs.shape[0]
        qlist = []
        for i in range(nholes):
            for j in range(nholes):
                for k in range(nholes):
                    for q in range(nholes):
                        if i < j and j < k and k < q:
                            qlist.append((i, j, k, q))
        qarray = np.array(qlist).astype(np.int)
        if self.verbose:
            print("qarray", qarray.shape, "\n", qarray)
        qname = []
        uvwlist = []
        # foreach row of 3 elts...
        for quad in qarray:
            qname.append("{0:d}_{1:d}_{2:d}_{3:d}".format(
                quad[0], quad[1], quad[2], quad[3]))
            if self.verbose:
                print('quad:', quad, qname[-1])
            uvwlist.append((self.ctrs[quad[0]] - self.ctrs[quad[1]],
                            self.ctrs[quad[1]] - self.ctrs[quad[2]],
                            self.ctrs[quad[2]] - self.ctrs[quad[3]]))
        if self.verbose:
            print(qarray.shape, np.array(uvwlist).shape)
        return qarray, np.array(uvwlist)

    def _maketriples_all(self):
        """ returns int array of triple hole indices (0-based),
            and float array of two uv vectors in all triangles
        """
        nholes = self.ctrs.shape[0]
        tlist = []
        for i in range(nholes):
            for j in range(nholes):
                for k in range(nholes):
                    if i < j and j < k:
                        tlist.append((i, j, k))
        tarray = np.array(tlist).astype(np.int)
        if self.verbose:
            print("tarray", tarray.shape, "\n", tarray)

        tname = []
        uvlist = []
        # foreach row of 3 elts...
        for triple in tarray:
            tname.append("{0:d}_{1:d}_{2:d}".format(
                triple[0], triple[1], triple[2]))
            if self.verbose:
                print('triple:', triple, tname[-1])
            uvlist.append((self.ctrs[triple[0]] - self.ctrs[triple[1]],
                           self.ctrs[triple[1]] - self.ctrs[triple[2]]))
        # print(len(uvlist), "uvlist", uvlist)
        if self.verbose:
            print(tarray.shape, np.array(uvlist).shape)
        return tarray, np.array(uvlist)

    def _makebaselines(self):
        """
        ctrs (nh,2) in m
        returns np arrays of eg 21 baselinenames ('0_1',...), eg (21,2) baselinevectors (2-floats)
        in the same numbering as implaneia
        """
        nholes = self.ctrs.shape[0]
        blist = []
        for i in range(nholes):
            for j in range(nholes):
                if i < j:
                    blist.append((i, j))
        barray = np.array(blist).astype(np.int)
        # blname = []
        bllist = []
        for basepair in blist:
            # blname.append("{0:d}_{1:d}".format(basepair[0],basepair[1]))
            baseline = self.ctrs[basepair[0]] - self.ctrs[basepair[1]]
            bllist.append(baseline)
        return barray, np.array(bllist)

    def _showdata(self, prec=4):
        """ set precision of your choice in calling this"""
        print('nh {0:d}  nslices {1:d}  nbl {2:d}  ncp {3:d}  nca {4:d}  '.format(
            self.nh, self.nslices, self.nbl, self.ncp, self.nca), end="")
        print("observables in np arrays with {:d} rows".format(self.nslices))

        if len(self.observables) == 4:
            print('nca', self.nca)
        else:
            print()
        np.set_printoptions(precision=prec)

        print(self.fp.shape, "fp (degrees, but stored internally in radians):\n",
              self.fp * self.degree, "\n")
        print(self.fa.shape, "fa:\n", self.fa, "\n")

        print(self.cp.shape, "cp (degrees, but stored internally in radians):\n",
              self.cp * self.degree, "\n")
        if len(self.observables) == 4:
            print(self.ca.shape, "ca:\n", self.ca, "\n")
        # print("implane2oifits._showdata: self.info4oif_dict['objname']", self.info4oif_dict)

        print("hole centers array shape:", self.ctrs.shape)

        print(len(self.bholes), "baseline hole indices\n", self.bholes)
        print(self.bls.shape, "baselines:\n", self.bls)

        print(self.tholes.shape, "triple hole indices:\n", self.tholes)
        print(self.tuv.shape, "triple uv vectors:\n", self.tuv)

        print(self.qholes.shape, "quad hole indices:\n", self.qholes)
        print(self.quvw.shape, "quad uvw vectors:\n", self.quvw)

    def _readfitsdata(self):
        # to only be called from init
        # for now we are assuming nslices = 1

        if self.verbose:
            print("\tfile names that are being looked for:")
            print("\t",self.fitsname)

        # load from fits table into data arrays:
        with fits.open(self.fitsname) as hdu1:
            for slice in range(self.nslices):
                self.fp[slice:] = hdu1[6].data.astype(np.float)
                self.fa[slice:] = hdu1[5].data.astype(np.float)
                self.cp[slice:] = hdu1[4].data.astype(np.float)
                if len(self.observables) == 4:
                    self.ca[slice:] = hdu1[3].data.astype(np.float)

        # use dict of oifits info...
        self.info4oif_dict = self.oifinfo
        if self.verbose:
            for key in self.info4oif_dict.keys():
                print(key)
        self.ctrs = self.info4oif_dict['ctrs']
        self.bholes, self.bls = self._makebaselines()
        self.tholes, self.tuv = self._maketriples_all()
        self.qholes, self.quvw = self._makequads_all()


def table2oif(nh=None, oifinfo=None, fitsname=None, oifprefix='', datadir=None, verbose=False):
    """
    Input:
        oifinfo (str) Dict of info taken from raw file header, for use in oifits file
        fitsname (str) File name of FITS file containing table of observables
        oifprefix (str) Mnemonic prefix added to the output oifits filename (eg "ov6_")
                       The oifits filename is constructed from fields in the pickled dictionary
                       written by implaneia into oitxtdir directory.
        datadir (str)  Directory to write the oifits file in

        Typically the dir names are full path ("/User/.../"

    """
    nrm = ObservablesFromFITSTable(nh, oifinfo, fitsname, verbose=verbose) # read in the nrm observables
    dct = observable2dict(nrm, display=False) # populate Anthony's dictionary suitable for oifits.py
                                             # nrm_c defaults to false: do not calibrate, no cal star given
    oifits.save(dct, oifprefix=oifprefix, datadir=datadir, verbose=False)
    print('in directory %s' % datadir)
    return dct

def pipeline2oifits(main_fn, datadir, uncal_fn=None):
    """
    This function saves a single oifits file made from a JWST pipeline ami3
    product. It requires information stored in the _uncal.fits pipeline
    product, in addition to the main file. The uncal file is assumed to be
    in the same directory as the main file.

    Input:
        main_fn (str): filename of the pipeline ami3 output file
        datadir (str): directory where file and uncal file are kept
        uncal_fn (str): name of uncal file (level 1b product) used
                        to produce the ami3 product
    Returns:
        fulldict (dict): dictionary of all info required to save an oifits file

    """
    fn_fullpath = os.path.join(datadir, main_fn)
    if uncal_fn is None:
        # try to find a matching )uncal file
        if 'psf' in main_fn:
            uncal_globname = os.path.join(datadir,(main_fn.split('_psf-amiavg')[0] + '*_uncal.fits').replace('001001', '002001'))
        else:
            uncal_globname = os.path.join(datadir,main_fn.split('_amiavg')[0]+'*_uncal.fits')
        print('looking for files that match the pattern:',uncal_globname)
        # take the first mirage file that matches the pattern
        uncal_list = sorted(glob.glob(uncal_globname))
        if len(uncal_list) < 1:
            raise Exception("No suitable '_uncal' file found; input filename with 'uncal_fn=' in function call.")
        uncal_fn = uncal_list[0]

    # get header info to use from MIRAGE/MAST file
    pri_hdr = fits.getheader(uncal_fn,0)
    sci_hdr = fits.getheader(uncal_fn,1)
    filt = pri_hdr['FILTER']
    # Initialize the instrument data object for NIRISS
    instdata = InstrumentData.NIRISS(filt)
    # Use header to make info4oif_dict
    instdata.updatewithheaderinfo(pri_hdr,sci_hdr)
    oifitsdict = instdata.info4oif_dict
    print(oifitsdict['ctrs'])
    # since mirage file is not 2 or 3 dimensional, missed 'itime' keyword in dict.
    # skirting around that issue inelegantly for now...
    # take EFFEXPTM keyword from main file
    main_hdr = fits.getheader(fn_fullpath)
    oifitsdict['itime'] = main_hdr['EFFEXPTM'] # is this the right number to use here?
    numholes = 7
    pref = str(main_hdr['TARGPROP']+'_') # prefix to use with oifits filename
    outdir = 'oifits_out/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # Do the converting to OIFITS
    fulldict = table2oif(nh=numholes, oifinfo=oifitsdict, fitsname=fn_fullpath, oifprefix=pref, datadir=outdir)
    return fulldict

if __name__ == "__main__":
    indir = '/user/rcooper/Projects/NIRISS/AMI/build75testing/ami_sims/'
    targ_fn = 'jw01093001001_01101_amiavg.fits'  # AB-Dor simulation (target)
    calib_fn = 'jw01093001001_01101_psf-amiavg.fits'  # HD-37093 simulation (calibrator)
    pipeline2oifits(targ_fn, indir)
    pipeline2oifits(calib_fn, indir)


