#! /usr/bin/env python

"""
Reads ImPlaneIA output text of [CPs,phases,apmlitudes]_nn.txt in one directory
Writes them out into oifits files in the same directory.

Chunkwide averaging might be a later development for real data.

anand@stsci.edu started 2019 09 26 
anand@stsci.edu beta 2019 12 04

"""

import glob
import os
import pickle

import numpy as np
from astropy.time.core import Time
from astropy.io import fits
from matplotlib import pyplot as plt
from munch import munchify as dict2class
from scipy.special import comb
from scipy import stats
import copy

from itertools import combinations

import nrm_analysis.misctools.oifits as oifits



plt.close('all')


class ObservablesFromText():
    """
        anand@stsci.edu 2019_09_27
    """

    def __init__(self, nh, txtpath=None,
                 oifpath=None,
                 observables=("phases", "amplitudes", "CPs", "t3amps", "CAs", "q4phases", "fringepistons"),
                 oifinfofn='info4oif_dict.pkl',
                 angunit="radians",
                 verbose=True):
        """
        Methods: 
            readtxtdata(): read in implania to internal arrays
            showdata(): print out data at user-specified precision

        Input: 
            nh: number of holes in mask (int)
            textpaxt: directory where fringe observables' .txt files are stored (str)
            oifpath: directory (str).  If omitted, oifits goes to same directory as textpath
            nslices: 1 for eg unpolarized single filter observation, 
                     2 or more for polarized observations, 
                     many for IFU or repeated integration observations (int)
            observables: If ("phases", "amplitudes", "CPs", "CAs") for example - ImPlaneIA nomenclature
                       then the files need to be named: "/phases_{0:02d}.txt".format(slc)
                       Three or four quantities (omitting CA is optional)
                       Order is relevant... pha, amp, cp [, ca]
                       Implaneia txt data will be in txtpath/*.txt
            oifinfofn: default 'info4oif_dict.pkl' suits ImPlaneIA
                      Pickle file of info oifits writers might need, in a dictionary
                      Located in the same dir as text observable files.  Only one for all slices...
                      Implaneia writes this dictionary out with the default name.
            If you want to trim low and/or high ends of eg IFU spectral observables trim them
            on-the-fly before calling this routine.

            ImPlaneIA saves fp cp in RADIANS.  Here we convert to DEGREES immediately on reading in txt files,
            so all OIFITS output from implaneia is saved in DEGREES - 2020.10.18

            Units: as SI as possible.

        """

        if verbose: print("ObservablesFromText: typical regex: {0:s}*.txt:\n", txtpath)
        print("ObservablesFromText: typical regex: {0:s}*.txt :\n", txtpath)
        self.txtpath = txtpath
        self.oifpath = oifpath
        self.verbose = verbose
        self.observables = observables
        self.oifinfofn = oifinfofn
        # Assume same number of observable output files for each observable.  
        # Eg 1 slice or NINT slices cube output from eg implaneia.
        # Each image analyzed has a phases, an amplitudes, ... txt output file in this txtdir.
        # Each file might contain different numbers of individual quantities
        if verbose: print('Example of txt file pattern:  txtpath/{0:s}*.txt'.format(self.observables[0]))
        #   - yes many fringes, more cp's, and so on.
        self.nslices = len(
            glob.glob(self.txtpath+'/{0:s}*.txt'.format(self.observables[0])))
        if verbose: print("misctools.implane2oifits: slices' observables text filesets found:", self.nslices)

        self.nh = nh
        self.nbl = int(comb(self.nh, 2))
        self.ncp = int(comb(self.nh, 3))
        self.nca = int(comb(self.nh, 4))
        # arrays of observables, (nslices,nobservables) shape.
        self.fp = np.zeros((self.nslices, self.nbl))
        self.fa = np.zeros((self.nslices, self.nbl))
        self.cp = np.zeros((self.nslices, self.ncp))
        if len(self.observables) > 3:
            self.t3amp = np.zeros((self.nslices, self.ncp))
            self.ca = np.zeros((self.nslices, self.nca))
            self.q4phi = np.zeros((self.nslices, self.nca))
            self.pistons = np.zeros((self.nslices, self.nh))
        self.angunit = angunit
        if verbose:
            print("ImPlaneIA text output angle unit: %s" % angunit)

        if angunit == 'radians':
            print("Will convert all angular quantities to degrees for saving")
            self.degree = 180.0 / np.pi
        else:
            self.degree = 1

        self._readtxtdata()
        if self.verbose:
            self._showdata()

    def _makequads_all(self):
        """ returns int array of quad hole indices (0-based), 
            and float array of three uvw vectors in all quads
        """
        nholes = self.ctrs_eqt.shape[0]
        qlist = []
        for i in range(nholes):
            for j in range(nholes):
                for k in range(nholes):
                    for q in range(nholes):
                        if i < j and j < k and k < q:
                            qlist.append((i, j, k, q))
        qarray = np.array(qlist).astype(np.int32)
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
            uvwlist.append((self.ctrs_eqt[quad[0]] - self.ctrs_eqt[quad[1]],
                            self.ctrs_eqt[quad[1]] - self.ctrs_eqt[quad[2]],
                            self.ctrs_eqt[quad[2]] - self.ctrs_eqt[quad[3]]))
        if self.verbose:
            print(qarray.shape, np.array(uvwlist).shape)
        return qarray, np.array(uvwlist)

    def _maketriples_all(self):
        """ returns int array of triple hole indices (0-based), 
            and float array of two uv vectors in all triangles
        """
        nholes = self.ctrs_eqt.shape[0]
        tlist = []
        for i in range(nholes):
            for j in range(nholes):
                for k in range(nholes):
                    if i < j and j < k:
                        tlist.append((i, j, k))
        tarray = np.array(tlist).astype(np.int32)
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
            uvlist.append((self.ctrs_eqt[triple[0]] - self.ctrs_eqt[triple[1]],
                           self.ctrs_eqt[triple[1]] - self.ctrs_eqt[triple[2]]))
        # print(len(uvlist), "uvlist", uvlist)
        if self.verbose:
            print(tarray.shape, np.array(uvlist).shape)
        return tarray, np.array(uvlist)

    def _makebaselines(self):
        """
        ctrs_eqt (nh,2) in m
        returns np arrays of eg 21 baselinenames ('0_1',...), eg (21,2) baselinevectors (2-floats)
        in the same numbering as implaneia
        """
        nholes = self.ctrs_eqt.shape[0]
        blist = []
        for i in range(nholes):
            for j in range(nholes):
                if i < j:
                    blist.append((i, j))
        barray = np.array(blist).astype(np.int32)
        # blname = []
        bllist = []
        for basepair in blist:
            # blname.append("{0:d}_{1:d}".format(basepair[0],basepair[1]))
            baseline = self.ctrs_eqt[basepair[0]] - self.ctrs_eqt[basepair[1]]
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
              self.fp*self.degree, "\n")
        print(self.fa.shape, "fa:\n", self.fa, "\n")

        print(self.cp.shape, "cp (degrees, but stored internally in radians):\n",
              self.cp*self.degree, "\n")
        if len(self.observables) == 4:
            print(self.ca.shape, "ca:\n", self.ca, "\n")

        print("hole centers array shape:", self.ctrs_eqt.shape)

        print(len(self.bholes), "baseline hole indices\n", self.bholes)
        print(self.bls.shape, "baselines:\n", self.bls)

        print(self.tholes.shape, "triple hole indices:\n", self.tholes)
        print(self.tuv.shape, "triple uv vectors:\n", self.tuv)

        print(self.qholes.shape, "quad hole indices:\n", self.qholes)
        print(self.quvw.shape, "quad uvw vectors:\n", self.quvw)

    def _readtxtdata(self):
        # to only be called from init
        # loop through all the requested observables,
        # read in the exposure slices of a data cube
        # Incoming imia text files' angles radians, want degrees in oifs

        # set up files to read
        # file name for each exposure (slice) in an image cube with nslices exposures
        fnheads = []
        if self.verbose:
            print("\tfile names that are being looked for:")
        for obsname in self.observables:
            # ImPlaneIA-specific filenames
            fnheads.append(self.txtpath+"/"+obsname+"_{0:02d}.txt")
            if self.verbose:
                print("\t"+fnheads[-1])

        # load from text into data arrays:
        for slice in range(self.nslices):
            # Sydney oifits prefers degrees 2020.10.17
            self.fp[slice:] = np.rad2deg(np.loadtxt(fnheads[0].format(slice)))# * 180.0 / np.pi
            self.fa[slice:] = np.loadtxt(fnheads[1].format(slice))
            self.cp[slice:] = np.rad2deg(np.loadtxt(fnheads[2].format(slice)))# * 180.0 / np.pi
            # Do the same to-degrees conversion with segment phases when we get to them!
            if len(self.observables) > 3: # expecting CAs, fringepistons
                self.t3amp[slice:] = np.loadtxt(fnheads[3].format(slice)) # triple product amplitudes
                self.ca[slice:] = np.loadtxt(fnheads[4].format(slice)) # closure (quad) amplitudes
                self.q4phi[slice:] = np.rad2deg(np.loadtxt(fnheads[5].format(slice))) # quad phases in deg
                self.pistons[slice:] = np.rad2deg(np.loadtxt(fnheads[6].format(slice)))  # segment pistons in deg

        # read in pickle of the info oifits might need...
        pfd = open(self.txtpath+'/'+self.oifinfofn, 'rb')
        self.info4oif_dict = pickle.load(pfd)
        if self.verbose:
            for key in self.info4oif_dict.keys():
                print(key)
        pfd.close()
        self.ctrs_eqt = self.info4oif_dict['ctrs_eqt'] # mask centers in equatorial coordinates
        self.ctrs_inst = self.info4oif_dict['ctrs_inst'] # as-built instrument mask centers
        self.pa = self.info4oif_dict['pa']

        """   seexyz.py
        Sydney oifitx oi_array:
            Found oi_array
            ColDefs(
            name = 'TEL_NAME'; format = '16A'
            name = 'STA_NAME'; format = '16A'
            name = 'STA_INDEX'; format = '1I'
            name = 'DIAMETER'; format = '1E'; unit = 'METERS'
            name = 'STAXYZ'; format = '3D'; unit = 'METERS'
            name = 'FOV'; format = '1D'; unit = 'ARCSEC'
            name = 'FOVTYPE'; format = '6A'
            )
            <class 'numpy.ndarray'>
           [[ 0.      -2.64     0.     ]
            [-2.28631  0.       0.     ]
            [ 2.28631 -1.32     0.     ]
            [-2.28631  1.32     0.     ]
            [-1.14315  1.98     0.     ]
            [ 2.28631  1.32     0.     ]
            [ 1.14315  1.98     0.     ]]
            implaneia flips x and y, and switches sign on x 
        """
        self.bholes, self.bls = self._makebaselines()
        self.tholes, self.tuv = self._maketriples_all()
        self.qholes, self.quvw = self._makequads_all()


def Plot_observables(tab, vmin=0, vmax=1.1, cmax=180, unit_cp='deg', display=False):
    cp = tab.cp

    if unit_cp == 'rad':
        conv_cp = np.pi/180.
        h1 = np.pi
    else:
        conv_cp = 1
        h1 = np.rad2deg(np.pi)

    cp_mean = np.mean(tab.cp, axis=0)*conv_cp
    cp_med = np.median(tab.cp, axis=0)*conv_cp

    Vis = tab.fa
    Vis_mean = np.mean(Vis, axis=0)
    Vis_med = np.median(Vis, axis=0)

    target = tab.info4oif_dict['objname']

    cmin = -cmax*conv_cp
    if display:
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Uncalibrated Vis. (%s)' % target)
        plt.plot(Vis.transpose(), 'gray', alpha=.2)
        plt.plot(Vis_mean, 'k--', label='Mean')
        plt.plot(Vis_med, linestyle='--', color='crimson', label='Median')
        plt.xlabel('Index', color='dimgray', fontsize=12)
        plt.ylabel(r'Vis.', color='dimgray', fontsize=12)
        plt.ylim([vmin, vmax])
        plt.legend(loc='best')

        plt.subplot(1, 2, 2)
        plt.title('Uncalibrated CP (%s)' % target)
        plt.plot(cp.transpose(), 'gray', alpha=.2)
        plt.plot(cp_mean, 'k--', label='Mean')
        plt.plot(cp_med, linestyle='--', color='crimson', label='Median')
        plt.xlabel('Index', color='dimgray', fontsize=12)
        plt.ylabel('CP [%s]' % unit_cp, color='dimgray', fontsize=12)
        plt.hlines(h1, 0, len(cp_mean),
                   lw=1, color='k', alpha=.2, ls='--')
        plt.hlines(-h1, 0, len(cp_mean),
                   lw=1, color='k', alpha=.2, ls='--')
        plt.ylim([cmin, cmax])
        plt.legend(loc='best')
        plt.tight_layout()
        return fig
    else:
        return

def rotate_matrix(cov_mat, theta):
    c, s = np.cos(theta), np.sin(theta)
    R_mat = [[c, -s],
             [s, c]]
    # coordinate rotation from real/imaginary to absolute value/phase (modulus/argument)
    cv_rotated = np.linalg.multi_dot([np.transpose(R_mat), cov_mat, R_mat])
    return cv_rotated

def average_observables2(nrm, averfunc):
    """ 
    Convert visamp, visphase arrays to complex visibilities arrays for averaging cv's
    Calculate covariance matrices between fringe amplitudes and fringe phases, 
    and between closure amplutides and closure phases (as well as variance of each).
    Convert r, theta (modulus, phase) to x,y. Calculate cov(x,y). Rotate resulting
    2x2 matrix back to r, theta. 

    https://www.probabilitycourse.com/chapter5/5_3_1_covariance_correlation.php

     """

    """
        input: nrm: ObservablesFromText instance
        input: averfunc: np.median or np.mean should be passed.
        Incoming angular quantities start in DEGREES, calculations should be done in RADIANS

        Incoming values:
        nrm.nh is number of holes 
        nrm.fa is fringe amplitude
        nrm.fp is fringe phase /deg
        nrm.cp is closure phases /deg
        nrm.t3amp is triple product amplitudes
        nrm.q4phi is quad phases /deg
    """
    # we still need to set up empty arrays for all observables and then populate them?
    # are we sigma-clipping the averages?
    # are we averaging complex numbers or not?

    avg_fa = averfunc(nrm.fa, axis=0)
    avg_fp = averfunc(nrm.fp, axis=0)
    
    # WIP

def observable_covariances(nrm):
    """
    input: nrm: ObservablesFromText instance
    """
    # loop over 21 baselines
    cov_mat_fringes = []
    # these currently operate on all slices. other one operated on already-averaged slices
    for bl in np.arange(nrm.nbl):
        fringeamps = nrm.fa[:,bl]
        fringephases = nrm.fp[:,bl]
        covmat = cov_r_theta(fringeamps, fringephases, averfunc)
        cov_mat_fringes.append(covmat)

    cov_mat_triples = []
    for triple in np.arange(nrm.ncp):
        tripamp = nrm.t3amp[:,triple]
        triphase = nrm.cp[:,triple]
        covmat = cov_r_theta(tripamp, triphase, averfunc)
        cov_mat_triples.append(covmat)

    cov_mat_quads = []
    for quad in np.arange(nrm.nca):
        quadamp = nrm.ca[:,quad]
        quadphase = nrm.quadphase[:,quad]
        covmat = cov_r_theta(quadamp, quadphase, averfunc)
        cov_mat_quads.append(covmat)

    # covmats to be written to oifits. store in nrm object?

    return np.array(cov_mat_fringes), np.array(cov_mat_triples), np.array(cov_mat_quads)


def cov_r_theta(rr, theta, averfunc):
    """
    rr: complex number modulus, array 
    theta: complex number phase, array
    averfunc: np.median or np.mean
    """
    xx = rr * np.cos(theta)
    yy = rr * np.sin(theta)
    cov_mat_xy = np.cov(xx, yy)
    # print('cov mat shape', cov_mat_xy.shape)
    # print('theta shape', theta.shape)
    cov_mat_r_theta = rotate_matrix(cov_mat_xy, averfunc(theta))
    return cov_mat_r_theta


def average_observables(nrm, averfunc):
    """ Convert visamp, visphase arrays to complex visibilities arrays for averaging cv's """

    """
        input: nrm: ObservablesFromText instance
        input: averfunc: np.median or np.mean should be passed.
        Incoming angular quantities start in DEGREES, calculations done in RADIANS
        modelled on SAMpip by Joel Sanchez Bermudez (see his reduce_SAM_poly2.py)

        Incoming values:
        nrm.nh is number of holes 
        nrm.fa is fringe amplitude
        nrm.fp is fringe phase/radians  """

    # change "mean" to "avg" in variable names

    # put in JSB notation
    nh = nrm.nh
    nbl = nrm.nbl
    ncp = nrm.ncp
    nca = nrm.nca

    data_visamp = np.zeros([nbl])
    data_visamperr = np.zeros([nbl])
    data_visphi = np.zeros([nbl])
    data_visphierr = np.zeros([nbl])
    data_v2 = np.zeros([nbl])
    data_v2err = np.zeros([nbl])
    data_t3amp = np.zeros([ncp])
    data_t3phi = np.zeros([ncp])
    data_t3phierr = np.zeros([ncp])
    data_t3amperr = np.zeros([ncp])
    data_ca = np.zeros([nca])       # Anand added
    data_caerror = np.zeros([nca])  # Anand added

    # First get nbl averages and stats of complex visibilities
    cv = nrm.fa * np.exp(1j*np.deg2rad(nrm.fp)) # array shape is [nslices, nbl] # CONVERTED DEG TO RAD HERE
    cv_mean = averfunc(cv, axis=0) # now there are nbl cv's

    # calculate CV stats for each baseline, averaging over slices (integrations)
    cv_real_var = np.var(cv.real, axis=0) / nrm.nslices 
    cv_im_var = np.var(cv.imag , axis=0) / nrm.nslices
    # cv_real_std = np.std(cv.real, axis=0) #/ np.sqrt(nrm.nslices)
    # cv_im_std = np.std(cv.imag, axis=0) #/ np.sqrt(nrm.nslices)

    cv_mod_mean = np.abs(cv_mean)
    cv_arg_mean = np.angle(cv_mean)

    # calculate average fringe amps and phases, errors, considering covariances
    fringe_cov_mat_list = []
    for bl in np.arange(nbl):
        fringe_cov_mat = np.cov(np.stack((cv.real[:,bl],cv.imag[:,bl]),axis=0))
        fringe_cov_mat_list.append(fringe_cov_mat)

        # coordinate rotation from real/imaginary to absolute value/phase (modulus/argument)
        cv_rotated = rotate_matrix(fringe_cov_mat, cv_arg_mean[bl])
        # print('rotation angle (rad)', bl, cv_arg_mean[bl])
        data_visamp[bl] = cv_mod_mean[bl]
        data_visphi[bl] = cv_arg_mean[bl]

        data_visamperr[bl] = np.sqrt(cv_rotated[0, 0])
        data_visphierr[bl] = np.arctan2(np.sqrt(cv_rotated[1, 1]), cv_mod_mean[bl])

        data_v2[bl] = cv_mean[bl].real ** 2 + cv_mean[bl].imag ** 2 - cv_real_var[bl] - cv_im_var[bl]
        data_v2err[bl] = 2 * data_v2[bl] * np.sqrt(cv_rotated[0, 0])


    triple_idx = nrm.tholes

    t3 = np.zeros([cv.shape[0], int(ncp)], dtype=complex)
    t3_phase = np.zeros([cv.shape[0], int(ncp)])
    t3_amp = np.zeros([cv.shape[0], int(ncp)])


    #for nslc in np.arange(t3.shape[0]): # nslices 
    for ncp in np.arange(t3.shape[1]): # n closure quantities (35)
        t3[:, ncp] = (cv[:, triple_idx[ncp, 0]] * 
                      cv[:, triple_idx[ncp, 1]] * 
              np.conj(cv[:, triple_idx[ncp, 2]]))
        t3_amp[:,ncp] = np.abs(t3[:,ncp])
        t3_phase[:,ncp] = np.angle(t3[:,ncp])

    t3_mean = averfunc(t3, axis=0)
    t3_mod_mean = np.abs(t3_mean)
    t3_arg_mean = np.angle(t3_mean)

    t3_real_var = np.var(t3.real, axis=0) / nrm.nslices
    t3_im_var = np.var(t3.imag, axis=0) / nrm.nslices
    t3_real_std = np.std(t3.real, axis=0) / np.sqrt(nrm.nslices)
    t3_im_std = np.std(t3.imag, axis=0) / np.sqrt(nrm.nslices)

    for tri in np.arange(ncp):
        cov_mat = [[t3_real_var[tri], t3_real_std[tri] * t3_im_std[tri]], 
                   [t3_real_std[tri] * t3_im_std[tri], t3_im_var[tri]]]
        c,s = np.cos(t3_arg_mean[tri]), np.sin(t3_arg_mean[tri])
        R_mat = [[c, -s], 
                 [s, c]]
        # coordinate rotation from real/imag axes to amplitude/phase axes in triple product space
        t3_rotated = np.linalg.multi_dot([np.transpose(R_mat), cov_mat, R_mat])
        data_t3amp[tri] = t3_mod_mean[tri]
        data_t3phi[tri] = t3_arg_mean[tri]
        data_t3amperr[tri] = np.sqrt(t3_rotated[0, 0])
        data_t3phierr[tri] = np.rad2deg(np.arctan(np.sqrt(t3_rotated[1, 1]) / t3_mod_mean[tri]))
        
    data_visphi = np.rad2deg((np.deg2rad(data_visphi) + np.pi) % (2 * np.pi) - np.pi) # in degrees
    data_t3phi = np.rad2deg(((np.deg2rad(data_t3phi) + np.pi) % (2 * np.pi)) - np.pi)


    #      vis2,     e_vis2,      visamp,       e_visamp,        visphi,       e_visphi,        cp,          e_cp,       cpamp,       e_cpamp,
    return data_v2, data_v2err, data_visamp, data_visamperr, data_visphi, data_visphierr, data_t3phi, data_t3phierr, data_t3amp, data_t3amperr


    
def populate_NRM(nrm_t, method='med'):
    """ 
    modelled on calib_NRM() but no calibration done because it's for a single object.
    Instead it just populates the appropriate dictionary.  
    So nomenclature looks funny with _5, etc., 
    Funny-looking clumsy straight handoffs to internal variable nmaes,...
    # RAC 3/3021
    If method='multi', preserve observables in each slice (integration) in the output class.
    Multi-slice observable arrays will have read-in shape (len(observable),nslices).
    Errors of multi-slice observables will be all zero (for now)
    Otherwise, take median or mean (assumed if method not 'med' or 'multi').

    nrm_t has angles in radians.  Convert to complex visibilities for averaging.

    """
    visamp_in = nrm_t.fa
    visphi_in = nrm_t.fp
    vis2_in = visamp_in**2
    cp_in = nrm_t.cp
    t3amp_in = nrm_t.t3amp
    ca_in = nrm_t.ca
    q4phi_in = nrm_t.q4phi
    pistons_in = nrm_t.pistons

    if method == 'multi': # no averaging
        vis2 = vis2_in.T
        e_vis2 = np.zeros(vis2.shape)
        visamp = visamp_in.T
        e_visamp = np.zeros(visamp.shape)
        visphi = visphi_in.T
        e_visphi = np.zeros(visphi.shape)
        cp = cp_in.T
        e_cp = np.zeros(cp.shape)
        t3amp = t3amp_in.T
        e_t3amp = np.zeros(t3amp.shape)
        ca = ca_in.T
        e_ca = np.zeros(ca.shape)
        q4phi = q4phi_in.T
        e_q4phi = np.zeros(q4phi.shape)
        pist = pistons_in.T
        e_pist = np.zeros(pist.shape)
    elif method == 'med': # average over complex quantities
        #vis2, e_vis2, visamp, e_visamp, visphi, e_visphi, cp, e_cp, cpamp, e_cpamp =  average_observables(nrm_t, np.median)
        vis2, e_vis2, visamp, e_visamp, visphi, e_visphi, cp, e_cp, t3amp, e_t3amp, ca, e_ca, q4phi, e_q4phi =  average_observables2(nrm_t, np.median)
        pist = np.median(pistons_in, axis=0)
        e_pist = np.std(pistons_in, axis=0)
    else: # average over complex quantities
        #vis2, e_vis2, visamp, e_visamp, visphi, e_visphi, cp, e_cp, cpamp, e_cpamp =  average_observables(nrm_t, np.mean)
        vis2, e_vis2, visamp, e_visamp, visphi, e_visphi, cp, e_cp, t3amp, e_t3amp, ca, e_ca, q4phi, e_q4phi =  average_observables2(nrm_t, np.mean) 
        pist = np.mean(pistons_in, axis=0)
        e_pist = np.std(pistons_in, axis=0)


    output = {'vis2': vis2,
              'e_vis2': e_vis2,
              'visamp': visamp,
              'e_visamp': e_visamp,
              'visphi': visphi,
              'e_visphi': e_visphi,
              'cp': cp,
              'e_cp': e_cp,
              't3amp': t3amp,
              'e_t3amp': e_t3amp,
              'ca': ca,
              'e_ca': e_ca,
              'q4phi': q4phi,
              'e_q4phi': e_q4phi
              'pist': pist,
              'e_pist': e_pist
              }

    return dict2class(output)


def observable2dict(nrm, multi=False, display=False):
    """ 
    Convert nrm data in an Observable loaded with `ObservablesFromText` into 
    a dictionary compatible with oifits.save and oifits.show function.

    nrm:   an ObservablesFromText instance, treated as a target if nrm_c=None.
           Note input nrm object angles are radians

    multi:  Bool. If true, do not take mean or median of slices
            (preserve separate integrations)
    """

    info4oif = nrm.info4oif_dict
    ctrs_inst = info4oif['ctrs_inst']
    t = Time('%s-%s-%s' %
             (info4oif['year'], info4oif['month'], info4oif['day']), format='fits')
    ins = info4oif['telname']
    filt = info4oif['filt']

    wl, e_wl = oifits.GetWavelength(ins, filt)

    bls = nrm.bls
    # Index 0 and 1 reversed to get the good u-v coverage (same fft)
    ucoord = bls[:, 1]
    vcoord = bls[:, 0]

    D = 6.5  # Primary mirror display

    theta = np.linspace(0, 2*np.pi, 100)

    x = D/2. * np.cos(theta)  # Primary mirror display
    y = D/2. * np.sin(theta)

    bl_vis = ((ucoord**2 + vcoord**2)**0.5)

    tuv = nrm.tuv
    v1coord = tuv[:, 0, 0]
    u1coord = tuv[:, 0, 1]
    v2coord = tuv[:, 1, 0]
    u2coord = tuv[:, 1, 1]
    u3coord = -(u1coord+u2coord)
    v3coord = -(v1coord+v2coord)

    bl_cp = []
    n_bispect = len(v1coord)
    for k in range(n_bispect):
        B1 = np.sqrt(u1coord[k] ** 2 + v1coord[k] ** 2)
        B2 = np.sqrt(u2coord[k] ** 2 + v2coord[k] ** 2)
        B3 = np.sqrt(u3coord[k] ** 2 + v3coord[k] ** 2)
        bl_cp.append(np.max([B1, B2, B3]))  # rad-1
    bl_cp = np.array(bl_cp)

    flagVis = [False] * nrm.nbl
    flagT3 = [False] * nrm.ncp

    if multi == True:
        nrmd2c = populate_NRM(nrm, method='multi') # RAC 2021
    else:
        nrmd2c = populate_NRM(nrm, method='med')

    dct = {'OI_VIS2': {'VIS2DATA': nrmd2c.vis2,
                       'VIS2ERR': nrmd2c.e_vis2,
                       'UCOORD': ucoord,
                       'VCOORD': vcoord,
                       'STA_INDEX': nrm.bholes,
                       'MJD': t.mjd,
                       'INT_TIME': info4oif['itime'],
                       'TIME': 0,
                       'TARGET_ID': 1,
                       'FLAG': flagVis,
                       'BL': bl_vis
                       },

           'OI_VIS': {'TARGET_ID': 1,
                      'TIME': 0,
                      'MJD': t.mjd,
                      'INT_TIME': info4oif['itime'],
                      'VISAMP': nrmd2c.visamp,
                      'VISAMPERR': nrmd2c.e_visamp,
                      'VISPHI': nrmd2c.visphi,
                      'VISPHIERR': nrmd2c.e_visphi,
                      'UCOORD': ucoord,
                      'VCOORD': vcoord,
                      'STA_INDEX': nrm.bholes,
                      'FLAG': flagVis,
                      'BL': bl_vis
                      },

           'OI_T3': {'TARGET_ID': 1,
                     'TIME': 0,
                     'MJD': t.mjd,
                     'INT_TIME': info4oif['itime'],
                     'T3PHI': nrmd2c.cp,
                     'T3PHIERR': nrmd2c.e_cp,
                     'T3AMP': nrmd2c.t3amp,
                     'T3AMPERR': nrmd2c.e_t3amp,
                     'U1COORD': u1coord,
                     'V1COORD': v1coord,
                     'U2COORD': u2coord,
                     'V2COORD': v2coord,
                     'STA_INDEX': nrm.tholes,
                     'FLAG': flagT3,
                     'BL': bl_cp
                     },

            'OI_Q4': {'TARGET_ID': 1,
                     'TIME': 0,
                     'MJD': t.mjd,
                     'INT_TIME': info4oif['itime'],
                     'Q4PHI': nrmd2c.q4phi,
                     'Q4PHIERR': nrmd2c.e_q4phi,
                     'CA': nrmd2c.ca,
                     'CAERR': nrmd2c.e_ca,
                     'U1COORD': u1coord,
                     'V1COORD': v1coord,
                     'U2COORD': u2coord,
                     'V2COORD': v2coord,
                     'U3COORD': u3coord,
                     'V3COORD': v3coord,
                     'STA_INDEX': nrm.tholes,
                     'FLAG': flagT3,
                     'BL': bl_cp # ??? get longest of quad baselines?
                     },

           'OI_WAVELENGTH': {'EFF_WAVE': wl,
                             'EFF_BAND': e_wl
                             },

           'info': {'TARGET': info4oif['objname'],
                    'CALIB': info4oif['objname'],
                    'OBJECT': info4oif['objname'],
                    'FILT': info4oif['filt'],
                    'INSTRUME': info4oif['instrument'],
                    'ARRNAME': info4oif['arrname'],
                    'MASK': info4oif['arrname'], # oifits.py looks for dct.info['MASK']
                    'MJD': t.mjd,
                    'DATE-OBS': t.fits,
                    'TELESCOP': info4oif['telname'],
                    'OBSERVER': 'UNKNOWN',
                    'INSMODE': info4oif['pupil'],
                    'PSCALE': info4oif['pscale_mas'],
                    'STAXY': info4oif['ctrs_inst'], # as-built mask hole coords
                    'ISZ': 77,  # size of the image needed (or fov)
                    'NFILE': 0,
                    'PA': info4oif['pa'],
                    'CTRS_EQT':info4oif['ctrs_eqt'], # mask hole coords rotated to equatotial
                    'PISTONS': nrmd2c.pist, # RAC 2021
                    'PIST_ERR': nrmd2c.e_pist
                    }
           }

    if display:
        plt.figure(figsize=(14.2, 7))
        plt.subplot(1, 2, 1)
        # Index 0 and 1 reversed to get the good u-v coverage (same fft)
        #lt.scatter(ctrs[:, 1], ctrs[:, 0], s=2e3, c='', edgecolors='navy')
        plt.scatter(ctrs[:, 1], ctrs[:, 0], s=2e3,       edgecolors='navy')
        #lt.scatter(-1000, 1000, s=5e1, c='',
        plt.scatter(-1000, 1000, s=5e1,      
                    edgecolors='navy', label='Aperture mask')
        plt.plot(x, y, '--', color='gray', label='Primary mirror equivalent')

        plt.xlabel('Aperture x-coordinate [m]')
        plt.ylabel('Aperture y-coordinate [m]')
        plt.legend(fontsize=8)
        plt.axis([-4., 4., -4., 4.])

        plt.subplot(1, 2, 2)
        #lt.scatter(ucoord, vcoord, s=1e2, c='', edgecolors='navy')
        plt.scatter(ucoord, vcoord, s=1e2,       edgecolors='navy')
        #lt.scatter(-ucoord, -vcoord, s=1e2, c='', edgecolors='crimson')
        plt.scatter(-ucoord, -vcoord, s=1e2,       edgecolors='crimson')

        plt.plot(0, 0, 'k+')
        plt.axis((D, -D, -D, D))
        plt.xlabel('Fourier u-coordinate [m]')
        plt.ylabel('Fourier v-coordinate [m]')
        plt.tight_layout()

        Plot_observables(nrm, display=display)
        #if nrm_c: Plot_observables(nrm_c=display)  # Plot calibrated or single object raw oifits data
    return dct


def oitxt2oif(nh=None, oitdir=None, oifn='', oifdir=None, verbose=False):
    """
    The interface routine called by implaneia's fit_fringes.
    Input: 
        oitdir (str) Directory where implaneia wrote the observables
                       observable files are named: CPs_nn.txt, amplitudes_nn.txt, and so on
                       02d format numbers, 00 start, number the slices in  the 
                       image 3D datacube processed by implaneia.
        oifn (str)     oifits file root name specified bt the driver (eg FitFringes.fringefitter())
        oifdir (str)   Directory to write the oifits file in

        Typically the dir names are full path ("/User/.../"

    Used to be 
    def implane2oifits2(OV, objecttextdir_c, objecttextdir_t, oifprefix, datadir):
    which calibrated a target with a calibrator and wrote single oifits file.
    Converted here to only write one oifits file to disk, including stats
    for the object's observables
    """
    nrm = ObservablesFromText(nh, oitdir, verbose=verbose) # read in the nrm observables
    dct = observable2dict(nrm, display=False) # populate Anthony's dictionary suitable for oifits.py
                                             # nrm_c defaults to false: do not calibrate, no cal star given
    oifits.save(dct, filename=oifn, datadir=oifdir, verbose=False)
    # save multi-slice fits
    dct_multi = observable2dict(nrm, multi=True, display=False)
    oifits.save(dct_multi, filename='multi_'+oifn, datadir=oifdir, verbose=False)
    print('\n in oifits directory {0:s}'.format(oifdir))
    return dct

def calib_dicts(dct_t, dct_c):
    """
    Takes two dicts from OIFITS files, such as those read with oifits.load()
    Calibrates closure phases and fringe amplitudes of target by calibrator
    by subtracting closure phases of calibrator from those of target,
    and dividing fringe amps of target by fringe amps of calibrator
    Input:
        dct_t (dict): oifits-compatible dictionary of target observables/info
        dct_c (dict): oifits-compatible dictionary of calibrator observables/info
    Returns:
        calib_dict (dict): oifits-compatible dictionary of calibrated observables/info
    """
    # cp is closure phase
    # sqv is square visibility
    # va is visibility amplitude

    cp_out = dct_t['OI_T3']['T3PHI'] - dct_c['OI_T3']['T3PHI']
    sqv_out = dct_t['OI_VIS2']['VIS2DATA'] / dct_c['OI_VIS2']['VIS2DATA']
    va_out = dct_t['OI_VIS']['VISAMP'] / dct_c['OI_VIS']['VISAMP']
    # now using correct propagation of error for multiplication/division
    # which assumes uncorrelated Gaussian errors (not true...?)    
    cperr_t = dct_t['OI_T3']['T3PHIERR']
    cperr_c = dct_c['OI_T3']['T3PHIERR']
    sqverr_c = dct_t['OI_VIS2']['VIS2ERR']
    sqverr_t = dct_c['OI_VIS2']['VIS2ERR']
    vaerr_t = dct_t['OI_VIS']['VISAMPERR']
    vaerr_c = dct_c['OI_VIS']['VISAMPERR']
    cperr_out = np.sqrt(cperr_t**2. + cperr_c**2.)
    sqverr_out = sqv_out * np.sqrt((sqverr_t/dct_t['OI_VIS2']['VIS2DATA'])**2. + (sqverr_c/dct_c['OI_VIS2']['VIS2DATA'])**2.)
    vaerr_out = va_out * np.sqrt((vaerr_t/dct_t['OI_VIS']['VISAMP'])**2. + (vaerr_c/dct_c['OI_VIS']['VISAMP'])**2.)

    # copy the target dict and modify with the calibrated observables
    calib_dict = dct_t.copy()
    calib_dict['OI_T3']['T3PHI'] = cp_out
    calib_dict['OI_VIS2']['VIS2DATA'] = sqv_out
    calib_dict['OI_VIS']['VISAMP'] = va_out
    calib_dict['OI_T3']['T3PHIERR'] = cperr_out
    calib_dict['OI_VIS2']['VIS2ERR'] = sqverr_out
    calib_dict['OI_VIS']['VISAMPERR'] = vaerr_out
    # preserve the name of the calibrator star
    calib_dict['info']['CALIB'] = dct_c['info']['OBJECT']
    # include pistons and piston errors from target and calibrator
    # if old files, raw oifits won't have any pistons
    if ('PISTONS' in dct_t['OI_ARRAY']) & ('PISTONS' in dct_c['OI_ARRAY']):
        pistons_t = dct_t['OI_ARRAY']['PISTONS']
        pisterr_t = dct_t['OI_ARRAY']['PIST_ERR']
        pistons_c = dct_c['OI_ARRAY']['PISTONS']
        pisterr_c = dct_c['OI_ARRAY']['PIST_ERR']
        # sum in quadrature errors from target and calibrator pistons (only if both oifits contain pistons)
        pisterr_out = np.sqrt(pisterr_t**2 + pisterr_c**2)
        # populate calibrated dict with pistons 
        calib_dict['OI_ARRAY']['PISTON_T'] = pistons_t
        calib_dict['OI_ARRAY']['PISTON_C'] = pistons_c
        calib_dict['OI_ARRAY']['PIST_ERR'] = pisterr_out
    # remove plain "pistons" key from dict
    if 'PISTONS' in calib_dict['OI_ARRAY']:
        del calib_dict['OI_ARRAY']['PISTONS']

    return calib_dict



def calibrate_oifits(oif_t, oif_c, oifn=None, oifdir=None, **kwargs):
    """
    Take an OIFITS file of the target and an OIFITS file of the calibrator and
    produce a single normalized OIFITS file.
    Input:
        oif_t (str): file name of the target OIFITS file
        oif_c (str): file name of the calibrator OIFITS file
        oifn (str): calibrated oifits output name
        oifdir (str): Directory to write the oifits file in (default cwd)
    Returns:
        calibrated (dict): dict containing calibrated OIFITS information,
        calibrated oifits filename
    """
    # housekeeping:
    # construct an output filename from the input names if none is provided
    if oifn is None:
        bn_t = os.path.basename(oif_t).split('.oifits')[0]
        bn_c = os.path.basename(oif_c).split('.oifits')[0]
        oifn = bn_t + '_cal_' + bn_c + '.oifits'

    rfn = False # anand 2022.01.13 return calib dict as before
    if 'returnfilename' in kwargs: # anand 2022.01.13
        rfn = True # return tuple of calib dictionary as well as oif calibrated filename

    # backwards compatibility with old kwargs:
    if 'oifprefix' in kwargs:
        oifn = kwargs['oifprefix']+oifn # use the prefix + fn constructed from input
    if 'datadir' in kwargs:
        oifdir = kwargs['datadir']
    if oifdir is None:
        oifdir = './'
    # if name doesn't end in '.oifits', change it.
    if oifn[-7:] != '.oifits':
        if oifn[-5:] == '.fits':
            oifn = oifn.replace('.fits', '.oifits')
        else:
            oifn = oifn + '.oifits'
    # load in the nrm observables dict from each oifits
    targ = oifits.load(oif_t)
    calb = oifits.load(oif_c)
    # calibrate the target by the calibrator
    # this produces a single calibrated nrm dict
    calibrated = calib_dicts(targ, calb)

    oifits.save(calibrated, filename=oifn, datadir=oifdir)
    print('\n in oifits directory {0:s}'.format(oifdir))

    if rfn: return calibrated, os.path.join(oifdir, oifn)
    else: return calibrated



if __name__ == "__main__":

    ov_main = 3 # only used to create oifits filename prefix to help organize output
    moduledir = os.path.expanduser('~') + '/gitsrc/ImPlaneIA/'  # dirname of where you work

    # convert one file...
    oifn_t = "t_ov{:d}_".format(ov_main) # mnemonic supplied by driver... 
                                              # if you explore different ov's you can 
                                              # put 'ov%d' in prefix, and save to a directory of your choice.
    oitdir_t = moduledir + "/example_data/example_niriss/bin_tgt_oitxt/" # implaneia observables txt dir
    oifdir_t =  oitdir_t # could add a subdir but this writes the oifits into text output dir.
    dct = oitxt2oif(nh=7, oitdir=oitdir_t, 
                          oifn=oifn_t,
                          datadir=oifdir_t)
    # oifits.show(dct, diffWl=True)
    # plt.show()

    if 0:
        # then convert another file...
        oifn_c = "c_ov{:d}_".format(ov_main)
        oitdir_c = moduledir + "/example_data/example_niriss/bin_cal_oitxt"
        oifdir_c =  oitdir_c + '/Saveoifits/'
        # Convert all txt observables in oitdir to oifits file
        dct = oitxt2oif(nh=7, oitdir=oitdir_c, 
                              oifn=oifn_c,
                              datadir=oifdir_c)
        oifits.show(dct, diffWl=True)
        #plt.show()

