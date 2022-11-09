#! /usr/bin/env python

"""
InstrumentData Class -- defines data format, wavelength info, mask geometry

Instruments/masks supported:
NIRISS AMI
GPI, VISIR, NIRC2 removed - too much changed for the JWST NIRISS class
"""

# Standard Imports
import numpy as np
from astropy.io import fits
import os, sys, time
import copy
# Module imports
import synphot
# import stsynphot
# mask geometries, GPI, NIRISS, VISIR supported...
from nrm_analysis.misctools.mask_definitions import NRM_mask_definitions 
from nrm_analysis.misctools import utils
from nrm_analysis.misctools import lpl_ianc



um = 1.0e-6

# utility routines for InstrumentData classes

def show_cvsupport_threshold(instr):
    """ Show threshold for where 'splodge' data in CV space contains signal """
    print("InstrumentData: ", "cvsupport_threshold is: ", instr.cvsupport_threshold)
    print("InstrumentData: ", instr.cvsupport_threshold)

def set_cvsupport_threshold(instr, k, v):
    """ Set threshold for where 'splodge' data in CV space contains signal
        
    Parameters
    ----------
    instr: InstrumentData instance
    thresh: Threshold for the absolute value of the FT(interferogram).
            Normalize abs(CV = FT(a)) for unity peak, and define the support 
            of "good" CV when this is above threshold
    """
    
    instr.cvsupport_threshold[k] = v
    print("InstrumentData: ", "New cvsupport_threshold is: ", instr.cvsupport_threshold)



class NIRISS:
    def __init__(self, filt, 
                       objname="obj", 
                       src='A0V', 
                       chooseholes=None, 
                       affine2d=None, 
                       bandpass=None,
                       nbadpix=4,
                       usebp=True,
                       firstfew=None,
                       nspecbin=None,
                       **kwargs):
        """
        Initialize NIRISS class

        ARGUMENTS:

        kwargs:
        UTR
        Or just look at the file structure
        Either user has webbpsf and filter file can be read, or...
        chooseholes: None, or e.g. ['B2', 'B4', 'B5', 'B6'] for a four-hole mask
        filt:     Filter name string like "F480M"
        bandpass: None or [(wt,wlen),(wt,wlen),...].  Monochromatic would be e.g. [(1.0, 4.3e-6)]
                  Explicit bandpass arg will replace *all* niriss filter-specific variables with 
                  the given bandpass (src, nspecbin, filt), so you can simulate 21cm psfs through 
                  something called "F430M". Can also be synphot.spectrum.SourceSpectrum object.
        firstfew: None or the number of slices to truncate input cube to in memory,
                  the latter for fast developmpent
        nbadpix:  Number of good pixels to use when fixing bad pixels DEPRECATED
        usebp:    Convert to usedq during initialization
                  Internally this is changed to sellf.usedq = usebp immediately for code clarity
                  True (default) do not use DQ with DO_NOT_USE flag in input MAST data when
                  fitting data with model.  False: Assume no bad pixels in input
        noise:    standard deviation of noise added to perfect images to enable candid
                  plots without crashing on np.inf limits!  Image assumed to be in (np.float64) dn.
                  Suggested noise: 1e-6.
        src:      source spectral type string e.g. "A0V" OR user-defined synphot.spectrum.SourceSpectrum object
        nspecbin: Number of wavelength bins to use across the bandpass. Replaces deprecated `usespecbin` which 
                  set **number of wavelengths to into each bin**, not nbins.

        """

        self.verbose = False
        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]

        self.noise = None
        if "noise" in kwargs:
            self.noise = kwargs["noise"]
        if "usespecbin" in kwargs: # compatability with previous arg
            # but not really, usespecbin was binning factor, not number of bins
            nspecbin = kwargs["usespecbin"]
        # change how many wavelength bins will be used across the bandpass
        if nspecbin is None:
            nspecbin = 19

        self.lam_bin = nspecbin

        # src can be either a spectral type string or a user-defined synphot spectrum object
        if isinstance(src, synphot.spectrum.SourceSpectrum):
            print("Using user-defined synphot SourceSpectrum")

        if chooseholes:
            print("InstrumentData.NIRISS: ", chooseholes)
        self.chooseholes = chooseholes

    
        # USEBP is USEDQ in the rest of code - use
        self.usedq = usebp
        print("Fitting omits bad pixels (identified by DO_NOT_USE value in the DQ extension)")
        self.jwst_dqflags() # creates dicts self.bpval, self.bpgroup
        # self.bpexist set True/False if  DQ fits image extension exists/doesn't

        self.firstfew = firstfew
        if firstfew is not None: print("InstrumentData.NIRISS: analysing firstfew={:d} slices".format(firstfew))

        self.objname = objname

        self.filt = filt

        if bandpass is not None:
            print("InstrumentData.NIRISS: OVERRIDING BANDPASS WITH USER-SUPPLIED VALUES.")
            print("\t src, filt, nspecbin parameters will not be used")
            # check type of bandpass. can be synphot spectrum
            # if so, get throughput and wavelength arrays
            if isinstance(bandpass, synphot.spectrum.SpectralElement):
                wl, wt = bandpass._get_arrays(bandpass.waveset)
                self.throughput = np.array((wt,wl)).T
            else:
                self.throughput = np.array(bandpass)  # type simplification
        else:
            filt_spec = utils.get_filt_spec(self.filt)
            src_spec = utils.get_src_spec(src)
            # **NOTE**: As of WebbPSF version 1.0.0 filter is trimmed to where throughput is 10% of peak
            # For consistency with WebbPSF simultions, use trim=0.1
            self.throughput = utils.combine_src_filt(filt_spec, 
                                          src_spec, 
                                          trim=0.01, 
                                          nlambda=nspecbin,
                                          verbose=self.verbose, 
                                          plot=False)

        self.lam_c, self.lam_w = utils.get_cw_beta(self.throughput)

        if self.verbose: print("InstrumentData.NIRISS: ", self.filt, 
              ": central wavelength {:.4e} microns, ".format(self.lam_c/um), end="")
        if self.verbose: print("InstrumentData.NIRISS: ", "fractional bandpass {:.3f}".format(self.lam_w))

        self.wls = [self.throughput,] 

        if self.verbose: print("self.throughput:\n", self.throughput)

        # Wavelength info for NIRISS bands F277W, F380M, F430M, or F480M
        self.wavextension = ([self.lam_c,], [self.lam_w,])
        self.nwav=1 # these are 'slices' if the data is pure imaging integrations - 
        #             nwav is old nomenclature from GPI IFU data.  Refactor one day...
        #############################

        # only one NRM on JWST:
        self.telname = "JWST"
        self.instrument = "NIRISS"
        self.arrname = "jwst_g7s6c"  # implaneia mask set with this - unify to short form later 
        self.holeshape="hex"
        self.mask = NRM_mask_definitions(maskname=self.arrname, chooseholes=chooseholes, 
                                         holeshape=self.holeshape )

        # save affine deformation of pupil object or create a no-deformation object. 
        # We apply this when sampling the PSF, not to the pupil geometry.
        # This will set a default Ideal or a measured rotation, for example,
        # and include pixel scale changes due to pupil distortion.
        # Separating detector tilt pixel scale effects from pupil distortion effects is 
        # yet to be determined... see comments in Affine class definition.
        # AS AZG 2018 08 15 Ann Arbor
        if affine2d is None:
            self.affine2d = utils.Affine2d(mx=1.0,my=1.0, 
                                           sx=0.0,sy=0.0, 
                                           xo=0.0,yo=0.0, name="Ideal")
        else:
            self.affine2d = affine2d

        # finding centroid from phase slope only considered cv_phase data 
        # when cv_abs data exceeds this cvsupport_threshold.  
        # Absolute value of cv data normalized to unity maximum
        # for the threshold application.
        # Data reduction gurus: tweak the threshold value with experience...
        # Gurus: tweak cvsupport with use...
        self.cvsupport_threshold = {"F277W":0.02, "F380M": 0.02, "F430M": 0.02, "F480M": 0.02}
        if self.verbose: show_cvsupport_threshold(self)
        self.threshold = self.cvsupport_threshold[filt]


    def set_pscale(self, pscalex_deg=None, pscaley_deg=None):
        """
        Override pixel scale in header
        """
        if pscalex_deg is not None:
            self.pscalex_deg = pscalex_deg
        if pscaley_deg is not None:
            self.pscaley_deg = pscaley_deg
        self.pscale_mas = 0.5 * (pscalex_deg +  pscaley_deg) * (60*60*1000)
        self.pscale_rad = utils.mas2rad(self.pscale_mas)

    def read_data(self, fn, mode="slice"):
        # mode options are slice or UTR
        # for single slice data, need to read as 3D (1, npix, npix)
        # for utr data, need to read as 3D (ngroup, npix, npix)
        # fix bad pixels using DQ extension and LPL local averaging, 
        # but send bad pixel array down to where fringes are fit so they can be ignored.
        # For perfectly noiseless data we add GFaussian zero mean self.noise std dev
        # to imagge data.  Then std devs don't cause plot crashes with limits problems.
        
        with fits.open(fn, memmap=False, do_not_scale_image_data=True) as fitsfile:
            # use context manager, memmap=False, deepcopy to avoid memory leaks
            scidata = copy.deepcopy(fitsfile[1].data)
            if self.noise is not None: scidata += np.random.normal(0, self.noise, scidata.shape)
        
            # usually DQ ext in MAST file... make it non-fatal for DQ to be missing
            try:
                bpdata=copy.deepcopy(fitsfile['DQ'].data).astype(np.uint32) # bad pixel extension, forced to uint32
                self.bpexist = True
                dqmask = bpdata & self.bpval["DO_NOT_USE"] == self.bpval["DO_NOT_USE"] #
                del bpdata # free memory

                # True => driver wants to omit using pixels with dqflag raised in fit,
                if self.usedq == True:
                    print('InstrumentData.NIRISS.read_data: will not use flagged DQ pixels in fit')
            except Exception as e:
                print('InstrumentData.NIRISS.read_data: raised exception', e)
                self.bpexist = False
                dqmask = np.zeros(scidata.shape, dtype=np.uint32) # so it doesn't break if issues with DQ data

            if scidata.ndim == 3:  #len(scidata.shape)==3:
                print("read_data() input: 3D cube")

                # Truncate all but the first few slices od data and DQ array for rapid development
                if self.firstfew is not None:
                    if scidata.shape[0] > self.firstfew:
                        scidata = scidata[:self.firstfew, :, :]
                        dqmask = dqmask[:self.firstfew, :, :]
                # 'nwav' name (historical) is actually number of data slices in the 3Dimage cube
                self.nwav=scidata.shape[0]
                [self.wls.append(self.wls[0]) for f in range(self.nwav-1)]

            elif len(scidata.shape)==2: # 'cast' 2d array to 3d with shape[0]=1
                print("'InstrumentData.NIRISS.read_data: 2D data array converting to 3D one-slice cube")
                scidata = np.array([scidata,])
                dqmask = np.array([dqmask,])
            else:
                sys.exit("InstrumentData.NIRISS.read_data: invalid data dimensions for NIRISS. \nShould have dimensionality of 2 or 3.")

            # refpix removal by trimming
            scidata = scidata[:,4:, :] # [all slices, imaxis[0], imaxis[1]]
            print('\tRefpix-trimmed scidata:', scidata.shape)
            #### fix pix using bad pixel map - runs now.  Need to sanity-check.
            if self.bpexist:
                # refpix removal by trimming to match image trim
                dqmask = dqmask[:,4:, :]     # dqmask bool array to match image trimmed shape
                print('\tRefpix-trimmed dqmask: ', dqmask.shape)

            prihdr=fitsfile[0].header
            scihdr=fitsfile[1].header
            # MAST header or similar kwds info for oifits writer:
            self.updatewithheaderinfo(prihdr, scihdr)

            # Directory name into which to write txt observables & optional fits diagnostic files
            # The input fits image or cube of images file rootname is used to create the output
            # text&fits dir, using the data file's root name as the directory name: for example,
            # /abc/.../imdir/xyz_calints.fits  results in a directory /abc/.../imdir/xyz_calints/
            self.rootfn =  fn.split('/')[-1].replace('.fits', '')
        return prihdr, scihdr, scidata, dqmask


    def cdmatrix_to_sky(self, vec, cd11, cd12, cd21, cd22):
        """ use the global header values explicitly, for clarity 
            vec is 2d, units of pixels
            cdij 4 scalars, conceptually 2x2 array in units degrees/pixel
        """
        return np.array((cd11*vec[0] + cd12*vec[1], cd21*vec[0] + cd22*vec[1]))


    def degrees_per_pixel(self, hdr):
        """
        input: hdr:  fits data file's header with or without CDELT1, CDELT2 (degrees per pixel)
        returns: cdelt1, cdelt2: tuple, degrees per pixel along axes 1, 2
                         EITHER: read from header CDELT[12] keywords 
                             OR: calculated using CD matrix (Jacobian of RA-TAN, DEC-TAN degrees
                                 to pixel directions 1,2.  No deformation included in this routine,
                                 but the CD matric includes non-linear field distortion.
                                 No pupil distortion or rotation here.
                        MISSING: If keywords are missing default hardcoded cdelts are returned.
                                 The exact algorithm may substitute this later.
                                 Below seems good to ~5th significant figure when compared to 
                                 cdelts header values prior to replacement by cd matrix approach.
            N.D. at stsci 11 Mar 20212

            We start in Level 1 with the PC matrix and CDELT.
            CDELTs come from the SIAF.

            The PC matrix is computed from the roll angle, V3YANG and the parity.
            The code is here
            https://github.com/spacetelescope/jwst/blob/master/jwst/assign_wcs/util.py#L153

            In the level 2 imaging pipeline, assign_wcs adds the distortion to the files.
            At the end it computes an approximation of the entire distortion transformation
            by fitting a polynomial. This approximated distortion is represented as SIP
            polynomials in the FITS headers.

            Because SIP, by definition, uses a CD matrix, the PC + CDELT are replaced by CD.

            How to get CDELTs back?

            I think once the rotation, skew and scale are in the CD matrix it's very hard to
            disentangle them. The best way IMO is to calculate the local scale using three
            point difference. There is a function in jwst that does this.

            Using a NIRISS image as an example:

            from jwst.assign_wcs import util
            from jwst import datamodels

            im=datamodels.open('niriss_image_assign_wcs.fits')

            util.compute_scale(im.meta.wcs, (im.meta.wcsinfo.ra_ref, im.meta.wcsinfo.dec_ref))

            1.823336635353374e-05

            The function returns a constant scale. Is this sufficient for what you need or
            do you need scales and sheer along each axis? The code in util.compute_scale can
            help with figuring out how to get scales along each axis.

            I hope this answers your question.
        """

        if 'CD1_1' in hdr.keys() and 'CD1_2' in hdr.keys() and  \
             'CD2_1' in hdr.keys() and 'CD2_2' in hdr.keys():
            cd11 = hdr['CD1_1']
            cd12 = hdr['CD1_2']
            cd21 = hdr['CD2_1']
            cd22 = hdr['CD2_2']
            # Create unit vectors in detector pixel X and Y directions, units: detector pixels
            dxpix  =  np.array((1.0, 0.0)) # axis 1 step
            dypix  =  np.array((0.0, 1.0)) # axis 2 step
            # transform pixel x and y steps to RA-tan, Dec-tan degrees
            dxsky = self.cdmatrix_to_sky(dxpix, cd11, cd12, cd21, cd22)
            dysky = self.cdmatrix_to_sky(dypix, cd11, cd12, cd21, cd22)
            print("Used CD matrix for pixel scales")
            return np.linalg.norm(dxsky, ord=2), np.linalg.norm(dysky, ord=2)
        elif 'CDELT1' in hdr.keys() and 'CDELT2' in hdr.keys():
            return hdr['CDELT1'], hdr['CDELT2']
            print("Used CDDELT[12] for pixel scales")
        else:
            print('InstrumentData.NIRISS: Warning: NIRISS pixel scales not in header.  Using 65.6 mas in deg/pix')
            return 65.6/(60.0*60.0*1000), 65.6/(60.0*60.0*1000)

    
    def updatewithheaderinfo(self, ph, sh):
        """ input: primary header, science header MAST"""

        # The info4oif_dict will get pickled to disk when we write txt files of results.
        # That way we don't drag in objects like InstrumentData into code that reads text results
        # and writes oifits files - a simple built-in dictionary is the only object used in this transfer.
        info4oif_dict = {}
        info4oif_dict['telname'] = self.telname

        info4oif_dict['filt'] = self.filt
        info4oif_dict['lam_c'] = self.lam_c
        info4oif_dict['lam_w'] = self.lam_w
        info4oif_dict['lam_bin'] = self.lam_bin


        # Target information - 5/21 targname UNKNOWN in nis019 rehearsal data
        # Name in the proposal always non-trivial, targname still UNKNOWN...:
        if ph["TARGNAME"] == 'UNKNOWN': objname = ph['TARGPROP']
        else: objname = ph['TARGNAME'] # allegedly apt name for archive, standard form
        #
        # if target name has confusing-to-astroquery dash
        self.objname =  objname.replace('-', ' '); info4oif_dict['objname'] = self.objname
        # AB Dor, ab dor, AB DOR,  ab  dor are all acceptable.
        #
        self.ra = ph["TARG_RA"]; info4oif_dict['ra'] = self.ra
        self.dec = ph["TARG_DEC"]; info4oif_dict['dec'] = self.dec

        # / axis 1 DS9 coordinate of the reference pixel (always POS1)
        # / axis 2 DS9 coordinate of the reference pixel (always POS1)
        self.crpix1 = sh["CRPIX1"]; info4oif_dict['crpix1'] = self.crpix1
        self.crpix2 = sh["CRPIX2"]; info4oif_dict['crpix2'] = self.crpix2
        # need Paul Goudfrooij's table for actual crval[1,2] for true pointing to detector pixel coords (DS9)

        self.instrument = ph["INSTRUME"]; info4oif_dict['instrument'] = self.instrument
        self.pupil =  ph["PUPIL"]; info4oif_dict['pupil'] = self.pupil
        # "ImPlaneIA internal mask name" - oifwriter looks for 'mask'...
        self.arrname = "jwst_g7s6c"  # implaneia internal name - historical
        info4oif_dict['arrname'] = 'g7s6' # for oif
        info4oif_dict['mask'] = info4oif_dict['arrname']  # Soulain mask goes into oif arrname

        # if data was generated on the average pixel scale of the header
        # then this is the right value that gets read in, and used in fringe fitting
        pscalex_deg, pscaley_deg = self.degrees_per_pixel(sh)
        #
        info4oif_dict['pscalex_deg'] = pscalex_deg
        info4oif_dict['pscaley_deg'] = pscaley_deg
        # Whatever we did set is averaged for isotropic pixel scale here
        self.pscale_mas = 0.5 * (pscalex_deg + pscaley_deg) * (60*60*1000); \
        info4oif_dict['pscale_mas'] = self.pscale_mas
        self.pscale_rad = utils.mas2rad(self.pscale_mas); info4oif_dict['pscale_rad'] = self.pscale_rad

        self.mask = NRM_mask_definitions(maskname=self.arrname, chooseholes=self.chooseholes,
                                         holeshape=self.holeshape) # for STAtions x y in oifs

        self.date = ph["DATE-OBS"] + "T" + ph["TIME-OBS"]; info4oif_dict['date'] = self.date
        datestr = ph["DATE-OBS"]
        self.year = datestr[:4]; info4oif_dict['year'] = self.year
        self.month = datestr[5:7]; info4oif_dict['month'] = self.month
        self.day = datestr[8:10]; info4oif_dict['day'] = self.day
        self.parangh= sh["ROLL_REF"]; info4oif_dict['parangh'] = self.parangh
        self.pa = sh["PA_V3"]; info4oif_dict['pa'] = self.pa
        self.vparity = sh["VPARITY"]; info4oif_dict['vparity'] = self.vparity

        # An INTegration is NGROUPS "frames", not relevant here but context info.
        # 2d => "cal" file combines all INTegrations (ramps)
        # 3d=> "calints" file is a cube of all INTegrations (ramps)
        if sh["NAXIS"] == 2:
            # all INTegrations or 'ramps'
            self.itime = ph["EFFINTTM"] * ph["NINTS"]; info4oif_dict['itime'] = self.itime
        elif sh["NAXIS"] == 3:
            # each slice is one INTegration or 'ramp'
            self.itime = ph["EFFINTTM"]; info4oif_dict['itime'] = self.itime


        np.set_printoptions(precision=5, suppress=True, linewidth=160, 
                            formatter={'float': lambda x: "%10.5f," % x})
        self.v3i_yang = sh['V3I_YANG']  # Angle from V3 axis to Ideal y axis (deg)
        # rotate mask hole center coords by PAV3 # RAC 2021
        ctrs_sky = self.mast2sky()
        oifctrs = np.zeros(self.mask.ctrs.shape)
        oifctrs[:,0] = ctrs_sky[:,1].copy() * -1
        oifctrs[:,1] = ctrs_sky[:,0].copy() * -1
        info4oif_dict['ctrs_eqt'] = oifctrs # mask centers rotated by PAV3 (equatorial coords)
        info4oif_dict['ctrs_inst'] = self.mask.ctrs # as-built instrument mask centers
        info4oif_dict['hdia'] = self.mask.hdia
        info4oif_dict['nslices'] = self.nwav # nwav: number of image slices or IFU cube slices - AMI is imager
        self.info4oif_dict = info4oif_dict # save it when writing extracted observables txt


    # rather than calling InstrumentData in the niriss example just to reset just call this routine
    def reset_nwav(self, nwav):
        print("InstrumentData.NIRISS: ", "Resetting InstrumentData instantiation's nwave to", nwav)
        self.nwav = nwav


    def jwst_dqflags(self):
        """ 
            dqdata is a 2d (32-bit U?)INT array from the DQ extension of the input file.
            We ignore all data with a non-zero DQ flag.  I copied all values from a 7.5 build jwst...
            but we ignore any non-zero flag meaning, and ignore the pixel in fringe-fitting
            The refpix are non-zero DQ, btw...
            I changed "pixel" to self.pbval and "group" to self.bpgroup. We may use these later, 
            so here they are but initially we just discriminate between good (zero value) and non-good.
        """

        """ JWST Data Quality Flags
            The definitions are documented in the JWST RTD:
            https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#data-quality-flags 
        """
        """ JWST Data Quality Flags
        The definitions are documented in the JWST RTD:
        https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#data-quality-flags
        Implementation
        -------------
        The flags are implemented as "bit flags": Each flag is assigned a bit position
        in a byte, or multi-byte word, of memory. If that bit is set, the flag assigned
        to that bit is interpreted as being set or active.
        The data structure that stores bit flags is just the standard Python `int`,
        which provides 32 bits. Bits of an integer are most easily referred to using
        the formula `2**bit_number` where `bit_number` is the 0-index bit of interest.
        2**n is gauche but not everyone loves 1<<n


        Rachel uses:
        from jwst.datamodels import dqflags
        DO_NOT_USE = dqflags.pixel["DO_NOT_USE"]
        dqmask = pxdq0 & DO_NOT_USE == DO_NOT_USE
        pxdq = np.where(dqmask, pxdq0, 0)

        """

        # Pixel-specific flags
        self.bpval = {
                 'GOOD':             0,      # No bits set, all is good
                 'DO_NOT_USE':       2**0,   # Bad pixel. Do not use.
                 'SATURATED':        2**1,   # Pixel saturated during exposure
                 'JUMP_DET':         2**2,   # Jump detected during exposure
                 'DROPOUT':          2**3,   # Data lost in transmission
                 'OUTLIER':          2**4,   # Flagged by outlier detection. Was RESERVED_1
                 'RESERVED_2':       2**5,   #
                 'RESERVED_3':       2**6,   #
                 'RESERVED_4':       2**7,   #
                 'UNRELIABLE_ERROR': 2**8,   # Uncertainty exceeds quoted error
                 'NON_SCIENCE':      2**9,   # Pixel not on science portion of detector
                 'DEAD':             2**10,  # Dead pixel
                 'HOT':              2**11,  # Hot pixel
                 'WARM':             2**12,  # Warm pixel
                 'LOW_QE':           2**13,  # Low quantum efficiency
                 'RC':               2**14,  # RC pixel
                 'TELEGRAPH':        2**15,  # Telegraph pixel
                 'NONLINEAR':        2**16,  # Pixel highly nonlinear
                 'BAD_REF_PIXEL':    2**17,  # Reference pixel cannot be used
                 'NO_FLAT_FIELD':    2**18,  # Flat field cannot be measured
                 'NO_GAIN_VALUE':    2**19,  # Gain cannot be measured
                 'NO_LIN_CORR':      2**20,  # Linearity correction not available
                 'NO_SAT_CHECK':     2**21,  # Saturation check not available
                 'UNRELIABLE_BIAS':  2**22,  # Bias variance large
                 'UNRELIABLE_DARK':  2**23,  # Dark variance large
                 'UNRELIABLE_SLOPE': 2**24,  # Slope variance large (i.e., noisy pixel)
                 'UNRELIABLE_FLAT':  2**25,  # Flat variance large
                 'OPEN':             2**26,  # Open pixel (counts move to adjacent pixels)
                 'ADJ_OPEN':         2**27,  # Adjacent to open pixel
                 'UNRELIABLE_RESET': 2**28,  # Sensitive to reset anomaly
                 'MSA_FAILED_OPEN':  2**29,  # Pixel sees light from failed-open shutter
                 'OTHER_BAD_PIXEL':  2**30,  # A catch-all flag
                 'REFERENCE_PIXEL':  2**31,  # Pixel is a reference pixel
        }

        # Group-specific flags. Once groups are combined, these flags
        # are equivalent to the pixel-specific flags.
        self.bpgroup = {
                 'GOOD':       self.bpval['GOOD'],
                 'DO_NOT_USE': self.bpval['DO_NOT_USE'],
                 'SATURATED':  self.bpval['SATURATED'],
                 'JUMP_DET':   self.bpval['JUMP_DET'],
                 'DROPOUT':    self.bpval['DROPOUT'],
        }


    def mast2sky(self):
        """
        Rotate hole center coordinates:
            Clockwise by the V3 position angle - V3I_YANG from north in degrees if VPARITY = -1
            Counterclockwise by the V3 position angle - V3I_YANG from north in degrees if VPARITY = 1
        Hole center coords are in the V2, V3 plane in meters.
        Return rotated coordinates to be put in info4oif_dict.
        implane2oifits.ObservablesFromText uses these to calculate baselines.
        """
        pa = self.pa
        mask_ctrs = copy.deepcopy(self.mask.ctrs)
        # rotate by an extra 90 degrees (RAC 9/21)
        # these coords are just used to orient output in OIFITS files
        # NOT used for the fringe fitting itself
        mask_ctrs = utils.rotate2dccw(mask_ctrs,np.pi/2.)
        vpar = self.vparity # Relative sense of rotation between Ideal xy and V2V3
        v3iyang = self.v3i_yang
        rot_ang = pa - v3iyang # subject to change!

        if pa != 0.0:
            # Using rotate2sccw, which rotates **vectors** CCW in a fixed coordinate system,
            # so to rotate coord system CW instead of the vector, reverse sign of rotation angle.  Double-check comment
            if vpar == -1:
                # rotate clockwise  <rotate coords clockwise?>
                ctrs_rot = utils.rotate2dccw(mask_ctrs, np.deg2rad(-rot_ang))
                print(f'InstrumentData.mast2sky: Rotating mask hole centers clockwise by {rot_ang:.3f} degrees')
            else:
                # counterclockwise  <rotate coords counterclockwise?>
                ctrs_rot = utils.rotate2dccw(mask_ctrs, np.deg2rad(rot_ang))
                print('InstrumentData.mast2sky: Rotating mask hole centers counterclockwise by {rot_ang:.3f} degrees')
        else:
            ctrs_rot = mask_ctrs
        return ctrs_rot
