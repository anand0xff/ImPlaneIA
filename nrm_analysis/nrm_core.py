#! /usr/bin/env python

"""
by A. Greenbaum & A. Sivaramakrishnan 
April 2016 agreenba@pha.jhu.edu

Contains 

FringeFitter - fit fringe phases and amplitudes to data in the image plane

Calibrate - Calibrate target data with calibrator data

BinaryAnalyze - Detection, mcmc modeling, visualization tools
                Much of this section is based on tools in the pymask code
                by F. Martinache, B. Pope, and A. Cheetham
                We especially thank A. Cheetham for help and advising for
                developing the analysis tools in this package. 

    LG++ anand@stsci.edu nrm_core changes:
        Removed use of 'centering' parameter, switch to psf_offsets, meant to be uniformly ds9-compatible 
        offsets from array center (regardless of even/odd array sizes).

            nrm_core.fit_image(): refslice UNTESTED w/ new utils.centroid()
            nrm_core.fit_image(): hold_centering UNTESTED w/ new utils.centroid()

"""


from __future__ import print_function
# Standard imports
import os, sys, time
import numpy as np
from astropy.io import fits
from scipy.special import comb
from scipy.stats import sem, mstats
import pickle as pickle
import matplotlib.pyplot as plt


# Module imports
from nrm_analysis.fringefitting.LG_Model import NRM_Model
from nrm_analysis.misctools import utils  # AS LG++
from nrm_analysis.misctools import implane2oifits 
from nrm_analysis.modeling.binarymodel import model_cp_uv, model_allvis_uv, model_v2_uv, model_t3amp_uv
from nrm_analysis.modeling.multimodel import model_bispec_uv

from multiprocessing import Pool

class FringeFitter:
    def __init__(self, instrument_data, **kwargs):
        """
        Fit fringes in the image plane

        Takes an instance of the appropriate instrument class
        Various options can be set

        kwarg options:
        weighted - least squares fit uses weighting by 1 / photon noise variance
        oversample - model oversampling (also how fine to measure the centering)
        psf_offset - If you already know the subpixel centering of your data, 
                     give it here (not recommended except when debugging with
                     perfectly known image placement))
        oitdir - Where text observables, derived fits files will get saved.  No default.
        oifdir - Where raw oifits files will get saved.  No default.
        npix - How many pixels of your data do you want to use? 
               Default is the shape of a data [slice or frame].  Typically odd?
        debug - will plot the FT of your data next to the FT of a reference PSF.
                Needs poppy package to run
        verbose_save - saves more than the standard files
        interactive - default True, prompts user to overwrite/create directory.  
                      False will overwrite files where necessary.
        find_rotation - will find the best pupil rotation that matches the data
        verbose - T/F

        main method:
        * fit_fringes


        """
        self.instrument_data = instrument_data

        #######################################################################
        # Options
        # setting a default oversample value
        # - fast, not necessarily  the most accurate... 
        # ok to ~1e-3 binary contrast (given other systematic limits, 2021 end)??
        self.weighted = False
        if "weighted" in kwargs:
            self.weighted = kwargs["weighted"]

        self.oversample = 3  
        if "oversample" in kwargs:
            self.oversample = kwargs["oversample"]

        self.find_rotation = False
        if "find_rotation" in kwargs:
            # can be True/False or 1/0
            self.find_rotation = kwargs["find_rotation"]

        self.psf_offset_ff = None #find center of image in data
        if "psf_offset_ff" in kwargs: # if so do not find center of image in data
            self.psf_offset_ff = kwargs["psf_offset_ff"]         

        ###############################
        # restructured output directories, with backward compatibility
        #
        # write OI text files here, [diagnostic images fits]. Was 'savedir'
        if "oitdir" in kwargs:
            self.oitdir = kwargs["oitdir"]
        elif 'savedir' in kwargs:
            self.oitdir = kwargs["savedir"]
            print("nrm_core.FringeFitter: savedir deprecated but will be used for oitdir variable.")
        else:
            sys.exit("FringeFitter: Fatal: no oitdir (or deprecated savedir) specified.")

        if self.oitdir[-1] != '/': self.oitdir = self.oitdir + '/'

        if "oifdir" in kwargs:  # write OIFITS files here.  New parameter 2021/05
            self.oifdir = kwargs["oifdir"]
        elif 'savedir' in kwargs:
            self.oifdir = kwargs["savedir"]
            print("nrm_core.FringeFitter: savedir deprecated, but oifdir wil be set to savedir")
            print("nrm_core.FringeFitter: new drivers should initialize with oifdir")
        else:
            sys.exit("FringeFitter: Fatal: no oifdir / deprecated savedir) specified.")
        if self.oifdir[-1] != '/': self.oifdir = self.oifdir + '/'
        #
        ###############################

        self.npix = 'default'
        if "npix" in kwargs:
            self.npix = kwargs["npix"]

        self.debug=False
        if "debug" in kwargs:
            self.debug=kwargs["debug"]

        self.verbose_save = False
        if "verbose_save" in kwargs:
            self.verbose_save = kwargs["verbose_save"]

        self.interactive = True
        if 'interactive' in kwargs:
            self.interactive = kwargs['interactive']

        self.save_txt_only = False
        if "save_txt_only" in kwargs:
            self.save_txt_only = kwargs["save_txt_only"]

        self.verbose = False
        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]
        #######################################################################


        #######################################################################
        # Create OI text & oifits directories if they don't already exist
        try:
            os.makedirs(self.oitdir)
        except:
            if self.interactive is True:
                print(self.oitdir+" Already exists, rewrite its contents? (y/n)")
                ans = input()
                if ans == "y":
                    pass
                elif ans == "n":
                    sys.exit("use alternative save directory with kwarg 'oitdir' when calling FringeFitter")
                else:
                    sys.exit("Invalid answer. Stopping.")
            else:
                pass
        try:
            os.makedirs(self.oifdir)
        except FileExistsError:
            pass

    ###
    # May 2017 J Sahlmann updates: parallelized fringe-fitting
    # Feb 2021 Dec 2021 anand added dqmask to fringe fitting, removed bpdata use
    ###

    def fit_fringes(self, fns, threads = 0):
        if type(fns) == str:
            fns = [fns, ]

        # Can get fringes for images in parallel
        store_dict = [{"object":self, "file":fn,"id":jj} \
                        for jj,fn in enumerate(fns)]

        t2 = time.time()
        for jj, fn in enumerate(fns):
            fit_fringes_parallel({"object":self, "file":fn,\
                                  "id":jj}, threads)
        t3 = time.time()
        print("Parallel with {0} threads took {1:.2f}s to fit all fringes".format(\
               threads, t3-t2))
        # If a JWST file name, strip the 'calints' suffix from it
        # if it doesn't contain '_calints' nothing will happen
        oifn_out = self.instrument_data.rootfn.split('_calints')[0]+'.oifits'

        # Read in all relevant text observables and save to oifits file...
        dct = implane2oifits.oitxt2oif(nh=7, oitdir=self.oitdir+self.instrument_data.rootfn+'/',
                                             oifn=oifn_out,
                                             oifdir=self.oifdir,
                                             verbose=self.verbose,
                                             )

        

    def save_output(self, slc, nrm):
        # cropped & centered PSF
        self.datapeak = self.ctrd.max()
        #TBD: Keep only n_*.fits files after testing is done and before doing ImPlaneIA delivery
        if self.save_txt_only == False:
            fits.PrimaryHDU(data=self.ctrd, \
                    header=self.scihdr).writeto(self.oitdir+\
                    self.instrument_data.rootfn+"/centered_{0}.fits".format(slc), \
                    overwrite=True)
            fits.PrimaryHDU(data=self.ctrd/self.datapeak, \
                    header=self.scihdr).writeto(self.oitdir+\
                    self.instrument_data.rootfn+"/n_centered_{0}.fits".format(slc), \
                    overwrite=True)

            model, modelhdu = nrm.plot_model(fits_true=1)
            # save to fits files
            fits.PrimaryHDU(data=nrm.residual).writeto(self.oitdir+\
                            self.instrument_data.rootfn+\
                            "/residual_{0:02d}.fits".format(slc), \
                            overwrite=True)
            fits.PrimaryHDU(data=nrm.residual/self.datapeak).writeto(self.oitdir+\
                            self.instrument_data.rootfn+\
                            "/n_residual_{0:02d}.fits".format(slc), \
                            overwrite=True)
            modelhdu.writeto(self.oitdir+\
                             self.instrument_data.rootfn+\
                             "/modelsolution_{0:02d}.fits".format(slc),\
                             overwrite=True)
            fits.PrimaryHDU(data=model/self.datapeak, \
                            header=modelhdu.header).writeto(self.oitdir+\
                            self.instrument_data.rootfn+\
                            "/n_modelsolution_{0:02d}.fits".format(slc), \
                            overwrite=True)
            try: # if there's an appropriately trimmed bad pixel map write it out
                fits.PrimaryHDU(data=self.ctrb, \
                    header=self.scihdr).writeto(self.oitdir+\
                    self.instrument_data.rootfn+"/bp_{0}.fits".format(slc), \
                    overwrite=True)
            except: AttributeError
                

        # default save to text files
        np.savetxt(self.oitdir+self.instrument_data.rootfn + \
                   "/solutions_{0:02d}.txt".format(slc), nrm.soln)
        np.savetxt(self.oitdir+self.instrument_data.rootfn + \
                   "/phases_{0:02d}.txt".format(slc), nrm.fringephase)
        np.savetxt(self.oitdir+self.instrument_data.rootfn + \
                   "/amplitudes_{0:02d}.txt".format(slc), nrm.fringeamp)
        np.savetxt(self.oitdir+self.instrument_data.rootfn + \
                   "/CPs_{0:02d}.txt".format(slc), nrm.redundant_cps)
        np.savetxt(self.oitdir+self.instrument_data.rootfn + \
                   "/CAs_{0:02d}.txt".format(slc), nrm.redundant_cas)
        np.savetxt(self.oitdir+self.instrument_data.rootfn + \
                  "/fringepistons_{0:02d}.txt".format(slc), nrm.fringepistons)

        # write info that oifits wants only when writing out first slice.
        # this will relevant to all slices... so no slice number here.
        if slc == 0:
            pfn = self.oitdir+self.instrument_data.rootfn + "/info4oif_dict.pkl"
            pfd = open(pfn, 'wb')
            pickle.dump(self.instrument_data.info4oif_dict, pfd)
            pfd.close()

        # optional save outputs
        if self.verbose_save:
            np.savetxt(self.oitdir+self.instrument_data.rootfn+\
                       "/condition_{0:02d}.txt".format(slc), nrm.cond)
            np.savetxt(self.oitdir+self.instrument_data.rootfn+\
                       "/flux_{0:02d}.txt".format(slc), nrm.flux)
          

    def save_auto_figs(self, slc, nrm):
        
        # rotation
        if self.find_rotation==True:
            plt.figure()
            plt.plot(nrm.rots, nrm.corrs)
            plt.vlines(nrm.rot_measured, nrm.corrs[0],
                        nrm.corrs[-1], linestyles='--', color='r')
            plt.text(nrm.rots[1], nrm.corrs[1], 
                     "best fit at {0}".format(nrm.rot_measured))
            plt.savefig(self.oitdir+self.instrument_data.rootfn+\
                        "/rotationcorrelation_{0:02d}.png".format(slc))

def fit_fringes_parallel(args, threads):
    self = args['object']
    filename = args['file']
    id_tag = args['id']
    self.prihdr, self.scihdr, self.scidata, self.dqmask = \
        self.instrument_data.read_data(filename)

    try:
        os.makedirs(self.oitdir+self.instrument_data.rootfn)
    except:
        pass

    store_dict = [{"object":self, "slc":slc} for slc in \
                  range(self.instrument_data.nwav)]

    if threads>0:
        pool = Pool(processes=threads)
        print("Running fit_fringes in parallel with {0} threads".format(threads))
        pool.map(fit_fringes_single_integration, store_dict)
        pool.close()
        pool.join()
    else:
        for slc in range(self.instrument_data.nwav):
            fit_fringes_single_integration({"object":self, "slc":slc})

def fit_fringes_single_integration(args):
    self = args["object"]
    slc = args["slc"]  # indexes each slice of 3D stack of images
    id_tag = args["slc"]
    nrm = NRM_Model(mask=self.instrument_data.mask,
                    pixscale=self.instrument_data.pscale_rad,
                    holeshape=self.instrument_data.holeshape,
                    affine2d=self.instrument_data.affine2d,
                    over = self.oversample)

    # for image data from single filter, this is the filter bandpass.
    # otherwise it's the IFU wavelength slice.
    nrm.bandpass = self.instrument_data.wls[slc]

    if self.npix == 'default':
        self.npix = self.scidata[slc,:,:].shape[0]
    
    # New or modified in LG++
    # center the image on its peak pixel:
    # AS subtract 1 from "r" below  for testing >1/2 pixel offsets
    # AG 03-2019 -- is above comment still relevant?
    
    # Where appropriate, the slice under consideration is centered, and processed
    if self.instrument_data.arrname=="jwst_g7s6c":
        # get the cropped image and identically-cropped bad pixel data:
        self.ctrd, self.dqslice = utils.center_imagepeak(
                                    self.scidata[slc,:,:], 
                                    dqm=self.dqmask[slc,:,:]) 
    else:
        self.ctrd = utils.center_imagepeak(self.scidata[slc,:,:])  
    

    # store the 2D cropped image centered on the brightest pixel, 
    # bad pixels smoothed over

    if self.psf_offset_ff is None:
        # returned values have offsets x-y flipped:
        # Finding centroids the Fourier way assumes no bad pixels case 
        #   - Fourier domain mean slope

        # offsets from brightest pixel ctr
        centroid = utils.find_centroid(self.ctrd, self.instrument_data.threshold)
        # use flipped centroids to update centroid of image for JWST 
        # pixel coordinates: - note the flip of [0] and [1] to match DS9 view
        image_center = utils.centerpoint(self.ctrd.shape) + \
                            np.array((centroid[1], centroid[0])) # info only, unused
        nrm.xpos = centroid[1]  # flip 0 and 1 to convert
        nrm.ypos = centroid[0]  # flip 0 and 1
        nrm.psf_offset = nrm.xpos, nrm.ypos  # renamed .bestcenter to .psf_offset
        if self.debug: 
            print("nrm.core.fit_fringes_single_integration: utils.find_centroid() -> nrm.psf_offset")
    else:
        # user-provided psf_offset python-style offsets from array center are here.
        nrm.psf_offset = self.psf_offset_ff 


    nrm.make_model(fov=self.ctrd.shape[0], 
                   bandpass=nrm.bandpass, 
                   over=self.oversample,
                   psf_offset=nrm.psf_offset,  
                   pixscale=nrm.pixel)

    # again, fit just one slice...
    if self.instrument_data.arrname=="jwst_g7s6c":
        nrm.fit_image(self.ctrd, 
                      modelin=nrm.model, 
                      psf_offset=nrm.psf_offset,
                      dqm=self.dqslice,
                      weighted=self.weighted)
    else:
        nrm.fit_image(self.ctrd,
                      modelin=nrm.model,
                      psf_offset=nrm.psf_offset,
                      weighted=self.weighted)

    """
    Attributes now stored in nrm object:

    -----------------------------------------------------------------------------
    soln            --- resulting sin/cos coefficients from least squares fitting
    fringephase     --- baseline phases in radians
    fringeamp       --- baseline amplitudes (flux normalized)
    redundant_cps   --- closure phases in radians
    redundant_cas   --- closure amplitudes
    residual        --- fit residuals [data - model solution]
    cond            --- matrix condition for inversion
    fringepistons   --- zero-mean piston opd in radians on each hole (eigenphases)
    -----------------------------------------------------------------------------
    For jwst_g7s6 cropped-to-match-data bad pixel array 'ctrb' is also stored
    """

    self.save_output(slc, nrm)  # Please elucidate what this is for
    self.nrm = nrm # store  extracted values here
    return None
