#! /usr/bin/env  python 
# Heritage mathematia nb from Alex & Laurent
# Heritage python by Alex Greenbaum & Anand Sivaramakrishnan Jan 2013
# updated May 2013 to include hexagonal envelope

# Goals: a more convenient module for analytic simulation & model fitting

# Anand: rewrite to remove on-the-fly decisions/defaults, multiple execution paths
# Anand: rewrite to unify psf offsets, rationalize analytical nrm code & usage from LM_Model.simulate()
# Anand: clean up internal module import to be relative
"""
Imports:
"""
import numpy as np
import scipy.special
#import math
import nrm_analysis.misctools.utils as utils
import nrm_analysis.misctools.mask_definitions as mask_definitions
import sys, os
import time
from astropy.io import fits
import logging
from argparse import ArgumentParser
from . import leastsqnrm as leastsqnrm
from . import analyticnrm2
from . import subpix

_default_log = logging.getLogger('NRM_Model')
#_default_log.setLevel(logging.INFO)
_default_log.setLevel(logging.ERROR)

"""
====================
NRM_Model
====================

A module for conveniently dealing with an "NRM object"
This should be able to take an NRM_mask_definitions object for mask geometry

Defines mask geometry and detector-scale parameters
Simulates PSF (broadband or monochromatic)
Builds a fringe model - either by user definition, or automated to data
Fits model to data by least squares

Masks:
  * gpi_g10s40
  * jwst
  * visir?

Methods:

simulate
make_model
fit_image
plot_model
perfect_from_model
new_and_better_scaling_routine
auto_find_center
save


First written by Alexandra Greenbaum 2013-2014
Algorithm documented in 
    Greenbaum, A. Z., Pueyo, L. P., Sivaramakrishnan, A., and Lacour, S., 
    Astrophysical Journal vol. 798, Jan 2015.
Developed with NASA APRA (AS, AZG), NSF GRFP (AZG), NASA Sagan (LP), and French taxpayer (SL)  support
Refactored by Greenbaum, Sivaramakrishnan

"""
phi_nb = np.array( [0.028838669455909766, -0.061516214504502634, \
     0.12390958557781348, -0.020389361461019516, 0.016557347248600723, \
    -0.03960017912525625, -0.04779984719154552] ) # phi in waves
# define phi at the center of F430M band:
phi_nb = phi_nb *4.3e-6 # phi_nb in m
m = 1.0
mm = 1.0e-3 * m
um = 1.0e-6 * m
mas = 1.0e-3 / (60*60*180/np.pi) # in radians


class NRM_Model():

    def __init__(self, mask=None, v3_yang=0.0, holeshape="circ", pixscale=None,
            over = 1, log=_default_log, pixweight=None,
            datapath="",
            phi=None, refdir="",
            chooseholes=False,
            affine2d = None,
            **kwargs):
        """
        mask will either be a string keyword for built-in values or
        an NRM_mask_geometry object.
        pixscale should be input in radians.
        phi (rad) default changedfrom "perfect" to None (with bkwd compat.)
        """ 

        if "debug" in kwargs:
            self.debug=kwargs["debug"]
        else:
            self.debug=False

        # define a handler to write log messages to stdout
        sh = logging.StreamHandler(stream=sys.stdout)

        # define the format for the log messages, here: "level name: message"
        formatter = logging.Formatter("[%(levelname)s]: %(message)s")
        sh.setFormatter(formatter)
        self.logger = log
        self.logger.addHandler(sh)

        self.holeshape = holeshape
        self.pixel = pixscale # det pix in rad (square)
        self.over = over
        self.maskname = mask  # should change to "mask", and mask.maskname is then eg jwst_g7s6c or whatever 2021 feb anand
        # Cos incoming 'mask' is str, this is a mask object.
        #elf.mask = mask  # should change to "mask", and mask.maskname is then eg jwst_g7s6c or whatever 2021 feb anand
        self.pixweight = pixweight


        if mask is None:
            print("LG_Model.__init__: No mask name specified for model, using jwst_g7s6c")
            mask = mask_definitions.NRM_mask_definitions(maskname="jwst_g7s6c", 
                                    chooseholes=chooseholes, 
                                    holeshape="hex")
        elif isinstance(mask, str):
            mask = mask_definitions.NRM_mask_definitions(maskname=mask, 
                                    chooseholes=chooseholes, 
                                    holeshape="hex")
        self.ctrs = mask.ctrs
        self.d = mask.hdia
        self.D = mask.activeD



        self.N = len(self.ctrs)
        self.datapath = datapath
        self.refdir = refdir
        self.fmt = "%10.4e"

        if phi: # meters of OPD at central wavelength
            if phi == "perfect": 
                self.phi = np.zeros(self.N) # backwards compatibility
                print('LG_Model.__init__(): phi="perfect" deprecated in LG++.  Omit phi or use phi=None')
            else:
                self.phi = phi
        else:
            self.phi = np.zeros(self.N)

        self.chooseholes = chooseholes

        """  affine2d property not to be changed in NRM_Model - create a new instance instead 
             Save affine deformation of pupil object or create a no-deformation object. 
             We apply this when sampling the PSF, not to the pupil geometry.
        """
        if affine2d is None:
            self.affine2d = utils.Affine2d(mx=1.0,my=1.0, 
                                           sx=0.0,sy=0.0, 
                                           xo=0.0,yo=0.0, name="Ideal")
        else:
            self.affine2d = affine2d

    def vprint(self, *args):  # debug mode printing
        if self.debug: print("LG_Model: ",  *args)
        else: pass

    def set_ctrs_rot(self, rotation_ccw_deg):
        """ 
        test routine or used for preferrably small rotations of as_built centers.
        No-op, unchanged self.ctrs,  if rotation exactly zero or False.
        Otherwise overwrites self.ctrs, the "live" centers of mask holes.
        In the interests of code clarity do not use for coordinate flips
        In data reduction piupelines DO NOT USE THIS ROUTINE - use the
        BEING REMOVED - DO NOT USE - use Affine2d resampling of the image plane instead.
        """
        self.vprint(self, """
        LG_Model.NRM_Model.set_ctrs_rot(): BEING REMOVED - DON'T USE - use Affine2d instead""")

        if rotation_ccw_deg:
            self.ctrs = utils.rotate2dccw(self.ctrs_asbuilt, rotation_ccw_deg * np.pi / 180.0)
            self.rotation_ccw_deg = rotation_ccw_deg # record-keeping
            self.vprint(self, "\tLG_Model.NRM_Model.set_ctrs_rot: ctrs in V2V3, meters after rot %.2f deg:"%rotation_ccw_deg)
        else:
            self.vprint(self, "\tLG_Model.NRM_ModelLG_Model.set_ctrs_rot: ctrs in V2V3, meters:")


    def set_pistons(self, phi_m):
        """Meters of OPD at center wavelength LG++ """
        self.phi = phi_m

    def set_pixelscale(self, pixel_rad):
        """Detector pixel scale (isotropic) """
        self.vprint(self, """
        LG_Model.NRM_Model.set_pixel_scale({0:.6e} radians = {1:.2f} mas""".format(pixel_rad, pixel_rad/mas)) 
        self.pixel = pixel_rad
        
    def show_ctrs(self, figstr="Fig_rot", figdir=""):
        import matplotlib.pyplot as plt
        f = plt.figure(figstr, figsize=(4,4))
        plt.plot(self.ctrs[:,0],self.ctrs[:,1], 
                color='b', linewidth=0.0,
                markersize=12, marker='o')
        plt.text(-3.0,  3.0, "M1 AS SEEN FROM M2", fontsize=8)
        plt.text(-3.0, -3.0, "LIVE CENTERS", fontsize=8)
        for i in range(self.ctrs.shape[0]): # x y in meters
            plt.text(self.ctrs[i,0]+0.2, self.ctrs[i,1], "(%+6.3f,%+6.3f)"%(self.ctrs[i,0],self.ctrs[i,1]), fontsize=8)
        a = plt.axes()
        plt.setp(a, xlim=(-3.4,+3.4))
        plt.setp(a, ylim=(-3.4,+3.4))
        plt.title(figstr + " " + repr(self.chooseholes))
        plt.xlabel('PM V2/m')
        plt.ylabel('PM V3/m')
        plt.savefig(figdir+"/"+figstr+".png")
        plt.clf()


    def simulate(self, fov=None, bandpass=None, over=None, psf_offset=(0,0)):
        """
        This method will simulate a detector-scale PSF using parameters input from the 
        call and already stored in the object. It also generates a simulation 
        fits header storing all of the parameters used to generate that PSF.
        If the input bandpass is one number it will calculate a monochromatic PSF.
        Use set_pistons() (meters) to set self.phi (rad) if desired   #### RE-CHECK! ####
        Will use the object's live mask centers self.cts

        fov (integer) in detector pixels, must be specified

        psf_offset (in detector pixels, PSF center offset from center of array)
            - PSF then centered at centerpoint(shape) + psf_offset
            - Works for both odd and even shapes.
            - default (0,0) - no offset from array center
            - places psf ctr at array ctr + offset, per ds9 view of offset[0],offset[1]
              Note: moving from pixel to pitch - psf_offset stays in detector pixels

        over (integer) is the oversampling of the simulation  
            - defaults to no oversampling (i.e. 1)
              Note: moving from pixel to pitch - psf_offset stays in detector pixels

        All mask rotations must be done outside simulate() prior to call
        Note: moving from pixel to pitch, rotations implemented in 
        utils's Affine2d class.

        Returns psf array (at requested or default oversampling)
        but stores hdr and detector-scale psf in 'self'
        """
        self.simhdr = fits.PrimaryHDU().header
        # First set up conditions for choosing various parameters
        # deleted fov logic, self.fov_sim
        self.bandpass = bandpass


        if over == None: 
            over = 1  # ?  Always comes in as integer.
            self.vprint(self, "defaulting to oversample=1, i.e. detector pixel scale pitch used.")
        #elf.pixel_sim = self.pixel/float(over)  #??? need?  
        # Deepashri prefers detector pixel scale & oversamp only in the object

        self.simhdr['OVER'] = (over, 'sim pix = det pix/over')
        self.simhdr['PIX_OV'] = (self.pixel/float(over), 'Sim pixel scale in radians')
        self.simhdr["psfoff0"] = (psf_offset[0], "psf ctr at arrayctr + off[0], detpix") # user-specified
        self.simhdr["psfoff1"] = (psf_offset[1], "psf ctr at arrayctr + off[1], detpix") # user-specified

        self.psf_over = np.zeros((over*fov, over*fov))
        nspec = 0
        # accumulate polychromatic oversampled psf in the object
        self.vprint(self, "**** simulate():  psf_offset {0}".format(psf_offset))
        for w,l in bandpass: # w: wavelength's weight, l: lambda (wavelength)
            self.vprint(self, "weight:", w, "wavelength:", l)
            self.vprint(self, "fov/detector pixels:", fov)
            self.vprint(self, "over:", over)
            self.vprint(self, "pixel:", self.pixel)
            self.psf_over += w * analyticnrm2.PSF(self.pixel, # det pixel scale, rad
                                                  fov,   # in detpix number
                                                  over,
                                                  self.ctrs, # live hole centers in object
                                                  self.d, l,
                                                  self.phi,
                                                  psf_offset, # det pixels
                                                  self.affine2d,
                                                  shape=self.holeshape)
            # offset signs fixed to agree w/DS9, +x shifts ctr R, +y shifts up
            self.simhdr["WAVL{0}".format(nspec)] = (l, "wavelength (m)")
            self.simhdr["WGHT{0}".format(nspec)] = (w, "weight")
            nspec += 1

        # store the detector pixel scale psf in the object
        self.psf = utils.rebin(self.psf_over, (over, over))

        return self.psf

    ############################################################################### AS 10 2017 mark

    def make_model(self, fov=None, bandpass=None, over=1, psf_offset=(0,0), pixscale=None):
        
        """
        make_model generates the fringe model with the attributes of the object.
        model is a collection of fringe intensities (nholes = 7 means model has
        21 cosines, 21 sines, a DC-like, and a flux slice: 44 2D slices in all.
        It can take either a single wavelength or a bandpass as a list of tuples.
        The bandpass should be of the form [(weight1, wavl1), (weight2, wavl2),...]

        2020.03.13
        Input pistons self.phi are not included (at this time) in the call to 
                  analyticnrm2.model_array()
        They should be included (for correctness) - but we're not sure how
        important this effect is currently.
        """
        if fov:
            self.fov = fov
        self.over=over

        if hasattr(self, 'pixscale_measured'):
            if self.pixscale_measured is not None:
                self.modelpix = self.pixscale_measured
        if pixscale==None:
            self.modelpix=self.pixel
        else:
            self.modelpix=pixscale

        self.modelctrs = self.ctrs

        # The model shape is (fov) x (fov) x (# solution coefficients)
        # the coefficient refers to the terms in the analytic equation
        # There are N(N-1) independent pistons, double-counted by cosine
        # and sine, one constant term and a DC offset.
        #elf.model = np.ones((self.fov, self.fov, self.N*(self.N-1)+2)) # corrected below AZG AS LG++
        self.model = np.zeros((self.fov, self.fov, self.N*(self.N-1)+2))
        self.model_beam = np.zeros((self.over*self.fov, self.over*self.fov))
        self.fringes = np.zeros((self.N*(self.N-1)+1, self.over*self.fov, self.over*self.fov))
        for w,l in bandpass: # w: weight, l: lambda (wavelength)
            self.vprint(self, "weight: {0}, lambda: {1}".format(w,l))
            # model_array returns the envelope and fringe model (a list of oversampled fov x fov slices)
            pb, ff = analyticnrm2.model_array(self.modelctrs, l, self.over,
                              self.modelpix,
                              self.fov,
                              self.d,
                              shape=self.holeshape,
                              psf_offset=psf_offset,
                              affine2d=self.affine2d, 
                              verbose=False)
            self.logger.debug("Passed to model_array: psf_offset: {0}".format(psf_offset))
            self.logger.debug("Primary beam in the model created: {0}".format(pb))
            self.model_beam += pb
            self.fringes += ff

            # this routine multiplies the envelope by each fringe "image"
            self.model_over = analyticnrm2.multiplyenv(pb, ff)
            #print("LG_Model.make_model: NRM MODEL model shape:", self.model_over.shape)

            model_binned = np.zeros((self.fov,self.fov, self.model_over.shape[2]))
            # loop over slices "sl" in the model
            for sl in range(self.model_over.shape[2]):
                model_binned[:,:,sl] =  utils.rebin(self.model_over[:,:,sl],  (self.over, self.over))

            self.model += w*model_binned
    
        #print("LG_Model.make_model: self.model", type(self.model), type(self.model[0,0,0]))
        return self.model


    def fit_image(self, image, reference=None, pixguess=None, rotguess=0, psf_offset=(0,0),
                  modelin=None, savepsfs=False, dqm=None, weighted=False):
        """
        This works on 2D "centered" images fed to it.
        dqm is optional 2D bad pixel bool array, same size as image.  
        self.maskname is a maskdef object with property self.maskname.mask a string
        2021
        """

        self.vprint(self, "\n    **** LG_Model.NRM_Model.fit_image: psf_offset {}".format(psf_offset))
        if hasattr(modelin, 'shape'):
            self.vprint(self, "    **** LG_Model.NRM_Model.fit_image modelin passed in")
        else:
            self.vprint(self, "    **** LG_Model.NRM_Model.fit_image: modelin is None\n")

        """
        fit_image will run a least-squares fit on an input image.
        Specifying a model is optional. If a model is not specified then this
        method will find the appropriate wavelength scale, rotation (and 
        hopefully centering as well -- This is not written into the object yet, 
        but should be soon).
    
        Without specifying a model, fit_image can take a reference image 
        (a cropped deNaNed version of the data) to run correlations. It is 
        recommended that the symmetric part of the data be used to avoid piston
        confusion in scaling. Good luck!
        """
        self.model_in = modelin
        self.weighted = weighted
        self.saveval = savepsfs

        if modelin is None:
            self.vprint(self, "     LG_Model.NRM_Model.fit_image: fittingmodel   no modelin")
            # No model provided - now perform a set of automatic routines

            # A Cleaned up version of your image to enable Fourier fitting for 
            # centering crosscorrelation with FindCentering() and
            # magnification and rotation via new_and_better_scaling_routine().
            if reference==None:
                self.reference = image
                if np.isnan(image.any()):
                    raise ValueError("Must have non-NaN image to "+\
                        "crosscorrelate for scale. Reference "+\
                        "image should also be centered. Get to it.")
            else:
                self.reference=reference

            # IMAGE CENTERING
            """ moved original ~12 lines code to determine_center routine...
            """
            determine_center(self, centering) # populates self.best_center



            self.new_and_better_scaling_routine(self.reference, 
                    scaleguess=self.pixel, rotstart=rotguess,
                    centering=self.bestcenter, fitswrite=self.saveval)

            self.pixscale_measured=self.pixel
            self.vprint(self, "pixel scale (mas):", utils.rad2mas(self.pixel))
            self.fov=image.shape[0]
            self.fittingmodel=self.make_model(self.fov, bandpass=self.bandpass, 
                            over=self.over, rotate=True,
                            psf_offset=self.bestcenter, 
                            pixscale=self.pixel)
        else:
            # This is the standard implaneia path on JWST NIRISS slice data 
            #
            self.vprint(self, "    **** LG_Model.NRM_Model.fit_image: fittingmodel=modelin")
            self.fittingmodel = modelin
            
        # working code 2022 next line...
        #self.soln, self.residual, self.cond,self.linfit_result = \
        #        leastsqnrm.matrix_operations(image, self.fittingmodel, verbose=False, dqm=dqm)

        #######################################################33  very old snippet below
        #######################################################33  very old snippet below
        if self.weighted is False:
            self.soln, self.residual, self.cond, self.linfit_result = \
                leastsqnrm.matrix_operations(image, self.fittingmodel, verbose=False, dqm=dqm)
        else:
            self.soln, self.residual, self.cond, self.singvals = leastsqnrm.weighted_operations(image, \
                                        self.fittingmodel, verbose=False, dqm=dqm)
        #######################################################33  inew code line above


        self.vprint(self, "NRM_Model Raw Soln:")
        self.vprint(self, self.soln)

        self.rawDC = self.soln[-1]
        self.flux = self.soln[0]
        self.soln = self.soln/self.soln[0]

        # fringephase now in radians
        self.fringeamp, self.fringephase = leastsqnrm.tan2visibilities(self.soln)
        self.fringepistons = utils.fringes2pistons(self.fringephase, len(self.ctrs))
        self.redundant_cps = leastsqnrm.redundant_cps(self.fringephase, N=self.N)
        self.redundant_cas = leastsqnrm.return_CAs(self.fringeamp, N=self.N)


    # LG++ with sim data - don't use this cos you already found center in nrm_core
    def determine_center(self, centering): # mostly for internal use from fit_image...
            self.vprint(self, "\n    **** LG_Model.NRM_Model.determine_center: centering {}".format(centering))
            sys.exit("    **** LG_Model.NRM_Model.determine_center")
            # First find the fractional-pixel centering
            if centering== "auto":
                if hasattr(self.bandpass, "__iter__"):
                    self.auto_find_center(
                        "centermodel_poly_{0}mas.fits".format(utils.rad2mas(self.pixel)))
                else:
                    self.auto_find_center(
                        "centermodel_{0}m_{1}mas.fits".format(self.bandpass, \
                        utils.rad2mas(self.pixel)))
                self.bestcenter = 0.5-self.over*self.xpos, 0.5-self.over*self.ypos
                sys.exit("    **** NRM_Model.fit_image: {}".format(self.bestcenter))
            else:
                self.bestcenter = centering


    def plot_model(self, show=False, fits_true=0):
        """
        plot_model makes an image from the object's model and fit solutions.
        """
        try:
            self.modelpsf = np.zeros((self.fov,self.fov))
        except:
            self.modelpsf = np.zeros((self.fov_sim, self.fov_sim))

        for ind, coeff in enumerate(self.soln):
            self.modelpsf += self.flux*coeff * self.fittingmodel[:,:,ind]
        if fits_true:
            # no reason to make an hdu list if not saving a fits file
            hdulist = fits.PrimaryHDU()
            hdulist.data=self.modelpsf
            #hdulist.writeto(fits, clobber=True)
        else:
            hdulist=None
        #hdulist.header.update('PIXEL', self.pixel)
        #hdulist.header.update('WAVL', self.bandpass[0])
        #hdulist.header.update('BANDSIZE', len(self.bandpass))
        #hdulist.header.update('WAVSTEP', self.bandpass[1] - self.bandpass[0])
            #hdulist.data.update('ROTRAD', self.rot_measured)
        #except AttributeError:
        #   print "A solved model has not been created for this object"
        if show:
            import matplotlib.pyplot as plt
            plt.imshow(self.modelpsf, interpolation='nearest', cmap='gray')
            plt.show()
        return self.modelpsf, hdulist

    def perfect_from_model(self, filename="perfect_from_model.fits"):
        """
        perfect_from_model makes an image with zero pistons from the model.
        this is useful for checking that the model matches a perfect analytic
        psf simulation (e.g. like one generated from simulate_mono).
        """
        self.perfect_model= np.zeros((self.fov, self.fov))
        nterms = self.N*(self.N-1) + 2
        self.perfect_soln = np.zeros(nterms)
        self.perfect_soln[0] = 1
        self.perfect_soln[-1] = 0
        for ii in range(len(self.perfect_soln)//2-1):
            self.perfect_soln[2*ii+1] = 1.
        for ind, coeff in enumerate(self.perfect_soln):
            #self.perfect_from_model += coeff * self.model[:,:,ind]
            self.perfect_model += coeff * self.fittingmodel[:,:,ind]
        fits.PrimaryHDU(data=self.perfect_model).writeto(self.datapath + \
                                  filename, \
                                  clobber=True)
        return self.perfect_model

 

    def auto_find_center(self, modelfitsname, overwrite=0):
        """
        This is the major method in this driver to be called. It is basically
        Deepashri's cormat driver & peak finder.

        Takes an image at the detector pixel scale and a fits file name
        * tries to read in the fits file
        * if there's no file it will go on to generate an oversampled PSF
        * if the data is a cube it will generate a cube of oversampled PSFs
        * if not it will generate a single array of oversampled PSF
        * then calculates cormat & finds the peak.

        returns center location on the detector scale.

        *************************************************************
        ***                                                       ***
        ***   NOTE: to plug this centering back into NRM_Model:   ***
        ***   ( 0.5 - model_oversampling*ctr[0],                  ***
        ***     0.5 - model_oversampling*ctr[1] )                 ***
        ***                                             LG+       ***
        *************************************************************
        This is different in LG++: 
        In nrm_core.fit_fringes_single_integration() I experimented with:
        nrm.bestcenter = 0.5-nrm.over*nrm.xpos, 0.5-nrm.over*nrm.ypos  ################ AS LG+
        nrm.bestcenter =    -nrm.over*nrm.xpos,    -nrm.over*nrm.ypos  ################ AS try in LG++
        nrm.bestcenter =              nrm.xpos,              nrm.ypos  ################ AS try in LG++  Works!
        *************************************************************

        """
        self.pscale_rad = self.pixel
        self.pscale_mas = utils.rad2mas(self.pscale_rad)
        _npix = self.reference.shape[1] +2
        if ( (not os.path.isfile(modelfitsname)) or (overwrite == 1)):
            # Creates a new oversampled model, default is pixel-centered
            self.simulate(fov=_npix, over=self.over, bandpass=self.bandpass)

            hdulist=fits.PrimaryHDU(data=self.psf_over)
            hdulist.header[""] = ("", "Written from auto_find_center method")
            #hdulist.header.update()
            hdulist.writeto(self.datapath+modelfitsname, clobber=True)
        else:
            # Looks for this file to read in if it's already been written and overwrite flag not set
            try:
                self.read_model(modelfitsname)
            except:
                self.psf_over = 0
            if self.psf_over.shape[0] ==self.reference.shape[0]:
                pass
            else:
                # if overwrite flag not set, but read model doesn't match, just make a new one
                self.simulate(fov=_npix, over=self.over, bandpass=self.bandpass)

        # finds correlation matrix between oversampled and detector psf
        self.cormat = utils.crosscorrelatePSFs(self.reference, self.psf_over, self.over)
        self.find_location_of_peak()

    def read_model(self, modelfitsname):
        self.psf_over = fits.getdata(self.datapath+modelfitsname)
        return self.psf_over

    def find_location_of_peak(self):
        peak_location=np.where(self.cormat==self.cormat.max())
        y_peak=peak_location[0][0]
        x_peak=peak_location[1][0]
        y_peak_ds9=y_peak+1
        x_peak_ds9=x_peak+1
        self.x_offset =  self.over - x_peak
        self.y_offset =  self.over - y_peak
        self.xpos, self.ypos = self.x_offset/float(self.over), self.y_offset/float(self.over)

        self.vprint(self, "x_peak_python,y_peak_python", x_peak,y_peak)
        self.vprint(self, "x_peak_ds9,y_peak_ds9", x_peak_ds9,y_peak_ds9)
        self.vprint(self, "first value is x, second value is y")
        self.vprint(self, "printing offsets from the center of perfect PSF in oversampled pixels...")
        self.vprint(self, "x_offset, y_offset", self.x_offset, self.y_offset)
        self.vprint(self, "printing offsets from the center of perfect PSF in detector pixels...")
        self.vprint(self, "x_offset, y_offset", self.xpos,self.ypos)         


def save(nrmobj, outputname, savdir = ""):
    """
    Probably don't need to use this unless have run a fit.
    This is only to save fitting parameters and results right now.
    """
    import json
    class savobj: 
        def __init__(self):
            return None
    savobj.test = 1
    with open(r"{0}.ffo".format(savdir, outputname), "wb") as output_file:
        json.dump(savobj, output_file)
    self.vprint(self, "success!")

    # init stuff
    savobj.pscale_rad, savobj.pscale_mas = nrmobj.pixel, utils.rad2mas(nrmobj.pixel)
    savobj.holeshape, savobj.ctrs, savobj.d, savobj.D, savobj.N, \
        savobj.datapath, savobj.refdir  =   nrmobj.holeshape, nrmobj.ctrs, nrmobj.d, \
                                            nrmobj.D, nrmobj.N, nrmobj.datapath, nrmobj.refdir

    if hasattr(nrmobj, "refpsf"):
        savobj.refpsf, savobj.rot_best = nrmobj.refpsf, nrmobj.rot_measured
    if hasattr(nrmobj, "fittingmodel"):
        # details
        savobj.weighted, savobj.pixweight, savobj.bestcenter, \
            savobj.bandpass, savobj.modelctrs, savobj.over,\
            savobj.modelpix  =  nrmobj.weighted, nrmobj.pixweight, nrmobj.bestcenter, \
                                nrmobj.bandpass, nrmobj.modelctrs, nrmobj.over, nrmobj.modelpix
        # resulting arrays
        savobj.modelmat, savobj.soln, \
            savobj.residual, savobj.cond, \
            savobj.rawDC, savobj.flux, \
            savobj.fringeamp, savobj.fringephase,\
            savobj.cps, savobj.cas  =   nrmobj.fittingmodel, nrmobj.soln, nrmobj.residual, \
                                nrmobj.cond, nrmobj.rawDC, nrmobj.flux, nrmobj.fringeamp, \
                                nrmobj.fringephase, nrmobj.redundant_cps, nrmobj.redundant_cas
        if not hasattr(nrmobj, "modelpsf"):
            nrmobj.plot_model()
        savobj.modelpsf = nrmobj.modelpsf
    with open(r"{0}.ffo".format(savdir, outputname), "wb") as output_file:
        pickle.dump(savobj, output_file)

def save_oifits(obj, ofn, mode="single"):
    from write_oifits import OIfits
    kws = {'path':obj.datapath,\
            'arrname':'mask', \
            'PARANG':0.0, 'PA':0.0, 'flip':False}
    oif = OIfits(obj.mask,mykeywords)
    oif.mode = mode
    # wavelength infor for a single slice
    oif.nwav = 1
    if not hasattr(bandpass, "__iter__"):
        oif.wls = np.array([obj.bandpass,])
        oif.eff_band = 0.01 # close enough for this
    else:
        # need to convert wght,wl list into wls
        oif.wls=np.array(obj.bandpass)[:,1]
        oif.eff_band = abs(oif.wls[-1] - oif.wls[0]) / oif.wls[oif.wls.shape[0]/2]
    oif.wavs = oif.wls
    oif.oiwav = oifits.OI_WAVELENGTH(oif.wavs, eff_band = oif.eff_band)
    oif.wavs = {oif.isname:oif.oiwav}
    oif.dummytables()
    # manually set v2
    oif.v2 = obj.fringeamp**2
    oif.v2_err = np.ones(oif.v2) # this is raw measurement, don't know errors yet
    oif.v2flag=np.resize(False, (len(oif.v2)))
    oif.oivis2=[]
    for qq in range(len(oif.v2)):
        vis2data = oifits.OI_VIS2(oif.timeobs, oif.int_time, oif.v2[qq],\
                                  oif.v2_err[qq], oif.v2flag[qq], oif.ucoord[qq],\
                                  oif.vcoord[qq], oif.oiwav, oif.target, \
                                  array=oif.array, station=[oif.station, oif.station])
        oif.oivis2=np.array(oif.oivis2)
    
    oif.oi_data()
    #print oif.oivis2
    #print oif.oit3
    oif.write(ofn)
    return None

def goodness_of_fit(data, bestfit, diskR=8, save=False):
    mask = np.ones(data.shape) +AG.makedisk(data.shape[0], 2) -\
                    AG.makedisk(data,shape[0], diskR)
    difference = np.ma.masked_invalid(mask*(bestfit-data))
    masked_data = np.ma.masked_invalid(mask*data)
    """
    gof = sum(abs(difference[support][data != np.nan])) / \
            sum(data[support][data != np.nan])
    """
    return abs(difference).sum() / abs(masked_data).sum()

def image_plane_correlate(data,model):
    """
    From A. Greenbaum's 'centering_correlate.py'
    Modified so that instead of throwing NaNs to 0, it masks them out.
    """
    multiply = np.ma.masked_invalid(model*data)
    if True in np.isnan(multiply):
        raise ValueError("data*model produced NaNs,"\
                    " please check your work!")
    self.vprint(self, "masked data*model:", multiply, "model sum:", model.sum())
    return multiply.sum()/((np.ma.masked_invalid(data)**2).sum())
    #return (multiply/(np.ma.masked_invalid(data)**2))
    #multiply = np.nan_to_num(model*data)
    #return multiply.sum()/(np.nan_to_num(data).sum()**2)

def run_data_correlate(data, model):
    sci = data
    self.vprint(self, "shape sci",np.shape(sci))
    self.vprint(self, "shape model", np.shape(model))
    return utils.rcrosscorrelate(sci, model)
