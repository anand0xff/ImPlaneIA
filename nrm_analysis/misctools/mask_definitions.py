#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import math
import sys
import time
from nrm_analysis.misctools.utils import makedisk, rotate2dccw
from astropy.io import fits
from copy import copy
"""

================
NRM_mask_definitions
================

A module defining mask geometry in pupil space.

Mask names (str):
  * jwst_g7s6c
  * jwst_g7s6
  * jwst


Code by 
Alexandra Greenbaum <agreenba@pha.jhu.edu> and 
Anand Sivaramakrishnan <anand@stsci.edu>
Dec 2012

"""


m = 1.0
mm = 1.0e-3 * m
um = 1.0e-6 * m


class NRM_mask_definitions():

    def __init__(self, maskname=None, rotdeg=None, holeshape="circ", rescale=False,\
                 verbose=False, chooseholes=None):

        if verbose: print("NRM_mask_definitions(maskname,...:" + maskname)
        if maskname not in ["jwst_g7s6", "jwst_g7s6c", "jwst" ]:
            raise ValueError("mask not supported")
        if holeshape == None:
            holeshape = 'circ'
        if verbose: print(holeshape)
       
        if holeshape not in ["circ", "hex",]:
            raise ValueError("Unsupported mask holeshape" + maskname)
        self.maskname = maskname
        if verbose:
            print("\n\t=====================================")
            print("Mask being created" + self.maskname)


        if self.maskname == "jwst_g7s6c" or maskname == "jwst_g7s6" or maskname == "jwst":
            """ activeD and D taken from webbpsf-data/NIRISS/coronagraph/MASK_NRM.fits"""
            if verbose: print('self.maskname = "jwst_g7s6c"')
            self.hdia, self.ctrs = jwst_g7s6c(chooseholes=chooseholes) # 
            self.activeD =  6.559*m # webbpsf kwd DIAM  - not a 'circle including all holes'
            self.OD = 6.610645669291339*m # Full pupil file size, incl padding, webbpsf kwd PUPLDIAM
            if rotdeg is not None:
                self.rotdeg = rotdeg
                
        else:
            print("\tmask_definitions: Unknown maskname: check back later")

    # make image at angular pixel scale, at given wavelength/bandpass
    # choose pupil pixel scale 
    # image oversampling before rebinning
    # possibly put in OPD

    def createpupilarray(self, puplscal=None, fitsfile=None):

        pupil = np.zeros((int(np.ceil(self.OD/puplscal)), int(np.ceil(self.OD/puplscal))))

        pupil = pupil + \
                makedisk(s=pupil.shape, c=(pupil.shape[0]/2.0 - 0.5, pupil.shape[1]/2.0 - 0.5),
                                   r=0.5*self.OD/puplscal, t=np.float64, grey=0) - \
                makedisk(s=pupil.shape,  c=(pupil.shape[0]/2.0 - 0.5, pupil.shape[1]/2.0 - 0.5),
                                   r=0.5*self.ID/puplscal, t=np.float64, grey=0) 

        hdu = fits.PrimaryHDU ()
        hdu.data = pupil.astype(np.uint8)
        hdu.header.update("PUPLSCAL", puplscal, "Pupil pixel scale in m/pixels DL")
        hdu.header.update("PIXSCALE", puplscal, "Pupil pixel scale in m/pixels MDP")
        hdu.header.update("PUPLDIAM", self.OD, "Full pupil file size, incl padding in m")
        hdu.header.update("DIAM", self.activeD, "Active pupil diameter in m") # changed from OD - AS Feb 13
        hdu.header.update("ROTATE", self.rotdeg, "Mask counterclockwise rotation (deg)")
        if fitsfile is not None:
            hdu.writeto(fitsfile, clobber=True)
        self.fullpupil = pupil.copy()
        self.fullpuplscale = puplscal
        hdulist = fits.HDUList([hdu])
        return hdulist

    def createnrmarray(self, puplscal=None, fitsfile=None, holeid=None, fullpupil=False):
        """ fullpupil is a possibly oversized array, in meters using puplscal """
        if fullpupil:
            D = self.OD # array side size, m
        else:
            D = self.activeD  # light-transmitting diameter, m

        pupil = np.zeros((int(np.ceil(D/puplscal)), int(np.ceil(D/puplscal))))
        print("creating pupil array with shape ", pupil.shape)

        factor=1
        #modify to add hex holes later
        for ctrn, ctr in enumerate(self.ctrs):
            if holeid:
                factor = ctrn +1
            # convert to zero-at-corner, meters
            center = (0.5*pupil.shape[0] + ctr[0]/puplscal - 0.5, 0.5*pupil.shape[1] + ctr[1]/puplscal - 0.5)
            pupil = pupil + \
            makedisk(s=pupil.shape, c=center,
                       r=0.5*self.hdia/puplscal, t=np.float64, grey=0)* factor
        self.nrmpupil = pupil.copy()
        self.puplscale = puplscal

        hdu = fits.PrimaryHDU ()
        hdu.data = pupil.astype(np.uint8)
        hdu.header.update("PUPLSCAL", puplscal, "Pupil pixel scale in m/pixels MDP")
        hdu.header.update("PIXSCALE", puplscal, "Pupil pixel scale in m/pixels DL")
        hdu.header.update("PUPLDIAM", D, "Full pupil file size, incl padding in m")
        hdu.header.update("DIAM", self.activeD, "Active pupil diameter in m")
        if hasattr(self, 'rotate'):
            hdu.header.update("ROTATE", self.rotdeg, "Mask counterclockwise rotation (deg)")
        (year, month, day, hour, minute, second, weekday, DOY, DST) =  time.gmtime()
        hdu.header.update("CODESRC", "NRM_mask_definitions.py", "Anand S. and Alex G.")
        hdu.header.update("DATE", "%4d-%02d-%02dT%02d:%02d:%02d" % \
                         (year, month, day, hour, minute, second), "Date of calculation")
        if fitsfile is not None:
            hdu.writeto(fitsfile, clobber=True)
        hdulist = fits.HDUList([hdu])
        return hdulist


    def showmask(self):
        """
        prints mask geometry, 
        returns diameter of smallest centered circle (D) enclosing live mask area
        """
        print("\t%s" % self.maskname)
        print("\tholeD\t%+6.3f" % self.hdia)

        print("\t\t  x/m  \t  y/m        r/m     r+h_rad/m  2(r+h)/m")
        radii = []
        for ctr in self.ctrs:
            print("\t\t%+7.3f\t%+7.3f" % (ctr[0], -1.0*ctr[1]), end=' ')
            radii.append(math.sqrt(ctr[0]*ctr[0] + ctr[1]*ctr[1]))
            print("    %.3f " % radii[-1], end=' ')
            print("    %.3f " % (radii[-1] + self.hdia/2.0), end=' ')
            print("    %.3f " % (2.0*radii[-1] + self.hdia))


        print("\t2X max (r+h) \t%.3f m" % (2.0*(max(radii) + 0.5*self.hdia)))
        print()
        return 2.0*(max(radii) + 0.5*self.hdia) 

""" Mathilde Beaulieu
    eg. Thu, Jun 18, 2009 at 06:28:19PM
    
    Thank you for the drawing. It really helps!
    The distance between the center of the 2 segments in your drawing does not
    match exactly with the distance I have (1.32 instead of 1.325).
    Could you please check if I have the good center coordinates?
    XY - PUPIL
    0.00000     -2.64000
    -2.28631      0.00000
    2.28631     -1.32000
    -2.28631      1.32000
    -1.14315      1.98000
    2.28631      1.32000
    1.14315      1.98000
    
    where y is the direction aligned with the spider which is not collinear with
    any of pupil edges (it is not the same definition as Ball).
    
    Thank you,
    
    Regards,
    
    Mathilde
    
n.b. This differs from the metal-mask-projected-to-PM-space with
Zheng Hai (Com Dev)'s mapping communicated by Mathilde Beaulieu to Anand.
This mapping has offset, rotation, shrink x, magnification.
Reference is a JWST Tech Report to be finalized 2013 by Anand.

jwst_g7s6_centers_asdesigned function superceded in LG++:
"""
def jwst_g7s6_centers_asbuilt(chooseholes=None):  # was jwst_g7s6_centers_asdesigned

    holedict = {} # as_built names, C2 open, C5 closed, but as designed coordinates
    # Assemble holes by actual open segment names (as_built).  Either the full mask or the
    # subset-of-holes mask will be V2-reversed after the as_designed centers  are defined
    # Debug orientations with b4,c6,[c2]
    allholes = ('b4','c2','b5','b2','c1','b6','c6')
    b4,c2,b5,b2,c1,b6,c6 = ('b4','c2','b5','b2','c1','b6','c6')
    #                                              design  built
    holedict['b4'] = [ 0.00000000,  -2.640000]       #B4 -> B4
    holedict['c2'] = [-2.2863100 ,  0.0000000]       #C5 -> C2
    holedict['b5'] = [ 2.2863100 , -1.3200001]       #B3 -> B5
    holedict['b2'] = [-2.2863100 ,  1.3200001]       #B6 -> B2
    holedict['c1'] = [-1.1431500 ,  1.9800000]       #C6 -> C1
    holedict['b6'] = [ 2.2863100 ,  1.3200001]       #B2 -> B6
    holedict['c6'] = [ 1.1431500 ,  1.9800000]       #C1 -> C6

    # as designed MB coordinates (Mathilde Beaulieu, Peter, Anand).
    # as designed: segments C5 open, C2 closed, meters V2V3 per Paul Lightsey def
    # as built C5 closed, C2 open
    #
    # undistorted pupil coords on PM.  These numbers are considered immutable.  
    # as designed seg -> as built seg in comments each ctr entry (no distortion)
    if chooseholes: #holes B4 B5 C6 asbuilt for orientation testing
        print("\n  chooseholes creates mask with JWST as_built holes ", chooseholes)
        #time.sleep(2)
        holelist = []
        for h in allholes:
            if h in chooseholes: 
                holelist.append(holedict[h])
        ctrs_asdesigned = np.array( holelist )
        #if len(allholes) == 1: ctrs_asdesigned = ctrs_asdesigned[..., None] # make 2d
    else:
        # the REAL THING - as_designed 7 hole, m in PM space, no distortion  shape (7,2)
        ctrs_asdesigned = np.array( [ 
                [ 0.00000000,  -2.640000],       #B4 -> B4  as-designed -> as-built mapping
                [-2.2863100 ,  0.0000000],       #C5 -> C2
                [ 2.2863100 , -1.3200001],       #B3 -> B5
                [-2.2863100 ,  1.3200001],       #B6 -> B2
                [-1.1431500 ,  1.9800000],       #C6 -> C1
                [ 2.2863100 ,  1.3200001],       #B2 -> B6
                [ 1.1431500 ,  1.9800000]    ] ) #C1 -> C6

    # Preserve ctrs.as-designed (treat as immutable)
    # Reverse V2 axis coordinates to close C5 open C2, and others follow suit... 
    # preserve cts.as_built  (treat as immutable)
    print(type(ctrs_asdesigned), ctrs_asdesigned.shape, ctrs_asdesigned)
    ctrs_asbuilt = ctrs_asdesigned.copy()

    # create 'live' hole centers in an ideal, orthogonal undistorted xy pupil space,
    # eg maps open hole C5 in as_designed to C2 as_built, eg C4 unaffacted....
    print(ctrs_asbuilt[:,0])
    ctrs_asbuilt[:,0] *= -1

    # LG++ rotate hole centers by 90 deg to match MAST o/p DMS PSF with 
    # no affine2d transformations 8/2018 AS
    # LG++ The above aligns the hole patern with the hex analytic FT, 
    # flat top & bottom as seen in DMS data. 8/2018 AS
    ctrs_asbuilt = rotate2dccw(ctrs_asbuilt, np.pi/2.0) # overwrites attributes

    # create 'live' hole centers in an ideal, orthogonal undistorted xy pupil space,
    return ctrs_asbuilt * m


def jwst_g7s6c(chooseholes=None):
    # WARNING! JWST CHOOSEHOLES CODE NOW DUPLICATED IN LG_Model.py WARNING! ###
    f2f = 0.80 * m # m flat to flat
    return f2f, jwst_g7s6_centers_asbuilt(chooseholes=chooseholes)



if __name__ == "__main__":

    # JWST G7S6 circular 0.8m dia holes...
    nrm = NRM_mask_definitions("gpi_g10s40")
    PUPLSCAL= 0.006455708661417323 # scale (m/pixels) from webbpsf-data/NIRISS/coronagraph/MASK_NRM.fits
    # for jwst-g7s6c  fullpupil=True gets us a file like that in webbpsf-data...
    #maskobj = nrm.createnrmarray(puplscal=PUPLSCAL,fitsfile='g7s6c.fits' % r, fullpupil=True)
    print(nrm.activeD)
    sys.exit()
    maskobj = nrm.createnrmarray(puplscal=PUPLSCAL,
                                 fitsfile='/Users/anand/Desktop/jwst_g7s6c_which.fits',
                                 fullpupil=True,
                                 holeid=True)
    maskobj = nrm.createnrmarray(puplscal=PUPLSCAL,
                                 fitsfile='/Users/anand/Desktop/jwst_g7s6c.fits',
                                 fullpupil=True,
                                 holeid=False)

