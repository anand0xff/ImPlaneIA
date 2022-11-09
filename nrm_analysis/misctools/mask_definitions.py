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
  * gpi_g10s40
  * jwst_g7s6c
  * jwst_g7s6
  * p1640
  * keck_nirc2
  * pharo


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
        if maskname not in ["gpi_g10s40",  "jwst_g7s6", "jwst_g7s6c", "visir_sam", \
                            "p1640", "keck_nirc2", "pharo", "NIRC2_9NRM"]:
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

        if self.maskname == "gpi_g10s40":
            self.hdia, self.ctrs = gpi_g10s40(rescale=rescale)
            #self.rotdeg = 115.0 # By inspection of I&T data Dec 2012
            #if rotdeg is not None:
            #    self.rotdeg = rotdeg
            #    print("rotating by {0} deg".format(rotdeg))
            #else:
            #    print("rotating by 115.0 deg -- hard coded.")
            #self.ctrs = rotate2dccw(self.ctrs, self.rotdeg*np.pi/180.0)
            #self.rotate = self.rotdeg
            self.activeD = self.showmask() # calculates circle dia including all holes
            self.OD = 7770.1 * mm   # DGS = 7770.1 * mm with M2 cardboard baffle
                        # GS OD from GPI fundamental values
            self.ID = 1023.75 * mm  # Formed by M2 diameter
                        # (Bauman GPI_Optical_Fundamental_Values.doc)

        elif self.maskname == "jwst_g7s6c":
            """ activeD and D taken from webbpsf-data/NIRISS/coronagraph/MASK_NRM.fits"""
            if verbose: print('self.maskname = "jwst_g7s6c"')
            self.hdia, self.ctrs = jwst_g7s6c(chooseholes=chooseholes) # 
            self.activeD =  6.559*m # webbpsf kwd DIAM  - not a 'circle including all holes'
            self.OD = 6.610645669291339*m # Full pupil file size, incl padding, webbpsf kwd PUPLDIAM
            if rotdeg is not None:
                self.rotdeg = rotdeg
                
        elif self.maskname == "jwst_g7s6":
            print("\tnot finished")

        elif self.maskname == "visir_sam":
            """Mask dimensions from Eric Pantin"""
            self.hdia, self.ctrs = visir_sam(rescale=rescale)
            self.rotdeg = 16.5 # By inspection of data
            if rotdeg is not None:
                self.rotdeg += rotdeg
                print("rotating by {0} + 9 (hardcoded) deg".format(rotdeg))
            else:
                print("rotating by 9 deg -- hard coded")
            self.ctrs = rotate2dccw(self.ctrs, self.rotdeg*np.pi/180.0)
            self.rotate = self.rotdeg
            self.activeD = self.showmask() # calculated circle dia including all holes
            self.OD = 8115.0 * mm   # DVLT = 8115.0 * mm -- but what is the M2 dia??
                                    # From Eric Pantin Document
            self.ID = 1000.0 * mm   # Don't know this, but this number shouldn't matter

        elif self.maskname == "NIRC2_9NRM":
            """Mask dimensions from Steph Sallum?"""
            self.hdia, self.ctrs = keck_nrm()
            self.rotdeg = 28.0 # By inspection of data
            if rotdeg is not None:
                self.rotdeg += rotdeg
                print("rotating by {0} + 9 (hardcoded) deg".format(rotdeg))
            else:
                print("rotating by 9 deg -- hard coded")
            self.ctrs = rotate2dccw(self.ctrs, self.rotdeg*np.pi/180.0)
            self.rotate = self.rotdeg
            self.activeD = self.showmask() # calculated circle dia including all holes
            self.OD = 10.0   # DVLT = 8115.0 * mm -- but what is the M2 dia??
                                    # From Eric Pantin Document
            self.ID = 1.0    # Don't know this, but this number shouldn't matter

        else:
            print("\tmask_definitions: Unknown maskname: check back later")

    # make image at angular pixel scale, at given wavelength/bandpass
    # choose pupil pixel scale 
    # image oversampling before rebinning
    # possibly put in OPD

    def get_scale(self, band='Hband', date='2013_Jan15', slc=0, fldr = ''):
        # write now file is J through H, can we somehow specify the band and use that to get this info by just putting in a slice?
        if band == 'Hband':
            start = 0
            end = 37
        elif band == 'Yband':
            start = 37
            end = 74
        elif band == 'Jband':
            start = 74
            end = None
        else:
            print('Must specify valid band: Yband, Jband, or Hband')
        fh='wlcorrection_'
        info = np.loadtxt(fldr+fh+date+'.txt')
        print('Returning scaled wavelengths from '+date)
        return info[start:end, 2]

    def get_rotation(self, band = 'Hband', date='2013_Jan15', slc=None, fldr=''):
        if band == 'Hband':
            start = 0
            end = 37
        elif band == 'Yband':
            start = 37
            end = 74
        elif band == 'Jband':
            start = 74
            end = None
        else:
            print('Must specify valid band: Yband, Jband, or Hband')
        fh='rotcorrection_'
        info = np.loadtxt(fldr+fh+date+'.txt')
        return info[start:end,1]

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
        """
        anand@silmakpro.local:172  ./gpipoppy.py
        All X,Y dimensions in m from center of mask
        All hole DIAMETERS are in m:
        
        G10S40lenox     holeD   +0.596
              x/m     y/m        r/m     r+h_rad/m  2(r+h)/m
             -0.687  +2.514     2.606      2.904      5.808 
             -0.252  +3.362     3.372      3.670      7.340 
             +1.822  +0.157     1.829      2.127      4.254 
             +2.342  -0.644     2.429      2.727      5.453 
             -2.862  -1.733     3.346      3.644      7.287 
             -0.869  -3.288     3.401      3.699      7.398 
             -3.026  +1.568     3.408      3.706      7.411 
             +2.692  +1.855     3.269      3.567      7.134 
             +3.297  -0.596     3.350      3.648      7.297 
             +1.036  -3.192     3.356      3.654      7.308 
        2X max (r+h)    7.411 m
        """


"anand@stsci.edu Feb 2011"

"""
    Barnaby Norris <bnorris@physics.usyd.edu.au>
to  Peter Tuthill <p.tuthill@physics.usyd.edu.au>
cc  Anand Sivaramakrishnan <anand0xff@gmail.com>,
James Lloyd <jpl@astro.cornell.edu>,
"michael.ireland@sydney.edu.au" <mirelandastro@gmail.com>,
"gekko@physics.usyd.edu.au" <gekko@physics.usyd.edu.au>,
"frantz@naoj.org" <frantz@naoj.org>,
Laurent Pueyo <lap@pha.jhu.edu>,
David Lafreniere <david@astro.umontreal.ca>
date    Tue, Feb 8, 2011 at 6:24 PM

Hi Everyone,

I've been working on the mask designs for Peter, and we've whittled
down the various mask solutions to a 10, 12 and 15 hole mask -
diagrams and the corresponding Fourier coverage are attached. I've
used the diagram Anand sent (GPI_LyotBadActuators.png) as a guide and
avoided the bad actuators by the size of the blacked-out circles in
that diagram, which corresponds to 2 actuator spacings away (these
obstructions are shown in green on the diagrams)..  The measurements
are in mm and based on the inside of the black filled area (in Anand's
png file) being the Lyot stop outer diameter (ie 9.532mm). The holes
are sized based on a slight relaxation of strict non-redundancy, with
the hole radius being 0.33 times the minimum baseline.

The measurements are included below - please let me know if you need
anything else.

Cheers
Barnaby


   GPI 10 hole mask, soln 40

   X Position    Y Position
   -0.580002     -3.13987
     0.00000     -4.14446
     2.32001    -0.126087
     2.90001     0.878506
    -3.48001      1.88310
    -1.16000      3.89229
    -3.48001     -2.13527
     3.48001     -2.13527
     4.06002     0.878506
     1.16000      3.89229

Hole Radius:     0.382802


"""


# go from Barnaby's first LS design to cut-metal-coords in GPI PPM
# mag is slight expansion (~0.5%) then 1.2 mag factor to PPM
def gpi_g10s40_asmanufactured(mag):
    """ In PPM metal space - measured cooordinates from sample G10S40"""
    holedia = 0.920*mm
    holectrs = [
    [-1.061*mm,   -3.882*mm],
    [-0.389*mm,   -5.192*mm],
    [ 2.814*mm,   -0.243*mm],
    [ 3.616*mm,    0.995*mm],
    [-4.419*mm,    2.676*mm],
    [-1.342*mm,    5.077*mm],
    [-4.672*mm,   -2.421*mm],
    [ 4.157*mm,   -2.864*mm],
    [ 5.091*mm,    0.920*mm],
    [ 1.599*mm,    4.929*mm],
    ]

    # design2metal mag
    holedia = holedia * mag
    print(mag)
    ctrs = []
    #REVERSE = -1 # Flip y dimensions to match I&T data Dec 2012
    REVERSE = 1 # With new affine updates, need to reverse the reverse March 2019
    for r in holectrs:
        ctrs.append([r[0]*mag, r[1]*mag*REVERSE])
    # return cut-the-metal coords per Lenox PPM mm spec in meters on PM
    ctrs_asbuilt = copy(ctrs)

    # March 2019:
    # Rotate hole centers by 90 deg to match GPI detector with new coordinate handling
    # no affine2d transformations 8/2018 AS
    ctrs_asbuilt = rotate2dccw(ctrs_asbuilt, 160*np.pi/180.0) # overwrites attributes

    return holedia, ctrs_asbuilt


def gpi_mag_asdesigned():

    #logging.basicConfig(level=logging.DEBUG,format='%(name)-10s: %(levelname)-8s %(message)s')
    datapath='/Users/anand/data/NRM/GPI/'

    """ 
    returns demag (dimensionless)

    pupil dimensions * demag gives manufactured dimensions
    """

    """
    COR-SATR...
      The coronagraph module shall achieve the specifications at an inner working distance
      of 4lam/D (goal 3.5 lam/D), where D is the telescope entrance diameter of 7.7701 m.
    """

    DGN = 7908.0
    DGS = 7770.1 * mm # with M2 baffle GS OD Bauman  http://dms.hia.nrc.ca/view.php?fDocumentId=2164
    D = DGS
    d = 11.998 * mm # with M2 baffle GS OD Bauman  http://dms.hia.nrc.ca/view.php?fDocumentId=2164
    #d = 11.68 * mm # March 2019 update based on manufactured values.
    #demag = 0.99*d/D  # about 1/800...
    demag = d/D  # about 1/800...

    dppm = 11.671 * mm  # Precision Optical or Aktiwave dapod 11.68

    flip = "_flip"

    print("""" 
    This program (gpipoppy.py) uses DGS = 7770.1 * mm with M2 cardboard baffle
    GS OD from GPI Fundamental Values (Bauman GPI_Optical_Fundamental_Values.doc)
    
        http://dms.hia.nrc.ca/view.php?fDocumentId=2164

    and Lenox Laser measured hole diameter of G40S10 sample design,
    with average hole size in 

        LenoxSTScI_delivery_APOD_NRM10.[pdf xlsx]   (also on HIA's KT)

    All X,Y dimensions in m from center of mask in PM space
    All hole DIAMETERS are in m:\n""")

    ##### Preliminary set-up:- design-to-metal scale
    ##### Multiply design by MAG to get metal coords, origin at part center
    ####assumedLSOD = 9.532 * mm # Barnaby's email Feb 2011
    ####correctLSOD = 9.571 * mm # source - Anand's order of GS version 3 apodizers to CTM
    ####magLS2PPM = 11.790/9.825  # GSOD@PPM/GSOD@Lyot source - Remi's COR dimensions final designs.pdf, 1.2 exactly
    ####magBarnaby2LS = correctLSOD / assumedLSOD # slight mag, about 0.5%
    ####magBarnaby2PPM = magBarnaby2LS * magLS2PPM
    ####MAG = magBarnaby2PPM
    ####print "DESIGN to PPM magnification is %.4f\n" % MAG
    print(demag)
    return demag

def gpi_g10s40(rescale=False):
    """
    Multiply by the 'rescale' factor to adjust hole sizes and centers in entrance pupil (PM)
    (Magnify the physical mask coordinates up to the primary mirror size)
    """
    demag = gpi_mag_asdesigned()
    if rescale:
        demag = demag/rescale # rescale 1.1 gives a bigger mask in PM pupil space
    print ("gpi_g10s4...")
    hdia, ctrs = gpi_g10s40_asmanufactured(1.0/demag) # meters
    return hdia, ctrs

    """  From GPI FPRD 2008    http://dms.hia.nrc.ca/view.php?fDocumentId=1398

       Filter        1/2 pwr       bandwidth
       name        wavelen/um        %
        Y           0.95-1.14        18
        J           1.12-1.35        19
        H           1.50-1.80        18
        K1          1.9-2.19         14
        K2          2.13-2.4         12

        Spectral Resolution 34-36 35-39 44-49 62-70 75-83
        # spectral pixels 12-13 13-15 16-18 18-20
        18-20

        pixels 14mas are nyquist at 1.1
    """     


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
    ctrs_asbuilt = ctrs_asdesigned.copy()

    # create 'live' hole centers in an ideal, orthogonal undistorted xy pupil space,
    # eg maps open hole C5 in as_designed to C2 as_built, eg C4 unaffacted....
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
    #return 0.80*m, jwst_g7s6_centers_asbuilt(chooseholes=chooseholes) #comment out 2019 Aug w/Joel to match webbpsf 0.8.something
    f2f = 0.82 * m # m flat to flat
    return f2f, jwst_g7s6_centers_asbuilt(chooseholes=chooseholes)


def visir_sam_asmanufactured(mag):
    """VISIR has hexagonal holes
        The SAM plate has a main external diameter of 26 mm
    """
    holedia = 2.8*mm # (=side length) => 0.61 m in telescope pupil of 8.115 m
                     # Equivalenent diameter is thus 1.22m
    # pupil footprint center (/= coordinates center in the drawing)
    holectrs = [
    [-5.707*mm,     -2.885*mm],
    [-5.834*mm,      3.804*mm],
    [ 0.099*mm,      7.271*mm],
    [ 7.989*mm,      0.422*mm],
    [ 3.989*mm,     -6.481*mm],
    [-3.790*mm,     -6.481*mm],
    [-1.928*mm,     -2.974*mm]
    ]

    """
    xc, yc = 387., 294.75
    holectrs = =[
    [342.000-xc,    272.000-yc],
    [341.000-xc,    324.750-yc],
    [387.780-xc,    352.080-yc],
    [449.992-xc,    298.080-yc],
    [418.456-xc,    243.648-yc],
    [357.112-xc,    243.648-yc],
    [371.800-xc,    271.296-yc]
    ]

    There is an offset between the mechanical plate and the fottprint of the
    cold stop of - 0.636 mm, which probably explains the need for manual (filing)
    increase of the size of the fixing holes on the size after mounting and
    checking the optical alignment on the instrument.
    """

    # design2metal mag
    holedia = holedia * mag
    print(mag)
    ctrs = []
    REVERSE = -1 # The pupil we measure with VISIR pupil imaging lens is 
                 # [probably] point symmetry reversed w.r.t reality. => OK checked
    for r in holectrs:
        ctrs.append([r[0]*mag*REVERSE, r[1]*mag*REVERSE])
    # return mask coords
    return holedia, ctrs

def visir_mag_asdesigned():

    #logging.basicConfig(level=logging.DEBUG,format='%(name)-10s: %(levelname)-8s %(message)s')
    datapath='/Users/agreenba/data/visir/'

    """ 
    returns demag (dimensionless)

    pupil dimensions * demag gives manufactured dimensions
    """

    """
    The hexagons have radii of 1.4 mm (=side length) => 0.61 m in telescope
    pupil of 8.115 m. Equivalenent diameter is thus 1.22m
    """

    DVLT = 8115. * mm # 
    DTH = 1220.*mm # diam of hex on primary 
    D = DTH
    d = 2.8 * mm # diam of hex in pupil
    demag = d/D  # about 1/800...


    print("""" 
    This program (VISIR.py) uses GVLT = 8115.0 * mm from SAM report c/o Eric Pantin
    (report_SAM_pupil.pdf)
    
    All X,Y dimensions in m from center of mask in PM space
    All hole DIAMETERS are ... :\n""")

    ####print "DESIGN to PPM magnification is %.4f\n" % MAG
    print(demag)
    return demag


def visir_sam(rescale=False):
    """
    Multiply by the 'rescale' factor to adjust hole sizes and centers in entrance pupil (PM)
    (Magnify the physical mask coordinates up to the primary mirror size)
    """
    demag = visir_mag_asdesigned()
    if rescale:
        demag = demag/rescale # rescale 1.1 gives a bigger mask in PM pupil space
    print ("VISIR SAM")
    hdia, ctrs = visir_sam_asmanufactured(1.0/demag) # meters
    return hdia, ctrs

    """  From VISIR documentation

       Filter        1/2 pwr       bandwidth
       name        wavelen/um        %

        Spectral Resolution 34-36 35-39 44-49 62-70 75-83
        # spectral pixels 12-13 13-15 16-18 18-20
        18-20

        pixels ??mas are nyquist at ??um
    """

def keck_nrm(rescale=False):
    """
    Multiply by the 'rescale' factor to adjust hole sizes and centers in entrance pupil (PM)
    (Magnify the physical mask coordinates up to the primary mirror size)
    """
    hdia = 1.098 # m
    # All projected onto the primary in m
    holectrs = np.array([[-3.44,  -2.22],
                    [-4.57 ,  -1.00],
                    [-2.01 ,   2.52],
                    [-0.20 ,   4.09],
                    [ 1.42 ,   4.46],
                    [ 3.19 ,   0.48],
                    [ 3.65 ,  -1.87],
                    [ 3.15 ,  -3.46],
                    [-1.18 ,  -3.01]])

    ctrs = []
    REVERSE = -1 # The pupil we measure with VISIR pupil imaging lens is 
                 # [probably] point symmetry reversed w.r.t reality. => OK checked
    for r in holectrs:
        ctrs.append([r[0], r[1]*REVERSE])

    return hdia, ctrs


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


    # GPI explorations
    ##for r in (0.99, 1.0, 1.01):
    ##  nrm = NRM_mask_definitions("gpi_g10s40", rescale = r)
    ##  maskobj = nrm.createnrmarray(puplscal=1.0e-2,fitsfile='g10s40Fr%.2f.fits' % r, fullpupil=True)
    ##pupobj = nrm.createpupilarray(puplscal=1.0e-2)
    ##print maskobj

    #nrm.createnrmarray(puplscal=1.0e-2,fitsfile='/Users/anand/Desktop/g10s40id.fits', holeid=True)
    #nrm.createpupilarray(puplscal=1.0e-2,fitsfile='/Users/anand/Desktop/gpipupil.fits')

    #pupil = NRM_mask_definitions("gpi_g10s40")
    #pupil.createpupilarray(puplscal)

    # make image at angular pixel scale, at given wavelength/bandpass
    # choose pupil pixel scale 
    # image oversampling before rebinning
    # possibly put in OPD
