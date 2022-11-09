#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import numpy.fft as fft
from astropy.io import fits
from astropy import units as u
import os, sys
import pickle
import scipy
from scipy.special import comb
import time
import nrm_analysis.misctools.matrixDFT as matrixDFT
import matplotlib.pyplot as plt
import synphot
import stsynphot
from stsynphot import grid_to_spec

m_ = 1.0
mm_ =  m_/1000.0
um_ = mm_/1000.0
nm_ = um_/1000.0



class Affine2d():
    """
    ========
    Affine2d
    ========

    A class to help implement the Bracewell Fourier 2D affine transformation
    theorem to calculate appropriate coordinate grids in the Fourier domain (eg
    image space, momentum space), given an affine transformation of the
    original (eg pupil space, configuration space) space.  This class provides
    the required normalization for the Fourier transform, and provides for
    a single way to set pixel pitch (including independent x and y scales)
    in the image plane.

    The theorem states that if f(x,y) and F(u,v) are Fourier pairs, and 
    g(x,y) = f(x',y'), where

        x' = mx * x  +  sx * y  +  xo
        y' = my * y  +  sy * x  +  yo,

    then G(u,v), the Fourier transform of g(x,y), is given by:

        G(u,v) = ( 1/|Delta| ) * exp { (2*Pi*i/Delta) *
                                          [ (my*xo - sx*yo) * u  +
                                            (mx*yo - sy*xo) * v  ] }  *
                                 F{ ( my*u - sy*v) / Delta, 
                                    (-sx*u + mx*v) / Delta  }
    where  
                Delta = mx * my - sx * sy.

    The reverse transformation, from (x',y') to (x,y) is given by:

        x = (1/Delta) * ( my * x' - sx * y'  -  my * xo + sx * yo )
        y = (1/Delta) * ( mx * y' - sy * x'  -  mx * yo + sy * xo )

    For clarity we call (x,y) IDEAL coordinates, and (x',y') DISTORTED coordinates.
    We know the analytical form of F(u,v) from the literature, and need to calculate
    G(u,v) at a grid of points in the (u,v) space, with two lattice vectors a and b
    defining the grid.  These lattice vectors have components a=(au,av) and b=(bu,bv) 
    along the u and v axes.

    Methods:
    --------

        forward()
        reverse()
        show()

    Discussion with Randall Telfer (2018.05.18)  clarified that:

        These constants, properly applied to the analytical transform in a
        "pitch matrix" instead of a scalar "pitch" variable, provide the PSF
        sampled in radians on an imaginary detector that is perpendicular to
        the chief ray.  The actual detector might be tilted in a different
        manner, changing the x pitch and y pitch of the detector pixels
        independent of the effects of pupil distortion.  
        
        We presume the main use of this object is to calculate intensity in the
        detector, so we include a DetectorTilt object in this class, although
        this object is constructed to have an 'identity' effect during the
        initial development and use of the Affine2d class in NRM data analysis.
        For most physical detector tilts we expect the DetectorTilt to have a
        small effect on an image simulated using the Fourier transform.  There
        are exceptions to this 'small effect' expectation (eg HST NICMOS 2 has
        a detector tilt of a few tens of degrees).  As long as the detector is
        small compared to the effective focal length (i.e. detector size <<<
        nominal f-ratio * primary diameter) of the system, detector tilts will
        change the pixel pitch (in radians) linearly across the field.

        There may be an ambiguity between the 'detector tilt effect' on pixel
        pitch and the diffractive effect (which results from pupil distortion
        between a pupil stop and the primary).  This might have to be broken
        using a pupil distortion from optical modelling such as ray tracing.
        Or it could be broken by requiring the detector tilt effect to be
        derived from optical models and known solid body models or metrology of
        the instrument/relescope, and the optical pupil distortion found from
        fitting on-sky data.

    Jean Baptiste Joseph Fourier 1768-1830
    Ron Bracewell 1921-2007
    Code by Anand Sivaramakrishnan 2018
    """

    def __init__(self, mx=None, my=None, 
                       sx=None, sy=None, 
                       xo=None, yo=None, 
                       rotradccw=None,
                       name="Affine"):
        """ Initialization wth transformation constants

        Parameters
        ----------
        mx : float x magnification # dimensionless
        my : float y magnification # dimensionless
        sx : float x shear with y # dimensionless
        sy : float y shear with x # dimensionless

        The following two parameters need understanding and documentation...
        xo : float offset in pupil space # dimension???  So far we only use zero.  Probably units of original (undistorted) x & y pixels.
        yo : float offset in pupil space # dimension???  So far we only use zero.  Probably units of original (undistorted) x & y pixels.

        rotradccw: None (no op, the default value), or...
                   a counter-clockwise rotation of *THE VECTOR FROM THE ORIGIN TO A POINT*,
                   in a FIXED COORDINATE FRAME, by this angle (radians)
                   (as viewed in ds9 or with fits NAXIS1 on X and NAXIS2 on Y).

        name: string, optional
        """

        self.rotradccw = rotradccw
        if rotradccw is not None:
            mx = np.cos(rotradccw)
            my = np.cos(rotradccw)
            sx = -np.sin(rotradccw)
            sy =  np.sin(rotradccw)
            xo = 0.0
            yo = 0.0

        self.mx = mx
        self.my = my
        self.sx = sx
        self.sy = sy
        self.xo = xo
        self.yo = yo
        self.determinant = mx * my - sx * sy
        self.absdeterminant = np.abs(self.determinant)
        self.name = name

        """
        numpy vector of length 2, (xprime,yprime) for use in manually writing
        the dot product needed for the exponent in the transform theorem.  Use
        this 2vec to dot with (x,y) in fromfunc to create the 'phase argument'
        Since this uses an offset xo yo in pixels of the affine transformation,
        these are *NOT* affected by the 'oversample' in image space.  The
        vector it is dotted with is in image space."""
        self.phase_2vector = np.array((my*xo - sx*yo, mx*yo - sy*xo)) / self.determinant

        if self.determinant == 0.0:
            print("Potentially fatal: Determinant of Affine2d transformation is zero")


    def forward(self, point):
        """ forward affine transformation, ideal-to-distorted coordinates

        Parameters
        ----------
        point : numpy vector of length 2, (x,y) in ideal space

        Returns
        ----------
        numpy vector of length 2, (xprime,yprime) in distorted space
        """

        return np.array((self.mx * point[0]  +  self.sx * point[1]  +  self.xo,
                         self.my * point[1]  +  self.sy * point[0]  +  self.yo))


    def reverse(self, point):
        """ reverse affine transformation, distorted-to-ideal coordinates

        Parameters
        ----------
        point : numpy vector of length 2, (x,y) in distorted space

        Returns
        ----------
        numpy vector of length 2, (xprime,yprime) in distorted space
        """

        return np.array(
          ( self.my * point[0] - self.sx * point[1]  -  self.my * xo + self.sx * yo ,
            self.mx * point[1] - self.sy * point[0]  -  self.mx * yo + self.sy * xo ) ) *\
                self.determinant
                      

    def distortFargs(self, u, v):
        """  Implement u,v to u',v' change in arguments of F (see theorem)

        Parameters
        ----------
        u,v : numpy array of same arbitrary shape, of arguments of (known) ideal transform
              typically generated in a fromfunction-invoked call

        Returns
        ----------
        numpy arrays uprime,vprime (like u,v) arguments of F when the config space
        is distorted by the affine2d transformation.
        """
        uprime = ( self.my*u - self.sy*v)/self.determinant
        vprime = (-self.sx*u + self.mx*v)/self.determinant
        return uprime, vprime

                      
    def distortphase(self, u, v):
        """  Calculate the phase term in the theorem

        Parameters
        ----------
        u,v : numpy array of same arbitrary shape, of arguments of (known) ideal transform
              typically generated in a fromfunction-invoked call

        Returns
        ----------
        numpy complex array like u or v, the phase term divided by the determinant.

         The phase term is:

         1/|Delta| * exp { (2*Pi*i/Delta) * [ (my*xo - sx*yo) * u  + (mx*yo - sy*xo) * v  ] }

         u and v have to be in inverse length units, viz. radians in image space / wavelength?
        """
        return np.exp( 2*np.pi*1j/self.determinant * \
                   (self.phase_2vector[0]*u + self.phase_2vector[1]*v) )  # / self.absdeterminant NO: error in first pass coding affine2d 

                      
    def get_rotd(self): 
        """Return the rotation that was used to creat a pure rot affine2d
           or None
        """
        if self.rotradccw:
            return 180.0*self.rotradccw/np.pi
        else:
            return None

    def show(self, label=None):
        """ print the transformation's parameters
        """

        if label: print("Affine transformation label: " + label)
        print("""Affine transformation "{7}" parameters are:
        mx, my  {0:+.4f}, {1:+.4f}
        sx, sy  {2:+.4f}, {3:+.4f}
        xo, yo  {4:+.4f}, {5:+.4f}
        Det {6:.4e}""".format( self.mx, self.my,
                               self.sx, self.sy,
                               self.xo, self.yo,
                               self.determinant, 
                               self.name))
        if self.rotradccw:
            print("    Created as pure {0:+.3f} degree CCW rotation".format(180.0*self.rotradccw/np.pi))


def avoidhexsingularity(rotation):
    """  Avoid rotation of exact multiples of 15 degrees to avoid NaN's in hextransformEE(). 

    Parameters
    ----------
    rotdegrees : rotation in degrees int or float

    Returns
    ----------
    replacement value for rotation with epsilon = 1.0e-12 degrees added.
    Precondition before using rotationdegrees in Affine2d for hex geometries

    """
    diagnostic = rotation/15.0 - int(rotation/15.0)
    epsilon = 1.0e-12
    if abs(diagnostic) < epsilon/2.0:
        rotation_adjusted = rotation + epsilon
    else:
        rotation_adjusted = rotation
    return rotation_adjusted

def affinepars2header(hdr, affine2d):
    """ writes affine2d parameters into fits header """
    hdr['affine'] = (affine2d.name, 'Affine2d in pupil: name')
    hdr['aff_mx'] = (affine2d.mx, 'Affine2d in pupil: xmag')
    hdr['aff_my'] = (affine2d.my, 'Affine2d in pupil: ymag')
    hdr['aff_sx'] = (affine2d.sx, 'Affine2d in pupil: xshear')
    hdr['aff_sy'] = (affine2d.sx, 'Affine2d in pupil: yshear')
    hdr['aff_xo'] = (affine2d.xo, 'Affine2d in pupil: x offset')
    hdr['aff_yo'] = (affine2d.yo, 'Affine2d in pupil: y offset')
    hdr['aff_dev'] = ('analyticnrm2', 'dev_phasor')
    return hdr



def rotate2dccw(vectors, thetarad):
    """ LG++ addition
    vectors is a list of 2d vectors - e.g. nrm hole  centers
    positive thetarad: x decreases under slight rotation
    positive thetarad: y increases under slight rotation
    positive thetarad = CCW rotation
    """
    c, s = (np.cos(thetarad), np.sin(thetarad))
    ctrs_rotated = []
    for vector in vectors:
        ctrs_rotated.append([c*vector[0] - s*vector[1], 
                             s*vector[0] + c*vector[1]])
    return np.array(ctrs_rotated)

    
def cp_var(nh, cps):
    """ True standard deviation given non-independent closure phases """
    return  ((cps - cps.mean())**2).sum() / scipy.special.comb(nh-1,2)

def ca_var(nh, cas):
    """ True standard deviation given non-independent closure amplitudes """
    return  ((cas - cas.mean())**2).sum() / scipy.special.comb(nh-3, 2)


def default_printoptions():
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8,
    suppress=False, threshold=1000, formatter=None)

def show_pistons_w(p_wav, prec=6, str=None):
    """
    p_wav: float 1-D array / m  input pistons
    """
    if str: print("\t"+str)
    np.set_printoptions(precision=prec,linewidth=160,formatter={'float': lambda x: "%10.4f," % x})
    print("\tp_wav stdev/w", round(p_wav.std(),prec))
    print("\tp_wav stdev/r", round(p_wav.std()*2*np.pi,prec))
    print("\tp_wav mean/r", round(p_wav.mean()*2*np.pi,prec))
    print("\tp_wav var/r^2", round((p_wav*2*np.pi).var(),prec))
    print("")
    default_printoptions()

def compare_pistons(pa, pb, prec=6, str=None):
    """
    pa: float 1-D array / radians  input pistons
    pb: float 1-D array / radians  output pistons
    prec: unused now... left in for older code
    call with e.g., 
        compare_pistons(ff_t.nrm.phi*2*np.pi, ff_t.nrm.fringepistons)
    Also prints Strehl contribution from piston OPD, Strehl change from error
    using Marechal approximation
    """
    if str: print(str)

    np.set_printoptions(precision=prec,linewidth=160,formatter={'float': lambda x: "%10.4f," % x})
    #
    p_err = pa - pb
    strehl = (pa).var()
    dstrehl = (p_err).var()
    #
    print("  input pistons/rad ", pa)
    print("  output pistons/rad", pb)
    print("          error/rad ", p_err)
    #
    print("      Strehl hit  {:.3e}%".format(strehl*100.0))
    print("   Strehl change  {:.3e}%".format( dstrehl*100.0))
    # 
    default_printoptions()

def centerpoint(s):
    """ 
        s is 2-d integer-valued (float or int) array shape
        return is a (2-tuple floats)
        correct for Jinc, hex transform, 'ff' fringes to place peak in
            central pixel (odd array) 
            pixel corner (even array)
    """
    return (0.5*s[0] - 0.5,  0.5*s[1] - 0.5)

def flip(holearray):
    fliparray= holearray.copy()
    fliparray[:,1] = -1*holearray[:,1]
    return fliparray

def mas2rad(mas):
    rad = 1.0e-3 * mas / (3600*180/np.pi)
    return rad

def rad2mas(rad):
    mas = rad * (3600*180/np.pi) * 10**3
    return mas


def replacenan(array):
    nanpos=np.where(np.isnan(array))
    array[nanpos]=np.pi/4
    return array


def makedisk(N, R, ctr=(0,0), array=None):
    
    if N%2 == 1:
        M = (N-1)/2
        xx = np.linspace(-M-ctr[0],M-ctr[0],N)
        yy = np.linspace(-M-ctr[1],M-ctr[1],N)
    if N%2 == 0:
        M = N/2
        xx = np.linspace(-M-ctr[0],M-ctr[0]-1,N)
        yy = np.linspace(-M-ctr[1],M-ctr[1]-1,N)
    (x,y) = np.meshgrid(xx, yy.T)
    r = np.sqrt((x**2)+(y**2))
    array = np.zeros((N,N))
    array[r<R] = 1
    return array


def makehex(N, s, ctr=(0,0)):
    """
    A. Greenbaum Sept 2015 - probably plenty of routines like this but I couldn't find
    one quickly.
    makes a hexagon with side length s in array size s at given rotation in degrees
    and centered at ctr coordinates. 
    0 rotation is flat side down
      ---   y
    /     \ ^ 
    \     / |
      ---   -> x
    """
    """
    needs replacing from the custom LG+ mask_definitions.py of Jan 2018 used to create JWST mask
    """
    array = np.zeros((N, N))
    if N%2 == 1:
        M = (N-1)/2
        xx = np.linspace(-M-ctr[0],M-ctr[0],N)
        yy = np.linspace(-M-ctr[1],M-ctr[1],N)
    if N%2 == 0:
        M = N/2
        xx = np.linspace(-M-ctr[0],M-ctr[0]-1,N)
        yy = np.linspace(-M-ctr[1],M-ctr[1]-1,N)
    (x,y) = np.meshgrid(xx, yy.T)
    h = np.zeros((N,N))
    d = np.sqrt(3)

    array[(y<(d*s/2.))*(y>(-d*s/2.))*\
      (y>(d*x)-(d*s))*\
      (y<(d*x)+(d*s))*\
      (y<-(d*x)+(d*s))*\
      (y>-(d*x)-(d*s))] = 1
    return array

def lambdasteps(lam, frac_width, steps=4):
    # frac_width =fractional bandwidth
    frac = frac_width/2.0
    steps = steps/ 2.0
    # add some very small number to the end to include the last number.
    lambdalist = np.arange( -1*frac*lam + lam, frac*lam + lam + 10e-10, frac*lam/steps)
    return lambdalist

def tophatfilter(lam_c, frac_width, npoints=10):
    wllist = lambdasteps(lam_c, frac_width, steps=npoints)
    filt = []
    for ii in range(len(wllist)):
        filt.append(np.array([1.0, wllist[ii]]))
    return filt

# from https://github.com/spacetelescope/jwst/blob/master/jwst/coron/median_replace_img.pydef 
def median_fill_value(input_array, input_dq_array, bsize, xc, yc):
    """
        Arguments:
        ----------
        input_array : ndarray
            Input array to filter.
        input_dq_array : ndarray
            Input data quality array
        bsize : scalar
            box size of the data to extract
        xc: scalar
           x position of the data extraction
        xc: scalar
           y position of the data extraction
        """
    # set the half box size
    hbox = int(bsize/2)

    # Extract the region of interest for the data
    try:
        data_array = input_array[xc - hbox:xc + hbox, yc - hbox: yc + hbox]
        dq_array = input_dq_array[xc - hbox:xc + hbox, yc - hbox: yc + hbox]
    except IndexError:
        # If the box is outside the data return 0
        print('utils.median_fill_value: Box for median filter is outside the data.')
        return 0.

    # good data only...
    filtered_array = data_array[dq_array == 0] # != dqflags.pixel['DO_NOT_USE']]
    median_value = np.median(filtered_array)

    if np.isnan(median_value): # Is this trouble for implaneia? We expect no NaNs... 
        # If the median fails return 0
        print('utils.median_fill_value: Median filter returned NaN setting value to 0.')
        median_value = 0.

    return median_value


def crosscorrelatePSFs(a, A, ov, verbose=False):
    """
    Assume that A is oversampled and padded by one pixel each side
    """
    print("\nInfo: performing cross correlations\n")
    print("\ta.shape is ", a.shape)
    print("\tA.shape is ", A.shape)

    p,q = a.shape
    padshape = (2*p, 2*q)  # pad the arrays to be correlated to avoid 
                           # 'aliasing bleed' from the edges
    cormat = np.zeros((2*ov, 2*ov))


    for x in range(2*ov):
        if verbose: print(x, ":", end=' ') 
        for y in range(2*ov):
            binA = krebin(A[x:x+p*ov, y:y+q*ov], a.shape)
            #print "\tbinA.shape is ", binA.shape  #DT 12/05/2014, binA.shape is  (5, 5)
            apad, binApad = (np.zeros(padshape), np.zeros(padshape))
            apad[:p,:q] = a         #shape (10,10) ,data is in the bottom left 5x5 corner 
            binApad[:p,:q] = binA   #shape (10,10) ,35x35 slice of perfect PSF 
                                      #binned down to 5x5 is in the bottom left corner
            cormat[x,y] = np.max(rcrosscorrelate(apad, binApad, verbose=verbose))
                       #shape of utils.rcrosscorrelate(apad, binApad, verbose=False is 10x10
            if verbose: print("%.3f" % cormat[x,y], end=' ')
        if verbose: print("\n")
    return cormat


#def centerit(img, r='default'):
#    print("deprecated - switch to using  'center_imagepeak'")
#    return center_imagepeak(img, r='default')
       
def center_imagepeak(img, r=None, cntrimg = True, dqm=False, verbose=False):

    """Cropped version of the input image centered on the peak pixel.

    Parameters
    ----------
    img : numpy input 2d array
    dqm : bad pixel location 2d numpy array of bools 
          eg STScI MAST DQ extension locations with DO_NOT_USE flag up.
    Disused: bpd : bad pixel array eg STScI MAST DQ extenion for JWST-NIRISS AmI 

    Returns
    -------
    cropped: numpy 2d array Cropped so brightest pixel is at the center
    Optional: dqmcrop, numpy 2d array,if dqmask array is passed.

    """
    peakx, peaky, h = min_distance_to_edge(img)
    if r is None:
        r = h.copy()
    else:
        pass

    cropped = img[int(peakx-r):int(peakx+r+1), int(peaky-r):int(peaky+r+1)]
    if type(dqm) is not bool: 
        dqmcrop = dqm[int(peakx-r):int(peakx+r+1), int(peaky-r):int(peaky+r+1)]

    if verbose:
        print('Cropped image shape:',cropped.shape)
        print('value at center:', cropped[r,r])
        print(np.where(cropped == cropped.max()))


    if type(dqm) is not bool: return cropped, dqmcrop
    else:                     return cropped
    


def min_distance_to_edge(img, cntrimg = False, verbose=False):
    """Return pixel distance to closest detector edge.

    Parameters
    ----------
    img : numpy input array

    Returns
    -------
    peakx, peaky: integer coordinates of the brightest pixel
    h: integer distance of the brightest pixel from the nearest edge of the input array
        
    """
    if verbose: print('Before cropping:', img.shape)

    if cntrimg==True:
        # Only look for the peak pixel at the center of the image
        ann = makedisk(img.shape[0], 31) # search radius around the center of array
    else:
        # Peak of the image can be anywhere
        ann = np.ones((img.shape[0], img.shape[1]))
        
    peakmask = np.where(img==np.nanmax(np.ma.masked_invalid(img[ann==1])))
    # following line takes care of peaks at two or more identical-value max pixel locations:
    peakx, peaky = peakmask[0][0], peakmask[1][0]
    if verbose: print('utils.min_distance_from_edge: peaking on: ',np.nanmax(np.ma.masked_invalid(img[ann==1])))
    if verbose: print('putils.min_distance_from_edge: peak x,y:', peakx, peaky)

    dhigh = (img.shape[0] - peakx - 1, img.shape[1] - peaky - 1)
    dlow = (peakx, peaky)
    h0 = min((dhigh[0],dlow[0]))
    h1 = min((dhigh[1],dlow[1]))
    h = min(h0, h1)
    return peakx, peaky, h # the 'half side' each way from the peak pixel

def find_centroid(a, thresh, verbose=False):
    """Return centroid of input image
 
    Parameters
    ----------
    a: input numpy array (real, square), considered 'image space'
    thresh: Threshold for the absolute value of the FT(a).
            Normalize abs(CV = FT(a)) for unity peak, and define the support 
            of "good" CV when this is above threshold, then find the phase 
            slope of the CV only over this support.
    
    Returns
    -------
    htilt, vtilt: Centroid of a, as offset from array center, in pythonese as calculated by the DFT's.

    Original domain a, Fourier domain CV
    sft square image a to CV array, no loss or oversampling - like an fft.
    Normalize peak of abs(CV) to unity
    Create 'live area' mask from abs(CV) with slight undersizing 
        (allow for 1 pixel shifts to live data still)
        (splodges, or full image a la KP)
    Calculate phase slopes using CV.angle() phase array
    Calculate mean of phase slopes over mask
    Normalize phase slopes to reflect image centroid location in pixels
    By eye, looking at smoothness of phase slope array...
    JWST NIRISS F480 F430M F380M 0.02
    JWST NIRISS F277 0.02 - untested
    GPI - untested

    XY conventions meshed to LG_Model conventions:
    if you simulate a psf with pixel_offset = ( (0.2, 0.4), ) then blind application

        centroid = utils.find_centroid()  
        
    returns the image centroid (0.40036, 0.2000093) pixels in image space. To use this
    in LG_Model, nrm_core,... you will want to calculate the new image center using

        image_center = utils.centerpoint(s) + np.array((centroid[1], centroid[0])

    and everything holds together sensibly looking at DS9 images of a.

    anand@stsci.edu 2018.02
    """

    ft = matrixDFT.MatrixFourierTransform()
    cv = ft.perform(a, a.shape[0], a.shape[0]) 
    cvmod, cvpha = np.abs(cv), np.angle(cv)
    cvmod = cvmod/cvmod.max() # normalize to unity peak
    cvmask = np.where(cvmod >= thresh)
    cvmask_edgetrim = trim(cvmask, a.shape[0])
    htilt, vtilt = findslope(cvpha, cvmask_edgetrim, verbose)

    M = np.zeros(a.shape)
    if verbose: print(">>> utils.find_centroid(): M.shape {0}, a.shape {1}".format(M.shape, a.shape))
    M[cvmask_edgetrim] = 1
    if verbose: print(">>> utils.find_centroid(): M.shape {0}, cvpha.shape {1}".format(M.shape, cvpha.shape))
    if 0:
        fits.PrimaryHDU(data=a).writeto("~/gitsrc/nrm_analysis/rundir/3_test_LGmethods_data/img.fits", overwrite=True)
        fits.PrimaryHDU(data=cvmod).writeto("~/gitsrc/nrm_analysis/rundir/3_test_LGmethods_data/cvmod.fits", overwrite=True)
        fits.PrimaryHDU(data=M*cvpha*180.0/np.pi).writeto("~/gitsrc/nrm_analysis/rundir/3_test_LGmethods_data/cvpha.fits", overwrite=True)
        if verbose: print("CV abs max = {}".format(cvmod.max()))
        if verbose: print("utils.find_centroid: {0} locations of CV array in CV mask for this data".format(len(cvmask[0])))

    return htilt, vtilt


def findslope(a, m, verbose=False):
    from astropy.stats import SigmaClip
    """ Find slopes of an array, over pixels not bordering the edge of the array
        You should have valid data either side of every pixel selected by the mask m.
        a is in radians of phase (in Fourier domain) when used in NRM/KP applications.

        The std dev of the middle 9 pixels are used to further clean the mask 'm'
        of invalid slope data, where we're subtracting inside-mask-support from
        outside-mask-support.  This mask is called newmask.

        Converting tilt in radians per Fourier Domain (eg pupil_ACF) pixel 
        Original Domain (eg image intensity) pixels:

            If the tilt[0] is 2 pi radians per ODpixel you recover the same OD
            array you started with.  That means you shifted the ODarray one
            full lattice spacing, the input array size, so you moved it by
            OD.shape[0].

                    2 pi / FDpixel of phase slope ==> ODarray.shape[0]
                1 radian / FDpixel of phase slope ==> ODarray.shape[0] / (2 pi) shift
                x radian / FDpixel of phase slope ==> x * ODarray.shape[0] / (2 pi) ODpixels of shift

        Gain between rad/pix phase slope and original domin pixels is  a.shape[0 or 1] / (2 pi)
        Multiply the measured phase slope by this gain for pixels of incoming array centroid shift away from array center.
        anand@stsci.edu 2018.02
        """
    a_up = np.zeros(a.shape)
    a_dn = np.zeros(a.shape)
    a_l = np.zeros(a.shape)
    a_r = np.zeros(a.shape)
    #
    a_up[:, 1:  ]  =  a[:,  :-1]
    a_dn[:,  :-1]  =  a[:, 1:  ]
    #
    a_r[1:  ,:]   =   a[ :-1,:]
    a_l[ :-1,:]   =   a[1:,   :]
    #
    offsetcube = np.zeros((4, a.shape[0], a.shape[1]))
    offsetcube[0,:,:] = a_up
    offsetcube[1,:,:] = a_dn
    offsetcube[2,:,:] = a_r
    offsetcube[3,:,:] = a_l
    if 0:
        fits.PrimaryHDU(data=offsetcube*180.0/np.pi).writeto(
             "~/gitsrc/nrm_analysis/rundir/3_test_LGmethods_data/offsetcube.fits",
             overwrite=True)

    tilt = np.zeros(a.shape), np.zeros(a.shape)
    tilt = (a_r - a_l)/2.0,  (a_up - a_dn)/2.0  # raw estimate of phase slope
    c = centerpoint(a.shape)
    C = (int(c[0]), int(c[1]))
    sigh, sigv = tilt[0][C[0]-1:C[0]+1,C[1]-1:C[1]+1].std(),   tilt[1][C[0]-1:C[0]+1,C[1]-1:C[1]+1].std()
    avgh, avgv = tilt[0][C[0]-1:C[0]+1,C[1]-1:C[1]+1].mean(),  tilt[1][C[0]-1:C[0]+1,C[1]-1:C[1]+1].mean()
    if verbose:
        print("C is {}".format(C))
        print("sigh, sinv = {0}.{1}".format(sigh,sigv), end='')
        print("avgh, avgv = {0}.{1}".format(avgh,avgv))

    # second stage mask cleaning: 5 sig rejection of mask
    newmaskh = np.where( np.abs(tilt[0] - avgh) < 5*sigh )
    newmaskv = np.where( np.abs(tilt[1] - avgv) < 5*sigv )
    if verbose:
        print("tilth over newmaskh mean {0:.4e} sig {1:.4e}".format( tilt[0][newmaskh].mean(), tilt[0][newmaskh].std() ))
        print("tiltv over newmaskh mean {0:.4e} sig {1:.4e}".format( tilt[1][newmaskv].mean(), tilt[1][newmaskv].std() ))
    th, tv = np.zeros(a.shape), np.zeros(a.shape)
    th[newmaskh] = tilt[0][newmaskh]
    tv[newmaskv] = tilt[1][newmaskv]
    # figure out units of tilt - 
    if 0:
        fits.PrimaryHDU(data=th*180.0/np.pi).writeto(
             "~/gitsrc/nrm_analysis/rundir/3_test_LGmethods_data/tilt0.fits",
             overwrite=True)
        fits.PrimaryHDU(data=tv*180.0/np.pi).writeto(
             "~/gitsrc/nrm_analysis/rundir/3_test_LGmethods_data/tilt1.fits",
             overwrite=True)
    G = a.shape[0] / (2.0*np.pi),  a.shape[1] / (2.0*np.pi)
    return G[0]*tilt[0][newmaskh].mean(), G[1]*tilt[1][newmaskv].mean() 


def trim(m, s):
    """ Removes edge pixels from an index mask m. For example, m created from np.where(a<X)

    Parameters
    ----------
    m: 2d index mask
    s: The side of the parent array that was used to generate m. 

    Returns
    -------
    2d index mask

    anand@stsci.edu
    """
    xl, yl = [], []  # trimmed lists 
    for ii in range(len(m[0])): # go through all indices in the mask x y lists
        # test for any index being an edge index - if none are on the edge, remember the indices in new list
        if (m[0][ii] == 0 or m[1][ii] == 0 or m[0][ii] == s-1 or m[1][ii] == s-1) == False:
            xl.append(m[0][ii])
            yl.append(m[1][ii])
    return (np.asarray(xl), np.asarray(yl))


def deNaN(s, datain):
    ## Get rid of NaN values with nearest neighbor median
    fov=datain.shape[0]
    a2 = np.zeros((2*fov, 2*fov))
    print("deNaN: fov:", fov, "selection shape:", (fov//2,3*fov//2,fov//2,3*fov/2))
    a2[fov//2:fov+fov//2, fov//2 : fov+fov//2 ] = datain
    xnan, ynan = np.where(np.isnan(a2))
    for qq in range(len(a2[np.where(np.isnan(a2))])):
        a2[xnan[qq], ynan[qq]] = neighbor_median((xnan[qq],ynan[qq]), s, a2)
    return a2[fov//2:fov +fov//2, fov//2: fov+fov//2]


def neighbor_median(ctr, s, a2):
    # take the median of nearest neighbors within box side s
    atmp = a2[ctr[0]-s:ctr[0]+s+1, ctr[1]-s:ctr[1]+s+1]
    med = np.median(atmp[np.isnan(atmp)==False])


# def get_fits_filter(fitsheader, verbose=False):
#     wavestring = "WAVE"
#     weightstring = "WGHT"
#     filterlist = []
#     if verbose: print(fitsheader[:])
#     j =0
#     for j in range(len(fitsheader)):
#         if wavestring+str(j) in fitsheader:
#             wght = fitsheader[weightstring+str(j)]
#             wavl = fitsheader[wavestring+str(j)]
#             if verbose: print("wave", wavl)
#             filterlist.append(np.array([wght,wavl]))
#     if verbose: print(filterlist)
#         #print "specbin - spec.shape", spec.shape
#     return filterlist
#


def makeA(nh, verbose=False):
    """ 
    Writes the "NRM matrix" that gets pseudo-inverterd to provide
    (arbitrarily constrained) zero-mean phases of the holes.

    makeA taken verbatim from Anand's pseudoinverse.py

     input: nh - number of holes in NR mask
     input: verbose - True or False
     output: A matrix, nh columns, nh(nh-1)/2 rows  (eg 21 for nh=7)

    Ax = b  where x are the nh hole phases, b the nh(nh-1)/2 fringe phases,
    and A the NRM matrix

    Solve for the hole phases:
        Apinv = np.linalg.pinv(A)
        Solution for unknown x's:
        x = np.dot(Apinv, b)

    Following Noah Gamper's convention of fringe phases,
    for holes 'a b c d e f g', rows of A are 

        (-1 +1  0  0  ...)
        ( 0 -1 +1  0  ...)
    

    which is implemented in makeA() as:
        matrixA[row,h2] = -1
        matrixA[row,h1] = +1

    To change the convention just reverse the signs of the 'ones'.

    When tested against Alex'' NRM_Model.py "piston_phase" text output of fringe phases, 
    these signs appear to be correct - anand@stsci.edu 12 Nov 2014

    anand@stsci.edu  29 Aug 2014
        """

    if verbose: print("\nmakeA(): ")
    #                   rows         cols
    ncols = (nh*(nh-1))//2
    nrows = nh
    matrixA = np.zeros((ncols, nrows))
    if verbose: print(matrixA)
    row = 0
    for h2 in range(nh):
        if verbose: print()
        for h1 in range(h2+1,nh):
            if h1 >= nh:
                break
            else:
                if verbose:
                    print("R%2d: "%row, end=' ') 
                    print("%d-%d"%(h1,h2))
                matrixA[row,h2] = -1
                matrixA[row,h1] = +1
                row += 1
    return matrixA


def fringes2pistons(fringephases, nholes):
    """
    For NRM_Model.py to use to extract pistons out of fringes, given its hole bookkeeping,
    which apparently matches that of this module, and is the same as Noah Gamper's
    anand@stsci.edu  12 Nov 2014
    input: 1D array of fringe phases, and number of holes
    returns: pistons in same units as fringe phases
    """
    Anrm = makeA(nholes)
    Apinv = np.linalg.pinv(Anrm)
    return np.dot(Apinv, fringephases)

def makeK(nh, verbose=False):
    """ 
    As above, write the "kernel matrix" that converts fringe phases
    to closure phases. This can be psuedo-inverted to provide a 
    subset of "calibrated" fringe phases (hole-based noise removed)

     input: nh - number of holes in NR mask
     input: verbose - True or False
     output: L matrix, nh(nh-1)/2 columns, comb(nh, 3) rows  (eg 35 for nh=7)

    Kx = b, where: 
        - x are the nh(nh-1)/2 calibrated fringe phases 
        - b the comb(nh, 3) closure phases,
    and K the kernel matrix

    Solve for the "calibrated" phases:
        Kpinv = np.linalg.pinv(K)
        Solution for unknown x's:
        x = np.dot(Kpinv, b)

    Following the convention of fringe phase ordering above, which should look like:
    h12, h13, h14, ..., h23, h24, ....
    rows of K should look like:

        (+1 -1  0  0  0  0  0  0 +1 ...) e.g., h12 - h13 + h23
        (+1 +1  0 +1  ...)
    

    which is implemented in makeK() as:
        matrixK[n_cp, f12] = +1
        matrixK[n_cp, f13] = -1
        matrixK[n_cp, f23] = +1

    need to define the row selectors
     k is a list that looks like [9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
     -----up to nh*(nh-1)/2
     i is a list that looks like [0,9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
     -----up to nh*(nh-1)/2 -1
    because there are 9 fringe phases per single hole (decreasing by one to avoid repeating)
    hope that helps explain this!

    agreenba@pha.jhu.edu  22 Aug 2015
        """

    if verbose: print("\nmakeK(): ")
    nrow = comb(nh, 3)
    ncol = nh*(nh-1)/2

    # first define the row selectors
    # k is a list that looks like [9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
    # -----up to nh*(nh-1)/2
    # i is a list that looks like [0,9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
    # -----up to nh*(nh-1)/2 -1
    countk=[]
    val=0
    for q in range(nh-1):
        val = val + (nh-1)-q
        countk.append(val)
    counti = [0,]+countk[:-1]
    # MatrixK
    row=0
    matrixK = np.zeros((nrow, ncol))
    for ii in range(nh-2):
        for jj in range(nh-ii-2):
            for kk in range(nh-ii-jj-2):
                matrixK[row+kk, counti[ii]+jj] = 1
                matrixK[row+kk, countk[ii+jj]+kk] = 1
                matrixK[row+kk, counti[ii]+jj+kk+1] = -1
            row=row+kk+1
    if verbose: print()

    return matrixK


def create_ifneed(dir_):
    """ http://stackoverflow.com/questions/273192/check-if-a-directory-exists-and-create-it-if-necessary
    kanja
    """
    print("utils.create_ifneed: delete me???")
    if not os.path.exists(dir_):
        os.mkdir(dir_)


def nb_pistons(multiplier=1.0, debug=False):
    """ copy nb values from NRM_Model.py and play with them using this function, as a sanity check to compare to CV2 data
        Define phi at the center of F430M band: lam = 4.3*um
        Scale linearly by "multiplier" factor if requested
        Returns *zero-meaned* pistons in meters for caller to interpret (typically as wavefront, i.e. OPD)
    """
    if debug:
        phi_nb_ = np.array( [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] ) * 50.0 * nm_  # -150 nm to 150 nm OPDs in 50 nm steps
        return (phi_nb_ - phi_nb_.mean())
    else:
        phi_nb_ = multiplier * np.array( [0.028838669455909766, -0.061516214504502634, 0.12390958557781348, \
                                         -0.020389361461019516, 0.016557347248600723, -0.03960017912525625, \
                                          -0.04779984719154552] ) # phi in waves
        phi_nb_ = phi_nb_ - phi_nb_.mean() # phi in waves, zero mean
        wl = 4.3 * um_
        if debug: print("std dev  of piston OPD: %.1e um" % (phi_nb_.std() * wl/um_))
        if debug: print("variance of piston OPD: %.1e rad" % (phi_nb_.var() * (4.0*np.pi*np.pi)))
        return phi_nb_ * 4.3*um_ # phi_nb in m

# -----------------------------------------------------------------
# Following functions added 2/22 by RAC to replace older filter/src
# combination routines: get_webbpsf_filter(), trim_webbpsf_filter(),
# combine_transmission(). Now use get_filt_spec(), get_src_spec(),
# and combine_src_filt(). These use synphot and stsynphot directly to
# avoid Poppy dependency (though much code is copied from Poppy funcs).
# -----------------------------------------------------------------

def get_filt_spec(filt, verbose=False):
    """
    Load WebbPSF filter throughput into synphot spectrum object
    filt: string, known NIRISS AMI filter name
    Returns:
        synphot Spectrum object
    """
    goodfilts = ["F277W", "F380M", "F430M", "F480M"]
    # make uppercase
    filt = filt.upper()
    if filt not in goodfilts:
        raise Exception("Filter name %s is not a known NIRISS AMI filter. Choose one of: %s" % (filt, tuple(goodfilts)))

    webbpsf_path = os.getenv('WEBBPSF_PATH')
    filterdir = os.path.join(webbpsf_path, 'NIRISS/filters/')
    filtfile = os.path.join(filterdir, filt + '_throughput.fits')
    if verbose: print("\n Using filter file:", filtfile)
    thruput = fits.getdata(filtfile)
    wl_list = np.asarray([tup[0] for tup in thruput])  # angstroms
    tr_list = np.asarray([tup[1] for tup in thruput])
    band = synphot.spectrum.SpectralElement(synphot.models.Empirical1D, points=wl_list,
                                            lookup_table=tr_list, keep_neg=False)
    return band

def get_src_spec(sptype):
    """
    Modified from poppy's spectrum_from_spectral_type
    src: either valid source string (e.g. "A0V") or existing synphot Spectrum object
    Defaults to A0V spectral type if input string not recognized.
    Returns:
        synphot Spectrum object
    """
    # check if it's already a synphot spectrum
    if isinstance(sptype, synphot.spectrum.SourceSpectrum):
        print('Input is a synphot spectrum')
        return sptype
    else:
        # phoenix model lookup table used in JWST ETCs
        lookuptable = {
            "O3V": (45000, 0.0, 4.0),
            "O5V": (41000, 0.0, 4.5),
            "O7V": (37000, 0.0, 4.0),
            "O9V": (33000, 0.0, 4.0),
            "B0V": (30000, 0.0, 4.0),
            "B1V": (25000, 0.0, 4.0),
            "B3V": (19000, 0.0, 4.0),
            "B5V": (15000, 0.0, 4.0),
            "B8V": (12000, 0.0, 4.0),
            "A0V": (9500, 0.0, 4.0),
            "A1V": (9250, 0.0, 4.0),
            "A3V": (8250, 0.0, 4.0),
            "A5V": (8250, 0.0, 4.0),
            "F0V": (7250, 0.0, 4.0),
            "F2V": (7000, 0.0, 4.0),
            "F5V": (6500, 0.0, 4.0),
            "F8V": (6250, 0.0, 4.5),
            "G0V": (6000, 0.0, 4.5),
            "G2V": (5750, 0.0, 4.5),
            "G5V": (5750, 0.0, 4.5),
            "G8V": (5500, 0.0, 4.5),
            "K0V": (5250, 0.0, 4.5),
            "K2V": (4750, 0.0, 4.5),
            "K5V": (4250, 0.0, 4.5),
            "K7V": (4000, 0.0, 4.5),
            "M0V": (3750, 0.0, 4.5),
            "M2V": (3500, 0.0, 4.5),
            "M5V": (3500, 0.0, 5.0),
            "B0III": (29000, 0.0, 3.5),
            "B5III": (15000, 0.0, 3.5),
            "G0III": (5750, 0.0, 3.0),
            "G5III": (5250, 0.0, 2.5),
            "K0III": (4750, 0.0, 2.0),
            "K5III": (4000, 0.0, 1.5),
            "M0III": (3750, 0.0, 1.5),
            "O6I": (39000, 0.0, 4.5),
            "O8I": (34000, 0.0, 4.0),
            "B0I": (26000, 0.0, 3.0),
            "B5I": (14000, 0.0, 2.5),
            "A0I": (9750, 0.0, 2.0),
            "A5I": (8500, 0.0, 2.0),
            "F0I": (7750, 0.0, 2.0),
            "F5I": (7000, 0.0, 1.5),
            "G0I": (5500, 0.0, 1.5),
            "G5I": (4750, 0.0, 1.0),
            "K0I": (4500, 0.0, 1.0),
            "K5I": (3750, 0.0, 0.5),
            "M0I": (3750, 0.0, 0.0),
            "M2I": (3500, 0.0, 0.0)}
        try:
            keys = lookuptable[sptype]
        except KeyError:
            print(
                "\n WARNING!!! \n Input spectral type %s did not match any in the catalog. Defaulting to A0V." % sptype)
            keys = lookuptable["A0V"]
        try:
            return grid_to_spec('phoenix', keys[0], keys[1], keys[2])
        except IOError:
            errmsg = ("Could not find a match in catalog {0} for key {1}. Check that is a valid name in the " +
                      "lookup table, and/or that synphot is installed properly.".format(catname, sptype))
            raise LookupError(errmsg)

def combine_src_filt(bandpass, srcspec, trim=0.01, nlambda=19, verbose=False, plot=False):
    """
    Get the observed spectrum through a filter.
    Largely copied from Poppy instrument.py
    Define nlambda bins of wavelengths, calculate effstim for each, normalize by effstim total.
    nlambda should be calculated so there are ~10 wavelengths per resolution element (19 should work)
    Inputs:
        bandpass: synphot Spectrum (from get_filt_spec)
        srcspec: synphot Spectrum (from get_src_spec)
        trim: if not None, trim bandpass to where throughput greater than trim
        nlambda: number of wavelengths across filter to return
    Returns:
        finalsrc: numpy array of shape (nlambda,2) containing wavelengths, final throughputs

    """

    wl_filt, th_filt = bandpass._get_arrays(bandpass.waveset)
    # print(len(wl_filt),len(th_filt))

    if trim:
        if verbose: print("Trimming bandpass to above %.1e throughput" % trim)
        goodthru = np.where(np.asarray(th_filt) > trim)
        low_idx, high_idx = goodthru[0][0], goodthru[0][-1]
        wl_filt, th_filt = wl_filt[low_idx:high_idx], th_filt[low_idx:high_idx]
        # print(len(wl_filt),len(th_filt))
    # get effstim for bins of wavelengths
    # plt.plot(wl_filt,th_filt)
    minwave, maxwave = wl_filt.min(), wl_filt.max()  # trimmed or not
    wave_bin_edges = np.linspace(minwave, maxwave, nlambda + 1)
    wavesteps = (wave_bin_edges[:-1] + wave_bin_edges[1:]) / 2
    deltawave = wave_bin_edges[1] - wave_bin_edges[0]
    area = 1 * (u.m * u.m)
    effstims = []
    ptsin = len(wl_filt)
    binfac = ptsin // nlambda
    if verbose: print("Binning spectrum by %i: from %i points to %i points" % (binfac, ptsin, nlambda))
    for wave in wavesteps:
        if verbose:
            print(f"\t Integrating across band centered at {wave.to(u.micron):.2f} "
                  f"with width {deltawave.to(u.micron):.2f}")
        box = synphot.spectrum.SpectralElement(synphot.models.Box1D, amplitude=1, x_0=wave,
                                               width=deltawave) * bandpass

        binset = np.linspace(wave - deltawave, wave + deltawave,
                             30)  
        binset = binset[binset >= 0]  # remove any negative values
        result = synphot.observation.Observation(srcspec, box, binset=binset).effstim('count', area=area)
        effstims.append(result)

    effstims = u.Quantity(effstims)
    effstims /= effstims.sum()  # Normalized count rate is unitless
    wave_m = wavesteps.to_value(u.m)  # convert to meters
    effstims = effstims.to_value()  # strip units

    if plot:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(wave_m, effstims)
        ax.set_xlabel(r"$\lambda$ [m]")
        ax.set_ylabel("Throughput")
        ax.set_title("Combined Binned Spectrum (filter and source)")
        plt.show()

    finalsrc = np.array((effstims,wave_m)).T # this is the order expected by InstrumentData

    return finalsrc


def test_spec_comb():
    """
    Simple test of filter and source spectra combination with verbosity and
    plotting for all four NIRISS AMI filters
    """
    filters = ["F277W","F380M","F430M","F480M"]
    srcspec = get_src_spec("A0V")
    srcspec.plot()

    for filt in filters:
        band = get_filt_spec(filt)
        band.plot()
        final = combine_src_filt(band, srcspec, verbose=True, plot=True)

def get_cw_beta(bandpass):
    """ 
    Bandpass: array where the columns are weights, wavelengths
    Return weighted mean wavelength in meters, fractional bandpass
    """
    from scipy.integrate import simps
    wt = bandpass[:,0]
    wl = bandpass[:,1]
    cw = (wl*wt).sum()/wt.sum() # Weighted mean wavelength in meters "central wavelength"
    area = simps(wt, wl)
    ew = area / wt.max() # equivalent width
    beta = ew/cw # fractional bandpass
    return cw, beta

# -----------------------------------------------------------------

    #########################################################################################
    ##################################### from ##############################################
    ################################## gpipoppy2.py #########################################
    #########################################################################################
    #########################################################################################


def krebin(a, shape): # Klaus P's fastrebin from web - return array of 'shape'
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)


def  rebin(a = None, rc=(2,2), verbose=None):  # Thinly-wrapped krebin 
    """  
    anand@stsci.edu
    Perform simple-minded flux-conserving binning... clip trailing
    size mismatch: eg a 10x3 array binned by 3 results in a 3x1 array
    """
    return krebin(a, (a.shape[0]//rc[0],a.shape[1]//rc[1]))


# used in NRM_Model.py
def rcrosscorrelate(a=None, b=None, verbose=False):

    """ Calculate cross correlation of two identically-shaped real arrays,
        returning a new array  that is the correlation of the two input 
        arrays.
    """

    c = crosscorrelate(a=a, b=b, verbose=verbose) / (np.sqrt((a*a).sum())*np.sqrt((b*b).sum()))
    return  c.real.copy()


def crosscorrelate(a=None, b=None, verbose=False):

    """ Calculate cross correlation of two identically-shaped real or complex arrays,
        returning a new complex array  that is the correl of the two input 
        arrays. 
        ACF(f) = FT(powerspectrum(f))
    """

    if a.shape != b.shape:
        print("crosscorrelate: need identical arrays")
        return None

    fac = np.sqrt(a.shape[0] * a.shape[1])
    if verbose: print("fft factor = %.5e" % fac)

    A = fft.fft2(a)/fac
    B = fft.fft2(b)/fac
    if verbose: print("\tPower2d -  a: %.5e  A: %.5e " % ( (a*a.conj()).real.sum() ,  (A*A.conj()).real.sum()))

    c = fft.ifft2(A * B.conj())  * fac * fac

    if verbose: print("\tPower2d -  A: %.5e  B: %.5e " % ( (A*A.conj()).real.sum() ,  (B*B.conj()).real.sum()))
    if verbose: print("\tPower2d -  c=A_correl_B: %.5e " % (c*c.conj()).real.sum())

    if verbose: print("\t(a.sum %.5e x b.sum %.5e) = %.5e   c.sum = %.5e  ratio %.5e  " % (a.sum(), b.sum(), a.sum()*b.sum(), c.sum().real,  a.sum()*b.sum()/c.sum().real))

    if verbose: print() 
    return fft.fftshift(c)

# used in NRM_Model.py
def quadratic_extremum(p):  # used to take an x vector and return y,x for smoot plotting
    "  max y = -b^2/4a + c occurs at x = -b/2a, returns xmax, ymax"
    #print("Max y value %.5f"%(-p[1]*p[1] /(4.0*p[0]) + p[2]))
    #print("occurs at x = %.5f"%(-p[1]/(2.0*p[0])))
    return -p[1]/(2.0*p[0]), -p[1]*p[1] /(4.0*p[0]) + p[2]


# used in NRM_Model.py
def findpeak_1d(yvec, xvec):
    p = np.polyfit(np.array(xvec), np.array(yvec), 2)
    return quadratic_extremum(p)


def findmax(mag, vals, mid=1.0):
    p = np.polyfit(mag, vals, 2)
    fitr = np.arange(0.95*mid, 1.05*mid, .01) 
    maxx, maxy, fitc = quadratic(p, fitr)
    return maxx, maxy


def findmax_detail(mag, vals, start=0.9,stop=1.1):
    ''' mag denotes x values (like magnifications tested) vals is for y values.'''
    ## e.g. polyfit returns highest degree first
    ## e.g. p[0].x^2 +p[1].x + p[2]
    p = np.polyfit(mag, vals, 2)
    fitr = np.arange(start, stop, 0.001) 
    maxx, maxy, fitc = quadratic(p, fitr)
    error = quadratic(p,np.array(mag))[2] - vals
    return maxx, maxy,fitr, fitc, error

def jdefault(o):
    return o.__dict__

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
    with open(r"{0}{1}.json".format(savdir, outputname), "wb") as output_file:
        json.dump(savobj, output_file, default=jdefault)
    print("success!")

    ffotest = json.load(open(outputname+".json", 'r'))
    
    print(hasattr(ffotest, 'test'))
    print(ffotest['test'])


    # init stuff
    savobj.pscale_rad, savobj.pscale_mas = nrmobj.pixel, rad2mas(nrmobj.pixel)
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
        json.dump(savobj, output_file, default=jdefault)

    return savdir+outputname+".ffo"

############### More useful functions ##################
def printout(ctrs, infostr=""):  # ctrs.shape=(Nholes,2)
    print("    ctrs " + infostr)
    for nn in range(ctrs.shape[0]):
        print("    {0:+.6f}    {1:+.6f}".format(ctrs[nn,0], ctrs[nn,1]))


def baselinify(ctrs):
    N = len(ctrs)
    uvs = np.zeros((N*(N-1)//2, 2))
    label = np.zeros((N*(N-1)//2, 2))
    bllengths = np.zeros(N*(N-1)//2)
    nn=0
    for ii in range(N-1):
        for jj in range(N-ii-1):
            uvs[jj+nn, 0] = ctrs[ii,0] - ctrs[ii+jj+1,0]
            uvs[jj+nn, 1] = ctrs[ii,1] - ctrs[ii+jj+1,1]
            bllengths[jj+nn] = np.sqrt((ctrs[ii,0]-ctrs[ii+jj+1,0])**2 +\
                        (ctrs[ii,1]-ctrs[ii+jj+1,1])**2)
            label[jj+nn,:] = np.array([ii, ii+jj+1])
        nn = nn+jj+1
    return uvs, bllengths, label

def count_cps(ctrs):
    from scipy.misc import comb
    N = len(ctrs)
    ncps = int(comb(N,3))
    cp_label = np.zeros((ncps, 3))
    u1 = np.zeros(ncps)
    v1 = np.zeros(ncps)
    u2 = np.zeros(ncps)
    v2 = np.zeros(ncps)
    nn=0
    for ii in range(N-2):
        for jj in range(N-ii-2):
            for kk in range(N-ii-jj-2):
                #print '---------------'
                #print nn+kk
                cp_label[nn+kk,:] = np.array([ii, ii+jj+1, ii+jj+kk+2])
                u1[nn+kk] = ctrs[ii,0] - ctrs[ii+jj+1, 0]
                v1[nn+kk] = ctrs[ii,1] - ctrs[ii+jj+1, 1]
                u2[nn+kk] = ctrs[ii+jj+1,0] - ctrs[ii+jj+kk+2, 0]
                #print ii
                #print ii+jj+1, ii+jj+kk+2
                v2[nn+kk] = ctrs[ii+jj+1,1] - ctrs[ii+jj+kk+2, 1]
                #print u1[nn+kk], u2[nn+kk], v1[nn+kk], v2[nn+kk]
            nn = nn+kk+1
    return u1, v1, u2, v2, cp_label

def baseline_info(mask, pscale_mas, lam_c):
    mask.ctrs = np.array(mask.ctrs)
    uvs, bllengths, label = baselinify(mask.ctrs)

    print("========================================")
    print("All the baseline sizes:")
    print(bllengths)
    print("========================================")
    print("Longest baseline:")
    print(bllengths.max(), "m")
    print("corresponding to lam/D =", rad2mas(lam/bllengths.max()), "mas at {0} m \n".format(lam_c))
    print("Shortest baseline:")
    print(bllengths.min(), "m")
    print("corresponding to lam/D =", rad2mas(lam/bllengths.min()), "mas at {0} m".format(lam_c))
    print("========================================")
    print("Mask is Nyquist at", end=' ')
    print(mas2rad(2*pscale_mas)*(bllengths.max()))

def corner_plot(pickfile, nbins = 100, save="my_calibrated/triangle_plot.png"):
    """
    Make a corner plot after the fact using the pickled results from the mcmc
    """
    import corner
    import matplotlib.pyplot as plt

    mcmc_results = pickle.load(open(pickfile))
    keys = list(mcmc_results.keys())
    print(keys)
    chain = np.zeros((len(mcmc_results[keys[0]]), len(keys)))
    print(len(keys))
    print(chain.shape)
    for ii,key in enumerate(keys):
        chain[:,ii] = mcmc_results[key]

    fig = corner.corner(chain, labels = keys, bins=nbins, show_titles=True)
    plt.savefig("triangle_plot.pdf")
    plt.show()

def populate_symmamparray(amps, N=7):
    fringeamparray = np.zeros((N,N))
    step=0
    n=N-1
    for h in range(n):
        #print "step", step, "step+n", step+n
        #print "h", h, "h+1", h+1, "and on"
        #print fringeamparray[h,h+1:].shape, amps[step:step+n].shape
        fringeamparray[h,h+1:] = amps[step:step+n]
        step = step+n
        n=n-1
    fringeamparray = fringeamparray + fringeamparray.T
    return fringeamparray

def t3vis(vis, N=7):
    """ provided visibilities, this put these into triple product"""
    amparray = populate_symmamparray(vis, N=N)
    t3vis = np.zeros(int(comb(N,3)))
    nn=0
    for kk in range(N-2):
        for ii in range(N-kk-2):
            for jj in range(N-kk-ii-2):
                t3vis[nn+jj] = amparray[kk,ii+kk+1] \
                * amparray[ii+kk+1,jj+ii+kk+2] \
                * amparray[jj+ii+kk+2,kk]
            nn=nn+jj+1
    return t3vis

def t3err(viserr, N=7):
    """ provided visibilities, this put these into triple product"""
    amparray = populate_symmamparray(viserr, N=N)
    t3viserr = np.zeros(int(comb(N,3)))
    nn=0
    for kk in range(N-2):
        for ii in range(N-kk-2):
            for jj in range(N-kk-ii-2):
                t3viserr[nn+jj] = np.sqrt(amparray[kk,ii+kk+1]**2 \
                + amparray[ii+kk+1,jj+ii+kk+2]**2 \
                + amparray[jj+ii+kk+2,kk]**2 )
            nn=nn+jj+1
    return t3viserr


"""
    Convert hard-coded [ami]sim files to mirage-y (MAST) format

    Read in [ami]sim cube of data, 2D or 3D cube and graft mirage-like headers on the data part, 
    Also cube the data if incoming data is 2D.
    Input files are in the list variable simfile, so edit this as needed.
    MIRAGE template file is in the string variable mirexample
"""

def amisim2mirage(datadir, amisimfns, mirexample, filt, verbose=False, trim2sub80=True):
    """
    datadir: where input amisim files (cubes or 2d) are located
    amisimfns: one or a list/tuple of simulated data files
    mirexample: fullpath/filename of mirage example to use for header values
    Returns: one or list of mirage-like file names (not full path).
    *** WARNING *** sim2mirage.py:
        The MIRAGE fits file that provides the structure to wrap your simulated
        data to look like MAST data is for the F480M filter.  If you use it to
        convert any  other other filter's simulated data into mirage format, change
        this with e.g., 'mirobj[0].header["FILTER"] = "F430M"' or other filter used
        to create the simulated data.
    """
    mirext = "_mir"
    mirfns = [] # list of mirage-like file names
    if type(amisimfns) == str:
            amisimfns = [amisimfns, ]
    for fname in amisimfns:
        if verbose: print(fname+':', end='')
        fobj_sim = fits.open(datadir+fname+".fits")
        if len(fobj_sim[0].data.shape) == 2:    #make single slice of data into a 1 slice cube
            d = np.zeros((1, fobj_sim[0].data.shape[0], fobj_sim[0].data.shape[1]))
            d[0,:,:] = fobj_sim[0].data
            fobj_sim[0].data = d

        try:
            mirobj = fits.open(mirexample) # read in sample mirage file
        except FileNotFoundError:
            sys.exit("*** ERROR sim2mirage.py:  mirage example {:s} file not found ***".format(mirexample))
        
        # make cube of data for mirage from input ami_sim file... even a 1-deep cube.
        mirobj[1].data = np.zeros((fobj_sim[0].data.shape[0], #"slices of cube of integrations"
                                     mirobj[1].data.shape[0], 
                                     mirobj[1].data.shape[1]))
        
        mirobj[1].data = mirobj[1].data.astype(np.float32)
        mirobj[1].data = fobj_sim[0].data # replace with ami_sim data
        if trim2sub80:  # trim ends of cols, rows (if needed) to 80 x 80
            lenx = mirobj[1].data.shape[1] - 80
            leny = mirobj[1].data.shape[2] - 80 
            lox = lenx//2
            loy = leny//2
            mirobj[1].data = mirobj[1].data[:,lox:lox+80,loy:loy+80]  # trim ends of cols, rows (if needed) to 80 x 80
            print("INFO: utils.amisim2mirage will trim input image(s) to SUB80")
            print("      approx centering middle pixel (even/odd trouble?)")
        else:
            print("INFO: utils.amisim2mirage will not trim input image(s) to SUB80")
        # make cube of bad pixel data to match size of science data. Really only necessary
        # if ami_sim file has more ints than the example MIRAGE file, due to how DQ array
        # is sliced to match science array in InstrumentData
        # RAC 9/21
        # already trimmed to N x 80 x 80 if required
        mirobj['DQ'].data = np.zeros(mirobj[1].data.shape, dtype=np.uint32)

        # Transfer non-conflicting keywords from sim data to mirage file header
        mirkeys = list(mirobj[0].header.keys())

        for kwd in fobj_sim[0].header.keys():
            if kwd not in mirkeys and 'NAXIS' not in kwd:
                try:
                    mirobj[0].header[kwd] = (fobj_sim[0].header[kwd], fobj_sim[0].header.comments[kwd])
                except ValueError: # ignore keyword if contains non-ASCII characters. May cause breakage
                    # usually a problem with HISTORY keywords
                    continue
       
        # change the example mirage header to the correct filter name
        # that amisim used... the nmae must be an allowd ami filter
        # actul bandpasses can be manipulated when creating the simulation.
        mirobj[0].header["FILTER"] = filt
        if verbose: print(" TARGNAME =", mirobj[0].header["TARGNAME"],  " ",
                          mirobj[0].header["FILTER"],
                          " Input", fobj_sim[0].data.shape,
                          " Output", mirobj[1].data.shape)

        # write out miragized sim data
        mirobj.writeto(datadir+fname+mirext+".fits", overwrite=True)
        mirfns.append(fname+mirext+".fits")

    if len(mirfns) == 1: mirfns = mirfns[0]
    return mirfns


def test_amisim2mirage():
    amisim2mirage(
        os.path.expanduser('~')+"/Downloads/asoulain_arch2019.12.07/Simulated_data/",
        ("t_dsk_100mas__F430M_81_flat_x11__00",
         "c_dsk_100mas__F430M_81_flat_x11__00",
        ),
        os.path.expanduser('~') +\
        "/gitsrc/ImPlaneIA/example_data/example_niriss/" + \
        "jw00793001001_01101_00001_nis_cal.fits" ,
        "F430M"
    )

if __name__ == "__main__":

    testmain()


