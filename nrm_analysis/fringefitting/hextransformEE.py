#! /usr/bin/env python
import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as pl
from astropy import units as u
# from astropy.units import cds
# cds.enable()
#>>> distance = 42.0 * u.meter; then: dfloat = distance.value
#>>> x = 1.0 * u.parsec
#>>> x.to(u.km)  
# cds.enable()
#>>> r = 1 * u.rad # <Quantity 1.0 rad>
#>>> r.value # 1.0
#>>> from astropy.units import cds
#>>> cds.enable()
#>>> d = r.to(u.arcsec) # <Quantity 206264.80624709636 arcsec>
#>>> e = r.to(u.marcsec)# <Quantity 206264806.24709633 marcsec>

""" Python implementation: anand@stsci.edu 6 Mar 2013
    Algorithm: eelliott@stsci.edu -  Applied Optics, Vol 44, No. 8 10 March 2005  Sabatke et al.
    Erin Elliott's analytical hexagon-aperture PSF, page 1361 equation 5
    Coordinate center at center of symmetry, and flat edge along xi axis
        ---  eta
      /     \ ^ 
      \     / |
        ---   -> xi
    hex(xi,eta) = g(xi,eta) + g(-xi,eta)
"""

def gfunction(xi, eta, **kwargs):  #was g_eeAG
    c = kwargs['c']
    pixel = kwargs['pixel']
    d = kwargs['d']
    lam = kwargs['lam']
    xi = (d/lam)*pixel*(xi - c[0])
    eta = (d/lam)*pixel*(eta - c[1])
    affine2d = kwargs["affine2d"]

    i = 1j
    Pi = np.pi

    xip, etap = affine2d.distortFargs(xi,eta)

    if kwargs['minus'] is True:
        xip = -1*xip

    g = np.exp(-i*Pi*(2*etap/np.sqrt(3) + xip)) * \
        ( \
          (np.sqrt(3)*etap - 3*xip) * \
          (np.exp(i*Pi*np.sqrt(3)*etap) - np.exp(i*Pi*(4*etap/np.sqrt(3) + xip)))  + \
          (np.sqrt(3)*etap + 3*xip) * \
          (np.exp(i*Pi*etap/np.sqrt(3)) - np.exp(i*Pi*xip)) \
         ) / \
         (4*Pi*Pi*(etap*etap*etap - 3*etap*xip*xip))

    return g*affine2d.distortphase(xi,eta)

"""
def glimit(xi, eta, **kwargs):
    c = kwargs['c']
    pixel = kwargs['pixel']
    d = kwargs['d']
    lam = kwargs['lam']
    xi = (d/lam)*pixel*(xi - c[0])
    eta = (d/lam)*pixel*(eta - c[1])
    affine2d = kwargs["affine2d"]

    if kwargs['minus'] is True:
        xi = -1*xi
    Pi = np.pi
    xip, etap = affine2d.distortFargs(xi,eta)
    g = ( np.exp(-1j*Pi*xi)/(2*np.sqrt(3)*Pi*Pi*xi*xi) ) * \
        ( -1 + 1j*Pi*xi + np.exp(1j*Pi*xi)-2j*Pi*xi*np.exp(1j*Pi*xi) )

    return g*affine2d.distortphase(xi,eta)
"""

"""
def centralpix(xi, eta, **kwargs):
    c = kwargs['c']
    pixel = kwargs['pixel']
    d = kwargs['d']
    lam = kwargs['lam']
    xi = (d/lam)*pixel*(xi - c[0])
    eta = (d/lam)*pixel*(eta - c[1])

    if kwargs['minus'] is True:
        xi = -1*xi
    i = 1j
    Pi = np.pi
    
    xip, etap = affine2d.distortFargs(xi,eta)
    g = np.sqrt(3) / 4.0
    return g*affine2d.distortphase(xi,eta)
"""


def hextransform(s=None, c=None, d=None, lam=None, pitch=None, affine2d=None, 
                 DEBUG=False, odir=None): # was hex_eeAG
    """
        returns the complex array analytical transform of a (distorted if necessary) hexagon
        d/m flat-to-flat distance, lam/m, pitch/rad
        c is the center of the PSF (if no phase aberrations), NOT an offset from array center.
    """
    """
    LG++ avoids NaN's by avoiding exact rotations of 15 degrees in Affine2d, for example,
    or finding pupil rotation by hand

    LG++ avoids central pixel singularity by offsetting any integer-like offset request by 1e-13
    of an oversampled pixel
    """
    if c is None:
        """  anand@stsci.edu Wed Mar 22 11:54:54 EDT 2017
        Further poking 2017 for LG++
        pre-2017 code tried to make fft2 slopes zero in recovered pupils... but this 
        is probably an incorrect interpretation of zero frequency in the analytical-to-FFT algo translation.
        Now both ASF modulus' are centered on pixel (odd array) /corner (even array),
        and the imaginary parts of the hex ASF are machine-noise close to zero.

        Test results:

        Center: (2.0, 2.0) Shape: (5, 5)
            hex imag mean max min 0.0 9.36750677027e-17 -9.36750677027e-17
        Center: (2.5, 2.5) Shape: (6, 6)
            hex imag mean max min 0.0 3.12250225676e-16 -3.12250225676e-16
        Center: (50.0, 50.0) Shape: (101, 101)
            hex imag mean max min 0.0 8.22258927613e-16 -8.22258927613e-16
        Center: (49.5, 49.5) Shape: (100, 100)
            hex imag mean max min 0.0 3.60952587264e-15 -3.60952587264e-15
        """
        c = (float(s[0])/2.0  - 0.5,  float(s[1])/2.0  - 0.5)

    # deal with central pixel singularity:
    c_adjust = c
    cpsingularityflag = False

    epsilon_offset = 1.0e-8 
    # check psf - psf_rotated_by_90 for 'noise' at
    # singularity lines to optimize choice
    # OV 1
    # with 1e-8 the noise is at most few x 1e-8 cf central hex value (0.75) of the PSF. 
    # At this offset, the 5deg - rot(95 deg) noise is ~1e-8 so not significant cf 0.75 peak.
    # OV 5  difference noise < 1e-7
    # OV 13  difference noise < 1e-5 for CP on 0 - 90, -13 elsewhere; for 5 - 95 noise is -13 everywhere.
    #
    # Conclusion - finer oversampling reduces the singularity noise effect...
    # For critical work, ignore the line of pixels vertically/horizontally through
    # the origin if the hex rotation is exactly zero.
    #
    # Could use refactoring to eg average two rows either side of singularity???
    # Might be better numerically.  Obviously a  WIP...  Jun 21 2018.
    #
    # anand@stsci.edu Jun 2018
    
    d0, d1 = (c[0] - int(c[0]), c[1] - int(c[1]))  # the fractional part of the offsets
    # Are they very small (i.e. 'almost exactly' centered on 0)?
    if (abs(d0) <  0.5*epsilon_offset): # we might have the singular central pixel here...
        c_adjust[0] = c[0]+ epsilon_offset
        cpsingularityflag = True  # if so set up the central pixel singularity flag
    if abs(d1) <  0.5*epsilon_offset: # we might have the singular central pixel here...
        c_adjust[1] = c[1]+ epsilon_offset
        cpsingularityflag = True  # if so set up the central pixel singularity flag

    hex_complex = np.fromfunction(gfunction, s, d=d, c=c_adjust, lam=lam, pixel=pitch, affine2d=affine2d, minus=False) + \
                  np.fromfunction(gfunction, s, d=d, c=c_adjust, lam=lam, pixel=pitch, affine2d=affine2d, minus=True)

    if cpsingularityflag:
        #print("**** info:  central pixel singularity - nudge center by epsilon_offset {0:.1e}, c0,c1=({1:f},{2:f}), determinant={3:.4e} ".format(epsilon_offset, int(c[0]), int(c[1]), affine2d.determinant))
        pass

    FUDGE = np.sqrt(4.0)  # this gives the analyticcentral PSF correctly.  Figure it out later if needed.
    # At center of psf distortion phasor is unity, so just use determinant...
    hex_complex[int(c[0]),int(c[1])] = FUDGE * (np.sqrt(3) / 4.0) # * affine2d.determinant seems not to be needed

    if DEBUG: # only to be used with caution, and affine2d being the Identity tfmn.
        ## hr = hex_complex.real
        ## hi = hex_complex.imag
        ## print(("\t***\thex real mean max min",  hr.mean(), hr.max(), hr.min()))
        ## print(("\t***\thex imag mean max min",  hi.mean(), hi.max(), hi.min(), " should be very small"))

        """  anand@stsci.edu Wed Mar 22 11:54:54 EDT 2017  DEBUG on..."""
        """  anand@stsci.edu Thu Nov  9 11:00:10 EST 2017
             This fft pupil recovery is approximate, because of differing 
             centering interpretations of FFT cf analytical calculation of ASF (probably) 
             The sanity check is to see the 'spatially-filtered by finite fov' pupil
             that matches the assumed hex geometry (flat up, not point-up)
        hexhole = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(hex_complex)))
        fits.PrimaryHDU(data=np.abs(hexhole) ).writeto(odir+"%d_hexpupilrecovered_mod.fits"%s[0], overwrite=True)
        maxmod = np.abs(hexhole).max()
        maskpha = np.where(np.abs(hexhole)<maxmod/10.0)
        """ """ phase via this fft is misleading - anand@stsci.edu """

        (xnan, ynan)= np.where(np.isnan(hex_complex))
        print("explore central pixel singularity", type(xnan), xnan, ynan)
        print("************Are there any NaN's???  len(np.where(np.isnan(hex_complex))[0])", len(xnan))

    return hex_complex



def recttransform(s=None, c=None, d=None, lam=None, pitch=None, affine2d=None, 
                 DEBUG=False, odir=None): 
    """
        returns the analytical transform of a (distorted if necessary) rectangle
        d/m longer flat-to-flat distance, lam/m, pitch/rad
        Rectangle assumed to be half as wide as long (rat=0.5).
        c is the center of the PSF (if no phase aberrations), NOT an offset from array center.
    """
    if c is None:
        c = (float(s[0])/2.0  - 0.5,  float(s[1])/2.0  - 0.5)

    rat = 0.5 # y dimension of rectangle / x dimension of rectangle
    rect_complex = np.fromfunction(sincxy, s, a=d, b=rat*d, c=c, lam=lam, pitch=pitch, affine2d=affine2d)
    return rect_complex



def sincxy(x, y, **kwargs): # LG++
    """ d/m diam, lam/m, pitch/rad , returns real tfm of rectangular aperture
        Use centerpoint(s): return (0.5*s[0] - 0.5,  0.5*s[1] - 0.5)
        Place peak in:
            central pixel (odd array) 
            pixel corner (even array)
        use c[0] - 1 to move peak *down* in ds9
        use c[1] - 1 to move peak *left* in ds9
    """
    # x, y are the integer fromfunc array indices
    c = kwargs['c']  # in 'integer detector pixel' units, in the oversampled detector pixel space.
    pitch = kwargs['pitch'] # AS assumes this is in radians.
    a = kwargs['a'] # length: eg meters
    b = kwargs['b'] # width: eg meters
    # Force a to be 6.5m
    a = 6.5
    b = a/2.0
    lam = kwargs['lam'] # length: eg meters
    affine2d = kwargs['affine2d']
    """
    Image plane origin xc,yc is given by x-c[0], y-c[1] 
        These xc,yc are the coords that have the affine tfmn defined appropriately,
        with the origin of the affine2d unchanged by the tfmn.
    The analytical sinc is centered on this origin
    """
    # where do we put pitchx pitchy in?  tbc
    xprime, yprime = affine2d.distortFargs(x-c[0],y-c[1])
    return np.sinc((np.pi*a/lam) * (xprime*pitch)) * np.sinc((np.pi*b/lam) * (yprime*pitch)) * \
           affine2d.distortphase( (x-c[0])*pitch/lam, (y-c[1])*pitch/lam )


"""
def test_hextransform(s, odir="", pitch=None, c=None):
    modasf = np.abs(hextransform(s=s,  c=c, d=0.80, lam=4.3e-6, pitch=pitch, DEBUG=True, odir=odir))
    fits.PrimaryHDU(data=modasf).writeto(odir+"%d_HexModASFlimtest.fits"%s[0], overwrite=True)


def main():

    # image plane sampled to large extent - less pupil-mushing due to finite fov
    # Check the recovered pupils are 'flat side up', not 'point-up'
    ODIR = "./test_hexee_dbg/10x_" # image plane sampled to large extent - less pupil-mushing due to finite fov
    pitch = 10.0 * ((65*u.marcsec).to(u.rad)).value
    test_hextransform(s=(101,101), odir=ODIR, pitch=pitch)
    test_hextransform(s=(100,100), odir=ODIR, pitch=pitch)

    ODIR = "./test_hexee_dbg/"
    pitch = ((65*u.marcsec).to(u.rad)).value
    test_hextransform(s=(5,5),     odir=ODIR, pitch=pitch)  # easy to see centering in modulus of ASF fits files
    test_hextransform(s=(6,6),     odir=ODIR, pitch=pitch)  # easy to see centering in modulus of ASF fits files
    test_hextransform(s=(101,101), odir=ODIR, pitch=pitch)
    test_hextransform(s=(100,100), odir=ODIR, pitch=pitch)

    # center of ASF is always float(shape[0]/2.0 - 0.5,...
    nn = 11
    imcx, imcy = (0.5*nn - 0.5,  0.5*nn - 0.5)
    pitch = 10.0 * ((65*u.marcsec).to(u.rad)).value
    
    ODIR = "./test_hexee_dbg/dx0dy0_" # move center of hex transform to see ds9 directions
    test_hextransform(s=(nn,nn), odir=ODIR, pitch=pitch, c=(imcx, imcy))

    ODIR = "./test_hexee_dbg/dx1dy0_" # move center of hex transform to see ds9 directions
    test_hextransform(s=(nn,nn), odir=ODIR, pitch=pitch, c=(imcx - 1.0, imcy))

    ODIR = "./test_hexee_dbg/dx0dy1_" # move center of hex transform to see ds9 directions
    test_hextransform(s=(nn,nn), odir=ODIR, pitch=pitch, c=(imcx, imcy - 1.0))

if __name__ == "__main__":

    main()
"""

