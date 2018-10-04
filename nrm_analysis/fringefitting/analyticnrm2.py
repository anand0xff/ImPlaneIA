#! /usr/bin/env  python 
# Heritage mathematia nb from Alex & Laurent
# Python by Alex Greenbaum & Anand Sivaramakrishnan Jan 2013
# updated May 2013 to include hexagonal envelope

from __future__ import print_function
import numpy as np
import scipy.special
import sys
from astropy.io import fits
import os
import matplotlib.pyplot as pl
import scipy.special
#import unittest

from astropy import units as u
from astropy.units import cds
cds.enable()

print ("CWD: "+os.getcwd())

# in a module...
from  .. import misctools  # why can't I import misctools.utils this way too????
from . import hextransformEE # change to rel imports!

def image_center(fov, oversample, psf_offset):
    """ Image center location in oversampled pixels
    input
    fov: integer nmber of detector pixels of field of view
    oversample: integer samples per detector pixel pitch
    psf_offset: 2d numpy array, offset from image center in detecor pixels
    returns 2d numpy array of the offset of the psf center from the array center.
    """

    ImCtr = np.array( misctools.utils.centerpoint((oversample*fov,oversample*fov)) ) + \
            np.array((psf_offset[1], psf_offset[0]))*oversample # NOTE flip 1 and 0
    return ImCtr


def Jinc(x, y, **kwargs): # LG++
    """ D/m diam, lam/m, pitch/rad , returns real tfm of circular aperture at wavelength
        Peak value Unity, first zero occurs when theta = 1.22 lambda/D,
        Dimensionless argument rho =  pi * theta D / lambda
        Jinc = 2 J1(rho) / rho
        TBD Extend to unequal x and y pitch by
            arg = pi sqrt((xpitch xdist)^2 + (ypitch ydist)^2) D / lambda
        Use centerpoint(s): return (0.5*s[0] - 0.5,  0.5*s[1] - 0.5)
        correct for Jinc, hextransform to place peak in:
            central pixel (odd array) 
            pixel corner (even array)
        use c[0] - 1 to move peak *down* in ds9
        use c[1] - 1 to move peak *left* in ds9

        As it stands here - 
        LG+  Jinc(0) = pi/4
        LG++ Jinc(0) = 1
    """
    # x, y are the integer fromfunc array indices
    c = kwargs['c']  # in 'integer detector pixel' units, in the oversampled detector pixel space.
    pitch = kwargs['pitch'] # AS assumes this is in radians.
    D = kwargs['D'] # length: eg meters
    lam = kwargs['lam'] # length: eg meters
    affine2d = kwargs['affine2d']
    """
    xc,yc defined as  x-c[0], y-c[1] - these xc,yc are the coords that have the
    affine tfmn defined appropriately, with the origin of the affine2d unchanged
    by the tfmn.
    The analytical Jinc is centered on this origin in xc,yc space.
    """
    # where do we put pitchx pitchy in?  tbd
    xprime, yprime = affine2d.distortFargs(x-c[0],y-c[1])
    rho = np.pi * (D / lam) *  np.sqrt(pow(pitch*xprime,2) + pow(pitch*yprime,2))

    J = 2.0 * scipy.special.jv(1, rho) / rho
    nanposition=np.where(np.isnan(J))
    if len(nanposition[0] == 0):  J[nanposition]=1.0
    return J*affine2d.distortphase(x-c[0],y-c[1])


def phasor(kx, ky, hx, hy, lam, phi_m, pitch, affine2d):
    """ 
    returns complex amplitude array of fringes phi to units of m -- way more
    physical for broadband simulations!!  kx ky image plane coords in radians
    (oversampling should be accounted for before this call) hx, hy hole centers
    in meters pitch is in radians in image plane
    LG
    =========================================== 


    k in units of "radians" hx/lam in units of "waves," requiring the 2pi.  

    Example calculation -- JWST longest baseline ~6m Nyquist sampled for 64mas
    at 4um hx/lam = 1.5e6 for one full cycle, when does 2pi k hx/lam = 2pi?  k
    = lam/hx = .66e-6 rad x ~2e5 = ~1.3e-1 as = 2 x ~65 mas, which is Nyquist.
    The 2pi needs to be there! That also means phi/lam is in waves, phi in
    meters 
    LG+
    2017 ===========================================
    
    affine2d.phase_2vector: numpy vector of length 2 for use in manually
    writing the dot product needed for the exponent in the transform theorem.
    Use this 2vec to dot with (x,y) in fromfunc to create the 'phase argument'
    generated by the affine2d transformation.  Since this uses an offset xo yo
    in pixels of the affine transformation, these are *NOT* affected by the
    'oversample' integer in image space at this point.  The pitch is already
    finer by *oversample here.  The x,y vector it is dotted with is in image
    space.

    u,v are transform domain (image) coordionates (not radio uv).
    From Affine2d code, G(u,v) = F{ ( my*u - sy*v) / Delta, 
                                    (-sx*u + mx*v) / Delta  }
    Identify kx numpy array with u, ky numpy array with v:
    F(u,v) is np.exp(-2*np.pi*1j*((pitch*u*hx + pitch*v*hy)/lam + (phi_m /lam) )) * affine_phase_term
    u =>  ( my*u - sy*v) / Delta    so write (pitch*hx*kxprime + pitch*hy*kyprime)/lam + phi_m/lam
    v =>  (-sx*u + mx*v) / Delta
    so we write 
        np.exp(-2*np.pi*1j*((pitch*hx*kxprime + pitch*hy*kyprime)/lam + phi_m/lam))
    where:
        kxprime = ( my*kx - sy*ky)/Delta
        kyprime = (-sx*kx + mx*ky)/Delta
    LG++
    2018 ===========================================
    """

    kxprime, kyprime = affine2d.distortFargs(kx,ky)
    return np.exp(-2*np.pi*1j*\
             ((pitch*hx*kxprime + pitch*hy*kyprime)/lam + phi_m/lam)) * \
             affine2d.distortphase(kx,ky)


def interf(kx, ky, **kwargs):
    """  returns complex amplitude of fringes.  
         Use k to remind us that it is spatial frequency (angles)
         in (oversampled by this point, if you set up the oversampling) image space.
         kx, ky is 'detector pitch/oversample' by this point, in radians
    """
    psfctr = kwargs['c'] # the center of the PSF, in simulation pixels (ie oversampled)
    ctrs = kwargs['ctrs'] # hole centers
    phi = kwargs['phi']
    lam = kwargs['lam']
    pitch = kwargs['pitch'] # detpixscale/oversample
    affine2d = kwargs['affine2d']
    print(" psfctr ", psfctr)
    print(" ctrs ", ctrs)
    print(" phi ",  phi)
    print(" lam ", lam)
    print(" pitch ", pitch)
    print(" affine2d ", affine2d.name)

    # Question: where should the affine transf of psf_offset be done?  Here before phasor?
    # Figure out wht is correct... 
    # I suspect we don't 'affine2d transform the psf offsets - 
    # they are already in distorted coords... (?)   AS LG++ 08/14 2018 Ann Arbor
    fringe_complexamp = 0j
    for hole, ctr in enumerate(ctrs):
        fringe_complexamp += phasor((kx - psfctr[0]), (ky - psfctr[1]), 
                                    ctr[0], ctr[1], lam, phi[hole], pitch, affine2d)
    # debugging shows fringe orients to be same as hex orients & rect orients
    return fringe_complexamp # now affine2d angle rotates image CCW.


def ASF(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset, affine2d, verbose=False):
    """ returns real array 
        psf_offset in oversampled pixels,  phi/m """
    """ Update with appropriate corrections from hexeeLGplusplus 2017 AS 
        straighten out transposes """
    """ 
        2018 01 22  switch offsets x for y to move envelope same way as fringes:
        Envelopes don't get transposed, fringes do
        BTW ctrs are not used, but left in for identical calling sequence of these 
        kinds of fromfunction feeders...
    """
    pitch = detpixel/float(oversample)
    ImCtr = np.array( misctools.utils.centerpoint((oversample*fov,oversample*fov)) ) + \
            np.array((psf_offset[1],psf_offset[0]))*oversample # note flip 1 and 0
    ImCtr =  image_center(fov, oversample, psf_offset)
    print("ASF ImCtr {0}".format(ImCtr))
    return np.fromfunction(Jinc, (oversample*fov,oversample*fov),
                           c=ImCtr, 
                           D=d, 
                           lam=lam, 
                           pitch=pitch,
                           affine2d=affine2d)


def ASFfringe(detpixel, fov, oversample, ctrs, lam, phi, psf_offset, affine2d, 
              verbose=False):
    " returns real +/- array "
    pitch = detpixel/float(oversample)
    ImCtr = np.array( misctools.utils.centerpoint((oversample*fov,oversample*fov)) ) + \
            np.array(psf_offset)*oversample 
    ImCtr =  image_center(fov, oversample, psf_offset)
    print("ASFfringe ImCtr {0}".format(ImCtr))
    return np.fromfunction(interf, (oversample*fov,oversample*fov), 
                           c=ImCtr,
                           ctrs=ctrs, 
                           phi=phi,
                           lam=lam, 
                           pitch=pitch,
                           affine2d=affine2d)

def ASFhex(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset, affine2d): 
    " returns real +/- array "
    """ 
        2018 01 22  switch offsets x for y to move envelope same way as fringes:
        BTW ctrs are not used, but left in for identical calling sequence of these 
        kinds of fromfunction feeders...
        2018 09 07: anand@stsci.edu - in getting to beta release of LG++ I se this arbitrary
        switching of x and y and feel this should be tested properly for each type of
        fromfunction use, interf, jinc. hex, and so on.
    """
    pitch = detpixel/float(oversample)
    ImCtr = np.array( misctools.utils.centerpoint((oversample*fov,oversample*fov)) ) + \
            np.array((psf_offset[1],psf_offset[0]))*oversample # note flip 1 and 0
    ImCtr =  image_center(fov, oversample, psf_offset)
    print("ASFhex ImCtr {0}".format(ImCtr))
    # debugging code to try out affine2d rotations, unify fringe model orient w/pupil orient
    if 1: # normal operations
        return hextransformEE.hextransform(
                           s=(oversample*fov,oversample*fov), 
                           c=ImCtr, 
                           d=d, 
                           lam=lam, 
                           pitch=pitch,
                           affine2d=affine2d)
    if 0: # debugging of hexEE 
        return hextransformEE.recttransform(
                           s=(oversample*fov,oversample*fov), 
                           c=ImCtr, 
                           d=d, 
                           lam=lam, 
                           pitch=pitch,
                           affine2d=affine2d)

def PSF(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset, affine2d,
        shape = 'circ', verbose=False):
    """
        detpixel/rad 
        fov/detpixels 
        oversample 
        ctrs/m 
        d/m, 
        lam/m, 
        phi/m

        returns real array

        NEW LG++: 
            psf_offset (x,y) in detector pixels, used as an offset, 
            the actual psf center is fov/2.0 + psf_offset[x or y] (in detector pixels)
        affine2d - a near-identity or near-unity abs(determinant).
        shape: one of 'circonly', 'circ', 'hexonly', 'hex', 'fringeonly'
    """

    misctools.utils.printout(ctrs, "                                   analyticnrm2:PSF_"+affine2d.name)

    # Now deal with primary beam shapes... 
    if shape == 'circ': 
        asf_fringe = ASFfringe(detpixel, fov, oversample, ctrs, lam, phi, psf_offset, affine2d)
        asf = ASF(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset, affine2d) * asf_fringe
    elif shape == 'circonly':
        asf = ASF(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset, affine2d)
    elif shape == 'hex': 
        asf_fringe = ASFfringe(detpixel, fov, oversample, ctrs, lam, phi, psf_offset, affine2d)
        asf = ASFhex(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset, affine2d) * asf_fringe
    elif shape == 'hexonly': 
        asf = ASFhex(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset, affine2d)
    elif shape == 'fringeonly':
        asf_fringe = ASFfringe(detpixel, fov, oversample, ctrs, lam, phi, psf_offset, affine2d)
        asf = asf_fringe
    else:
        raise ValueError(
            "pupil shape %s not supported - choices: 'circonly', 'circ', 'hexonly', 'hex', 'fringeonly'"\
            % shape)

    if verbose:
        print("-----------------")
        print(" PSF Parameters:")
        print("-----------------")
        print(("pitch: {0}, fov: {1}, oversampling: {2}, centers: {3}".format(detpixel,
            fov, oversample, ctrs) + 
            "d: {0}, wavelength: {1}, pistons: {2}, shape: {3}".format(d, lam, 
            phi, shape)))

    return  (asf*asf.conj()).real

######################################################################
#  New in LG++ - harmonic fringes
#  New in LG++ - model_array(), ffc, ffs moved here from leastsqnrm.py
######################################################################
def harmonicfringes(**kwargs):
    """  returns sine and cosine fringes.  real arrays, in image space 
         this works in the oversampled space that is each slice of the model
         TBD: switch to pitch for calc here in calls to ffc ffs
    """
    fov = kwargs['fov'] # in detpix
    pitch = kwargs['pitch'] # detpixscale
    psf_offset = kwargs['psf_offset'] # the PSF ctr, detpix
    baseline = kwargs['baseline'] # hole centers' vector, m
    lam = kwargs['lam'] # m
    oversample = kwargs['oversample']
    affine2d = kwargs['affine2d']

    cpitch = pitch/oversample
    ImCtr = np.array( misctools.utils.centerpoint((oversample*fov,oversample*fov)) ) + \
            np.array(psf_offset)*oversample # first  no flip of 1 and 0, no transpose
    ImCtr =  image_center(fov, oversample, psf_offset)

    if 0:
        print(" harmonicfringes: ", end='')
        print(" ImCtr {}".format( ImCtr), end="" )
        print(" lam {}".format( lam) )
        print(" detpix pitch {}".format( pitch) )
        print(" pitch for calculation {}".format( pitch/oversample) )
        print(" over  {}".format( oversample), end="" )
        print(" fov/detpix  {}".format( fov), end="" )

    return (np.fromfunction(ffc, (fov*oversample, fov*oversample), c=ImCtr,
                                                                   baseline=baseline,
                                                                   lam=lam, pitch=cpitch,
                                                                   affine2d=affine2d),
            np.fromfunction(ffs, (fov*oversample, fov*oversample), c=ImCtr,
                                                                   baseline=baseline,
                                                                   lam=lam, pitch=cpitch,
                                                                   affine2d=affine2d))

def ffc(kx, ky, **kwargs):
    ko = kwargs['c'] # the PSF ctr
    baseline = kwargs['baseline'] # hole centers' vector
    lam = kwargs['lam'] # m
    pitch = kwargs['pitch'] # pitch for calcn = detpixscale/oversample
    affine2d = kwargs['affine2d']
    kxprime, kyprime = affine2d.distortFargs(kx-ko[0], ky-ko[1])
    return 2*np.cos(2*np.pi*pitch*(kxprime*baseline[0] + kyprime*baseline[1]) / lam)

def ffs(kx, ky, **kwargs):
    ko = kwargs['c'] # the PSF ctr
    baseline = kwargs['baseline'] # hole centers' vector
    lam = kwargs['lam'] # m
    pitch = kwargs['pitch'] # pitch for calcn = detpixscale/oversample
    affine2d = kwargs['affine2d']
    kxprime, kyprime = affine2d.distortFargs(kx-ko[0], ky-ko[1])
    # print("*****  pitch for ffc ffs {}".format( pitch) )
    return 2*np.sin(2*np.pi*pitch*(kxprime*baseline[0] + kyprime*baseline[1]) / lam)



def model_array(ctrs, lam, oversample, pitch, fov, d, psf_offset=(0,0),
                shape ='circ', affine2d=None, verbose=False):
    # pitch is detpixel
    # psf_offset in detpix
    # returns real 2d array of primary beam, list of fringe arays

    misctools.utils.printout(ctrs, "                                   analyticnrm2:model_array"+affine2d.name)

    nholes = ctrs.shape[0]
    phi = np.zeros((nholes,)) # no phase errors in the model slices...
    modelshape = (fov*oversample, fov*oversample)  # spatial extent of image model - the oversampled array
    
    if verbose:
        print("------------------")
        print(" Model Parameters:")
        print("------------------")
        print("pitch: {0}, fov: {1}, oversampling: {2}, centers: {3}".format(pitch,
            fov, oversample, ctrs) + \
            " d: {0}, wavelength: {1}, shape: {2}".format(d, lam, shape) +\
            "\ncentering:{0}\n {1}".format(centering, off))

    # calculate primary beam envelope (non-negative real)
    # ASF(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset) * asf_fringe
    if shape=='circ':
        asf_pb = ASF(   pitch, fov, oversample, ctrs, d, lam, phi, psf_offset, affine2d)
    elif shape=='hex':
        asf_pb = ASFhex(pitch, fov, oversample, ctrs, d, lam, phi, psf_offset, affine2d)
    else:
        raise KeyError("Must provide a valid hole shape. Current supported shapes are" \
                " 'circ' and 'hex'.")
    #rimary_beam = asf_pb * asf_pb  LG++ change in type explicit
    
    # test that this array is almost completely real...
    #print("***>>> asf_pb.reals: {}  asf_pb.imags:{} ".format( np.abs(asf_pb.real).sum(), np.abs(asf_pb.imag).sum()))
    # ... yes it is overwhelmingly real, like 1e-8ish in imaginary cf real.
    primary_beam = (asf_pb*asf_pb.conj()).real
    

    alist = []
    for i in range(nholes - 1):
        for j in range(nholes - 1):
            if j + i + 1 < nholes:
                alist = np.append(alist, i)
                alist = np.append(alist, j + i + 1)
    alist = alist.reshape((len(alist)//2, 2))

    ffmodel = []
    ffmodel.append(nholes * np.ones(modelshape))
    for basepair in alist:
        #print("i", int(basepair[0]), end="")
        #print("  j", int(basepair[1]), end="")
        baseline = ctrs[int(basepair[0])] - ctrs[int(basepair[1])]
        #print(baseline)
        cosfringe, sinfringe = harmonicfringes(fov=fov, pitch=pitch, psf_offset=psf_offset,
                                               baseline=baseline,
                                               oversample=oversample,
                                               lam=lam,
                                               affine2d=affine2d)
        ffmodel.append( cosfringe )
        ffmodel.append( sinfringe )

    return primary_beam, ffmodel


def multiplyenv(env, fringeterms):
    # The envelope is size (fov, fov). This multiplies the envelope by each of the 43 slices
    # (if 7 holes) in the fringe model
    full = np.ones((np.shape(fringeterms)[1], np.shape(fringeterms)[2], np.shape(fringeterms)[0]+1))
    for i in range(len(fringeterms)):
        full[:,:,i] = env * fringeterms[i]
    return full
