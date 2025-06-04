#! /usr/bin/env python

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
import sys
import argparse
import time
from scipy.integrate import simps 
from nrm_analysis import nrm_core, InstrumentData, fringefitting

""" Hokey but in webbpsf_driver_NIRISS_nojitter.py output for A0V on a 255x255 detector we see:   

filter      sampling  psftot    psfpeak       cpf
F430_G2V    detector    0.1503   2.6923e-5   0.01945

F480M A0V:
    Total of PSF is: 0.14983928265312782
    Peak pixel is:  0.0023813565623978152
    Central pixel fraction is:  0.015892738674614215
F430M_A0V:
    Total of PSF is: 0.15029221331182707
    Peak pixel is:  0.0029221329659187313
    Central pixel fraction is:  0.01944300973102229
F380M_A0V
    Total of PSF is: 0.15064757333183165
    Peak pixel is:  0.0035334896463460517
    Central pixel fraction is:  0.023455337302797623
F377W_A0V:
    Total of PSF is: 0.1515569249392621
    Peak pixel is:  0.005893341903346654
    Central pixel fraction is:  0.038885335696197766
"""
#use A0V for photometry for now...

filt_psftot = {"F480M":0.1498,
               "F430M":0.1503,
               "F380M":0.1506,
               "F277W":0.1516
               }
filt_cpf    = {"F480M":0.002381,
               "F430M":0.002922,
               "F380M":0.023455,
               "F277W":0.038885,
               }

webbpsffiledir = "./webbpsffiles_for_filters/"
def get_webbpsffilters(mono=False):
    """ 
        mono = False: full filter bandpass
        mono = True: transmission-weighted central wavelength of filter

        read in psf headers and scoop out filter bandpasses 
        returns bpd, cwd, ewd, betad dictionaries keyed by filter name
        filter bandpass dictionary values are np array[npoints,2] with
           WGHT = filt_bpd[filt][:,0]
           WAVE = filt_bpd[filt][:,1] 
        central wavelength dictionary values weighted central wavelength / m
        equivalent width dictionary values in wavelength / m
        fractional bandpass dictionary values dimensionless
    """
    m = 1.0
    um = 1.0e-6 * m

    filters = ("F480M", "F430M", "F380M", "F277W")
    pupil = "MASK_NRM"
    sptype = "A0V" # just to match file name of webbpsf psf files

    # number of points used by webbpsf to calc the psf
    # /Users/_moi_/anaconda3/envs/astroconda/share/webbpsf-data/NIRISS/filters.tsv
    # number of wavelength across filters in dictionary form
    filt_n = {"F480M":19, "F430M":19, "F380M":19, "F277W":41}  

    # filter bandpass dictionary
    filt_bpd = {"F480M":np.zeros((filt_n["F480M"],2)), 
                "F430M":np.zeros((filt_n["F430M"],2)), 
                "F380M":np.zeros((filt_n["F380M"],2)), 
                "F277W":np.zeros((filt_n["F277W"],2))
                }
    filt_cw = {}  # weighted central wavelength / m
    filt_ew = {}  # equiv width / m
    filt_beta = {}  # fractional bandpass
    for filt in filters:
        print(filt, filt_n[filt], "points")
        hdr  = fits.getheader(webbpsffiledir+"PSF_det_%s_%s_%s.fits"%(pupil, filt, sptype))

        # read in filter weights & wavelengths (meters)
        # order of 0 & 1 matches use in simulateG7S6_f430m.py
        for ii in range(filt_n[filt]):
            filt_bpd[filt][ii,1] = hdr["WAVE{:d}".format(ii)]
            filt_bpd[filt][ii,0] = hdr["WGHT{:d}".format(ii)]
        WGHT = filt_bpd[filt][:,0]
        WAVE = filt_bpd[filt][:,1]

        wavelen = (WAVE*WGHT).sum()/WGHT.sum() # Weighted mean wavelength in meters
        area = simps(WGHT, WAVE)
        ew = area / WGHT.max()
        filt_cw[filt] = wavelen
        filt_ew[filt] = ew
        filt_beta[filt] = ew/wavelen
        print("\tctr wavelength {:.4e} um".format(wavelen/um))
        print("\tequiv width {:.4e} um".format(ew/um))
        print("\tfrac bandpass {:.2e}".format(ew/wavelen))
        print()

    if mono:
         filt_bpd, filt_cw, filt_ew, filt_beta = monochromatize(filters, filt_bpd, filt_cw, filt_ew, filt_beta)
    
    return filt_bpd, filt_cw, filt_ew, filt_beta

def monochromatize(ffs, bpd, cwd, ewd, betad):
    """ If we need to use the bandpass-weighted monochromatic psf we'll add a different flag, and code 
        It seems better to calculate at a user-specific wavelength """
    # Define nominal wavelengths of each filter in dictionary form
    nominal_cwd = {"F480M":4.8e-6, "F430M":4.3e-6, "F380M":3.8e-6, "F277W":2.77e-6}  
    for ff in ffs:
        bpdm = np.zeros((1,2)) # filter's bandpass dictionary value for one wavelength
        bpdm[0,0] = 1.0        # weight for the wavelength
        bpdm[0,1] = nominal_cwd[ff]    # wavelength / m, used to be weighted central wavelenngth earlire
        cwd[ff] = nominal_cwd[ff]      # this is put in the fits header - keep it consistent!
        bpd[ff] = bpdm         # replace polychromatioc bandpass dictionary value with entries for monochromatic 
        ewd[ff] = 0.0          # equivalent width dictionary value for one wavelength
        betad[ff] = 0.0        # fractional bandwidth dictionary value for one wavelength
    return bpd, cwd, ewd, betad


def psf(filt, fbp, cw, ew, beta, data_dir,
              oversample = 11, 
              n_image = 81, 
              pixelscale_as=0.0656, 
              f2f = 0.80,
              saveover = True,
              savedet = False,
              chooseholes=None):
    
    arcsec2rad = u.arcsec.to(u.rad)

    WGHT = fbp[:,0]
    WAVE = fbp[:,1]

    # setup parameters for simulation
    verbose = 1
    overwrite = 1

    mask = 'MASK_NRM'

    # directory containing the test data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    name_seed = 'PSF_%s_%s_x%d'%(mask,filt,oversample)
    hmnem = ''
    for hh in chooseholes: hmnem = hmnem+hh
    print(hmnem)

    psf_image_name = name_seed + '_' + hmnem + '.fits'
    psf_image = os.path.join(data_dir,psf_image_name)
    psf_image_without_oversampling = os.path.join(data_dir,psf_image_name.replace('.fits','_det.fits'))
    print(psf_image_without_oversampling)

    if (not os.path.isfile(psf_image_without_oversampling)) | (overwrite): 
        from nrm_analysis.fringefitting.LG_Model import NRM_Model
        jw = NRM_Model(mask='jwst', holeshape="hex", verbose=True, chooseholes=chooseholes)
        jw.set_pixelscale(pixelscale_as*arcsec2rad)
        jw.simulate(fov=n_image, 
            bandpass=fbp, 
            over=oversample,
            )
        print("simulateG7S6psf: simulation oversampling is", oversample)
        (year, month, day, hour, minute, second, weekday, DOY, DST) =  time.gmtime()
        # optional writing of oversampled image
        if saveover:
            psf_over_n = jw.psf_over[:80*oversample,:80*oversample]/jw.psf_over[:80*oversample,:80*oversample].sum()
            psf_over_n = psf_over_n * filt_psftot[filt] # "sum of webbpsf sim done at 11x over"
            fits.writeto(psf_image, psf_over_n, overwrite=True)
            header = fits.getheader(psf_image)
            header['PIXELSCL'] = pixelscale_as/oversample, "arcsec per pixel"
            header['FILTER'] = filt
            header['PUPIL'] = mask 
            header["OVER"] = ( oversample, "oversample")
            header["SPTYPE"] = "FLAT",  "source spectral type unspecified"
            header["NWAVES"] = fbp.shape[0]
            header["WAVELEN"] = cw, "Weighted mean wavelength in meters "
            header["FILT_EW"] = ew, "Filter equiv width"
            header["FILT_BP"] = beta, "Fractional bandpass = WL/EqW"
            for i in range(len(WAVE)):
                header["WAVE%d"%i] = WAVE[i], "Wavelength %d"%i
                header["WGHT%d"%i] = WGHT[i], "Wavelength weight %d"%i
            header["FT"] = ( "analytical", "hexee * fringes")
            header['NRM_GEOM'] =  'G7S6', 'Beaulieu, PGT, AS, active ctrs'
            header['F2F'] = jw.d, "flat2flat hole size m"
            header['PSFTOT'] = filt_psftot[filt], "sum of webbpsf sim done at 11x over"
            header['PSFPEAK'] = psf_over_n.max()
            header["SRC"] = ( "simulateG7S6psf.py", "ImplaneIA/notebooks/")
            header['AUTHOR'] = '%s@%s' % (os.getenv('USER'), os.getenv('HOST')), 'username@host for calculation'
            header['DATE'] = '%4d-%02d-%02dT%02d:%02d:%02d' %  (year, month, day, hour, minute, second), 'Date of calculation'
            header['F2F'] = (0.82, "flat2flat hole size m")
            fits.update(psf_image, psf_over_n, header=header)

        if savedet:
            # PSF without oversampling
            psf_det_n = jw.psf[:80,:80]/jw.psf[:80,:80].sum()  # we can delete this line but I'm in a hurry...
            webbpsfnorm = filt_cpf[filt] / psf_det_n.max()
            psf_det_n = psf_det_n * webbpsfnorm # forces central pixel values to agree ^see above
            fits.writeto(psf_image_without_oversampling, psf_det_n, overwrite=True)
            header = fits.getheader(psf_image_without_oversampling)
            header['PIXELSCL'] = pixelscale_as, "arcsec per pixel"
            header['FILTER'] = filt
            header['PUPIL'] = mask 
            header["OVER"] = ( oversample, "oversample pre-rebin")
            header["SPTYPE"] = "FLAT", "source spectral type unspecified"
            header["NWAVES"] = fbp.shape[0]
            header["WAVELEN"] = cw, "Weighted mean wavelength in meters "
            header["FILT_EW"] = ew, "Filter equiv width"
            header["FILT_BP"] = beta, "Fractional bandpass = WL/EqW"
            for i in range(len(WAVE)):
                header["WAVE%d"%i] = WAVE[i], "Wavelength %d"%i
                header["WGHT%d"%i] = WGHT[i], "Wavelength weight %d"%i
            header["FT"] = ( "analytical", "hexee * fringes")
            header['NRM_GEOM'] =  'G7S6', 'Beaulieu, PGT, AS active ctrs'
            header['F2F'] = jw.d, "flat2flat hole size m"
            header['PSFTOT'] = psf_det_n.sum()
            header['PSFPEAK'] = psf_det_n.max()
            header['CPF'] = filt_cpf[filt]
            header["SRC"] = ( "simulateG7S6psf.py", "ImplaneIA/notebooks/")
            header['AUTHOR'] = '%s@%s' % (os.getenv('USER'), os.getenv('HOST')), 'username@host for calculation'
            header['DATE'] = '%4d-%02d-%02dT%02d:%02d:%02d' %  (year, month, day, hour, minute, second), 'Date of calculation'
            fits.update(psf_image_without_oversampling, psf_det_n, header=header)


if __name__ == "__main__":

    allholes = ('b4','c2','b5','b2','c1','b6','c6')
    b4,c2,b5,b2,c1,b6,c6 = allholes
    
    holesequence = (b6, b2, b4, c6, b5, c1, c2)
    print(len(holesequence))


    parser = argparse.ArgumentParser(description="Creates JWST AMI psfs in supported filters")
    parser.add_argument('-o','--over',  type=int, default='3', help='oversampling before binning to detector pixels (>=1)')
    parser.add_argument("-m", "--monochromatic", default=True, help = "monochromatic psf at nominal band centers", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    print("monochromatic", args.monochromatic)
    print("oversampling", args.over)
    bpd, cwd, ewd, betad = get_webbpsffilters(mono=args.monochromatic)
    filters = ("F480M", ) # "F430M", "F380M", "F277W")

    for ff in filters:
        print(ff)
        for ihole in range(len(holesequence)):
            psf(ff, bpd[ff], cwd[ff], ewd[ff], betad[ff],
                './simulatedpsfs_chooseholes/', n_image=81, 
                saveover=True, savedet=False, oversample=args.over,
                chooseholes=holesequence[:ihole+1])
