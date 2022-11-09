#!/usr/bin/env python
"""
NIRISS AMI calibration of binary point source AB Dor and calibrators HD37093, HD36805
Run ImPlaneIA ([Greenbaum, A. et al. 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...798...68G/abstract)) to extract observables in oifits format.

Introduction
This script runs ImPlaneIA on *_calints.fits files calibrated with JWST pipeline.
Raw observables are extracted.
To calibrate the raw observables run run_implaneia2.py.
"""

# for implaneia extraction...
import glob
import os
import time
import warnings
from argparse import ArgumentParser

import numpy as np
from astropy.io import fits
from nrm_analysis import InstrumentData, nrm_core
from nrm_analysis.misctools import utils
from nrm_analysis.misctools.implane2oifits import calibrate_oifits

# additionally, for candid binary extraction function...
import multiprocessing
from astropy import units as u
import matplotlib.pyplot as plt
from scipy import ndimage
import amical


# *Developer Note:*
# Plese follow the instructions on https://webbpsf.readthedocs.io/en/latest/installation.html to download WebbPSF data
# files and create WEBBPSF_PATH location.

# Define functions to run ImPlaneIA
np.set_printoptions(precision=4, linewidth=160)

def examine_observables(ff):
    """input: FringeFitter instance after fringes are fit"""

    print( "\nExamine_observables, standard deviations & variances of *independent* CP's and CAs:")
    print( "   Closure phase mean {:+.4f}  std dev {:.2e}  var {:.2e}".format(
            ff.nrm.redundant_cps.mean(),
            np.sqrt(utils.cp_var(ff.nrm.N, ff.nrm.redundant_cps)),
            utils.cp_var(ff.nrm.N, ff.nrm.redundant_cps),))
    print( "   Closure amp   mean {:+.4f}  std dev {:.2e}  var {:.2e}".format(
            ff.nrm.redundant_cas.mean(),
            np.sqrt(utils.cp_var(ff.nrm.N, ff.nrm.redundant_cas)),
            utils.cp_var(ff.nrm.N, ff.nrm.redundant_cas),))
    print( "    Fringe amp   mean {:+.4f}  std dev {:.2e}  var {:.2e}".format(
            ff.nrm.fringeamp.mean(), ff.nrm.fringeamp.std(), ff.nrm.fringeamp.var()))

    np.set_printoptions( precision=3, formatter={"float": lambda x: "{:+.1e}".format(x)}, 
                         linewidth=80)
    print(" Normalized residuals central 6 pixels")

    tlo, thi = (ff.nrm.residual.shape[0] // 2 - 3, ff.nrm.residual.shape[0] // 2 + 3)
    print((ff.nrm.residual / ff.datapeak)[tlo:thi, tlo:thi])
    print( " Normalized residuals max and min: {:.2e}, {:.2e}".format(
            ff.nrm.residual.max() / ff.datapeak, ff.nrm.residual.min() / ff.datapeak))
    utils.default_printoptions()


def raw_observables( fitsfn=None, fitsimdir=None, oitdir=None, oifdir=None, affine2d=None,
                     psf_offset_find_rotation=(0.0, 0.0), psf_offset_ff=None, 
                     rotsearch_d=None, set_pistons=None, oversample=3, mnem="", firstfew=None,
                     usebp=False, verbose=False,):
    """
    Reduce calibrated image data to raw fringe observables

    returns: affine2d (measured or input),
    psf_offset_find_rotation (input),
    psf_offset_ff (input or found),
    fringe pistons/r (found)
    """

    if verbose: print("raw_observables: input", os.path.join(fitsimdir, fitsfn))
    if verbose: print("raw_observables: oversample", oversample)

    fobj = fits.open(os.path.join(fitsimdir, fitsfn))

    if verbose: print(fobj[0].header["FILTER"])

    print("InstrumentData file:",  InstrumentData.__file__)
    niriss = InstrumentData.NIRISS( fobj[0].header["FILTER"], usebp=usebp, # bpexist=False,
                                    firstfew=firstfew,  # read_data truncation to only read first few slices...
                                    )

    ff = nrm_core.FringeFitter( niriss, oitdir=oitdir,  # write OI text files here, and diagnostic images if desired
                                        oifdir=oifdir,  # write OI fits files here
                                        weighted=True, oversample=oversample, 
                                        interactive=False, save_txt_only=False,)

    ff.fit_fringes(os.path.join(fitsimdir, fitsfn))
    examine_observables(ff)

    np.set_printoptions( formatter={"float": lambda x: "{:+.2e}".format(x)}, linewidth=80)
    if verbose: print("raw_observables: fringepistons/rad", *ff.nrm.fringepistons)
    utils.default_printoptions()
    fobj.close()
    return affine2d, psf_offset_find_rotation, ff.nrm.psf_offset, ff.nrm.fringepistons


def run_extraction( fitsimdir=None, oitdir=None, oifdir=None, ifn=None, oversample=3, mnem="",
                    firstfew=None, verbose=False, usebp=True,):
    """
    fitsimdir: string: dir containing data file
    ifn: str inout file name
    """

    np.set_printoptions(formatter={"float": lambda x: "{:+.2e}".format(x)}, linewidth=80)
    if verbose:
        print("extraction: ", ifn)
        print("extraction: fitsimdir", fitsimdir)

    aff, psf_offset_r, psf_offset_ff, fringepistons =  \
        raw_observables(fitsfn=ifn, fitsimdir=fitsimdir, oitdir=oitdir, oifdir=oifdir, 
        oversample=oversample, mnem=mnem, firstfew=firstfew, usebp=usebp, verbose=verbose)
    print( "aff", aff, "\npsf_offset_r", psf_offset_r, "\npsf_offset_ff", psf_offset_ff,
        "\nfringepistons", fringepistons, "\n",)

    del aff
    del psf_offset_r
    del psf_offset_ff
    del fringepistons


### For quick com results
def calibrate_nis019_pos1(fullpathoifdir):
    oi_abdor = "jw01093001001_01101_00005_nis.oifits"
    oi_37093 = "jw01093004001_01101_00005_nis.oifits"
    oi_36805 = "jw01093006001_01101_00005_nis.oifits"

    # Produce a single calibrated OIFITS file for each  pair
    print("************  Running calibrate ***************")

    oifdir = fullpathoifdir
    print(oifdir)
    cfnlist = [] # calibrated oifits file names list
    mnmlist = [] # calibrated oifits file mnemonics for candid output plot files

    cd, cfn = calibrate_oifits(os.path.join(oifdir,oi_abdor), os.path.join(oifdir,oi_37093), oifdir=oifdir, returnfilename=True)
    cfnlist.append(cfn)
    mnmlist.append('abdor_hd37093')

    cd, cfn = calibrate_oifits(os.path.join(oifdir,oi_abdor), os.path.join(oifdir,oi_36805), oifdir=oifdir, returnfilename=True)
    cfnlist.append(cfn)
    mnmlist.append('abdor_hd36805')

    cd, cfn = calibrate_oifits(os.path.join(oifdir,oi_37093), os.path.join(oifdir,oi_36805), oifdir=oifdir, returnfilename=True)
    cfnlist.append(cfn)
    mnmlist.append('hd37093_hd36805')

    cd, cfn = calibrate_oifits(os.path.join(oifdir,oi_36805), os.path.join(oifdir,oi_37093), oifdir=oifdir, returnfilename=True)
    cfnlist.append(cfn)
    mnmlist.append('hd36805_hd36805')

    print("\nAB Dor AC and two calibrators, POS1, pairwise calibration done")
    return cfnlist, mnmlist

### For implaneia develoopment
def calibrate_pair(fullpathoifdir):
    oi_tgt = "jw_tgt.oifits"
    oi_cal = "jw_cal.oifits"

    # Produce a single calibrated OIFITS file for each  pair
    print("************  Running calibrate ***************")

    oifdir = fullpathoifdir
    print(oifdir)
    cfnlist = [] # calibrated oifits file names list
    mnmlist = [] # calibrated oifits file mnemonics for candid output plot files

    cd, cfn = calibrate_oifits(os.path.join(oifdir,oi_tgt), os.path.join(oifdir,oi_cal), oifdir=oifdir, returnfilename=True)
    cfnlist.append(cfn)
    mnmlist.append('tgt_cal')

    print("\ntgt and cal calibration done")
    return cfnlist, mnmlist


def main():

    start = time.time()

    psr = ArgumentParser( description="Extraction of raw observables with ImPlaneIA",)
    psr.add_argument( "-d", "--datadir", type=str, default="pipeline_calibrated_data_nobadpix/",
        help="Directory of the input files (calints) calibrated by the JWST pipeline.\n"+\
        "Outputs are saved in subdirectories under this directory.",)
    psr.add_argument( "--firstfew", type=int, default=None,
        help="Analyse the first few frames. All frames are analyzed by default (firstfew=None).",)
    psr.add_argument( "-o", "--oversample", type=int, default=7,
        help="Model oversampling (also how fine to measure the centering).",)
    psr.add_argument( "-p", "--pattern", type=str, default="jw*calints.fits",
        help="Pattern match to find image files in input directory.",)
    psr.add_argument( "-s", "--silent", dest="verbose", action="store_false",
        help="Make the script less verbose.",)
    psr.add_argument( "-w", "--webbpsf-path", type=str, dest="webbpsf_path", default=None,
        help="Specify webbpsf path (environment variable WEBBPSF_PATH is used by default)",)
    args = psr.parse_args()

    datadir = args.datadir
    pattern=args.pattern
    firstfew = args.firstfew
    oversample = args.oversample
    verbose = args.verbose

    # Make sure WEBBPSF_PATH is set
    if os.environ.get("WEBBPSF_PATH", None) is None:
        if args.webbpsf_path is not None: os.environ["WEBBPSF_PATH"] = args.webbpsf_path
        else: raise TypeError( "Environment variable WEBBPSF_PATH or CLI arg -w/--webbpsf-path should be set")
    elif args.webbpsf_path is not None:
        warnings.warn( "Environment variable WEBBPSF_PATH exists, but will be overriden by CLI arg -w/--webbpsf-path",
            RuntimeWarning,)
        os.environ["WEBBPSF_PATH"] = args.webbpsf_path
    if verbose: print("WEBBPSF_PATH set to ", os.environ.get("WEBBPSF_PATH"))

    # Run ImPlaneIA to reduce calibrated images to raw fringe observables
    calintfiles = sorted((glob.glob(os.path.join(datadir, pattern))))
    calintfiles = [os.path.basename(f) for f in calintfiles]
    print(calintfiles)
    if verbose: print("FIRSTFEW", firstfew, "OVERSAMPLE", oversample)

    COUNT = 0
    for fnmir in calintfiles:
        print("\nAnalyzing\n   ", COUNT, fnmir.replace(".fits", ""), end=" ")
        hdr = fits.getheader(os.path.join(datadir, fnmir))
        print(hdr["FILTER"], end=" ")
        print(hdr["TARGNAME"], end=" ")
        print(hdr["TARGPROP"])
        # next line for convenient use in oifits writer which looks up target online
        if  "-" in hdr["TARGPROP"]:
            catname = hdr["TARGPROP"].replace( "-", " ") # for target lookup on-line, otherwise UNKNOWN used
            fits.setval(os.path.join(datadir, fnmir), "TARGNAME", value=catname)
            fits.setval(os.path.join(datadir, fnmir), "TARGPROP", value=catname)

        # Directory only here... not absolute paths
        oitdir = f"Saveoit_ov{oversample:d}"
        oifdir = f"Saveoif_ov{oversample:d}"

        run_extraction( fitsimdir=datadir, 
                        oitdir=os.path.join(datadir, oitdir),
                        oifdir=os.path.join(datadir, oifdir),
                        ifn=fnmir, oversample=oversample, mnem="", firstfew=firstfew,
                        usebp=True, verbose=verbose,) 

        # List a sample set of output products
        print( "======  Sanity check:: integration 0 of observation", fnmir[9], "exposure 00005 (1-4 are TA)  ======\n")
        results_int0 = sorted( glob.glob(os.path.join(datadir, oitdir, 
                          fnmir.replace(".fits",""), "*00*")))
        COUNT += 1

    print(*results_int0, sep="\n")

    observables_info = """
    Information about observables calculated from the 1st integration
    - phases_00.txt: 35 fringe phases
    - amplitudes_00.txt: 21 fringe amplitudes
    - CPs_00.txt: 35 closure phases
    - CAs_00.txt: 35 closure amplitudes
    - fringepistons_00.txt: 7 pistons (optical path delays between mask holes)
    - solutions_00.txt: 44 fringe coefficients of terms in the analytical model
    - modelsolution_00.fits: analytical model
    - n_modelsolution_00.fits: normalized analytical model
    - residual_00.fits: data - model
    - n_residual_00.fits: normalized residual
    """

    print(observables_info)

    oifiles = sorted(glob.glob(os.path.join(datadir, oifdir, "jw01093*oifits")))
    print("OUTPUT Uncalibrated OIFITS files:", *oifiles, sep="\n")

    # Raw OIFITS v2 files are created from each exposure, in odir/saveoif.
    end = time.time()
    print("RUNTIME for observables' extraction: %.2f s" % (end - start))

    print("\nNext, we calibrate the raw oifits observables... get back list of fullpath calibrated oifits files")
    
    cfnlist, mnmlist = calibrate_nis019_pos1(os.path.join(datadir,oifdir))
    print("Calibrated oifits files to be processed for binary search are:")
    for ii in range(len(cfnlist)):
        cfn, mnm = cfnlist[ii], mnmlist[ii]
        print('\t', cfn.split('/')[-1])
        candid_binary_extraction(cfn, mnm)


def candid_binary_extraction(calib_oifits, mnemonic):
    """
    calib_oifits: full path file name of [calibrated, usually] oifits file
    This directory is where subdirectory of results are written (text, plots)
    using the oifits file name root
    mnemonic: short string to name output file plots - usually with eg HD #s or names of target_calibrator
    """


    
    outputfile = os.path.dirname(calib_oifits) + '/' + mnemonic

    # ***
    # These are the binary parameters we expect CANDID to extract
    sep = 363.04 # binary separation [mas]
    theta = 285.398112 # position angle (pa) [deg]
    dm = 4.2  # delta magnitude [mag]

    fits.info(calib_oifits)

    fits.getheader(calib_oifits)
    # Your input data is an oifits file
    with fits.open(calib_oifits) as hdu:
        cp_ext = hdu['OI_T3'].data
        sqvis_ext = hdu['OI_VIS2'].data
        oiarray = hdu['OI_ARRAY'].data
        wavel = hdu['OI_WAVELENGTH'].data['EFF_WAVE']
        pscale = hdu['OI_ARRAY'].header['PSCALE']
        pav3 = hdu[0].header['PA']
    print('Wavelength: %.2e m' % wavel) 
    print('V3 PA: %.2f degrees' % pav3)
    cp = cp_ext['T3PHI']
    cp_err = cp_ext['T3PHIERR']
    tri_idx = cp_ext['STA_INDEX']

    sqvis = sqvis_ext['VIS2DATA']
    sqvis_err = sqvis_ext['VIS2ERR']
    bl_idx = sqvis_ext['STA_INDEX']

    hole_ctrs = oiarray['STAXYZ']
    hole_idx = oiarray['STA_INDEX']

    # Calculate the length of the baseline [m] for each pair
    baselines = []
    for bl in bl_idx:
        hole1,hole2 = (bl[0] - 1), (bl[1] - 1) # because hole numbers start at 1
        x1, y1 = hole_ctrs[hole1][0], hole_ctrs[hole1][1] 
        x2, y2 = hole_ctrs[hole2][0], hole_ctrs[hole2][1] 
        length = np.abs(np.sqrt((x2 - x1)**2. + (y2 - y1)**2.))
        baselines.append(length)
    # Calculate the length of three baselines for each triangle
    # Select the longest for plotting
    tri_baselines = []
    tri_longest = []
    for tri in tri_idx:
        hole1, hole2, hole3 = tri[0] - 1, tri[1] - 1, tri[2] - 1
        x1, y1 = hole_ctrs[hole1][0], hole_ctrs[hole1][1] 
        x2, y2 = hole_ctrs[hole2][0], hole_ctrs[hole2][1] 
        x3, y3 = hole_ctrs[hole3][0], hole_ctrs[hole3][1] 
        length12 = np.abs(np.sqrt((x2 - x1)**2. + (y2 - y1)**2.))
        length23 = np.abs(np.sqrt((x3 - x2)**2. + (y3 - y2)**2.))
        length31 = np.abs(np.sqrt((x1 - x3)**2. + (y1 - y3)**2.))
        tri_lengths = [length12,length23,length31]
        tri_baselines.append(tri_lengths)
        tri_longest.append(np.max(tri_lengths))
        
    # Calculate B_max/lambda
    bmaxlambda_sqvis = baselines / wavel
    bmaxlambda_cp = tri_longest / wavel

    # Label baselines and triangles
    bl_strings = []
    for idx in bl_idx:
        bl_strings.append(str(idx[0])+'_'+str(idx[1]))

    tri_strings = []
    for idx in tri_idx:
        tri_strings.append(str(idx[0])+'_'+str(idx[1])+'_'+str(idx[2]))

    print(sorted(baselines))


    # Plot closure phases, square visibilities
    # Label which point corresponds to which hole pair or triple

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,7))
    ax1.errorbar(bmaxlambda_cp, cp, yerr=cp_err, fmt='go')
    ax1.set_xlabel(r'$B_{max}/\lambda$',size=16)
    ax1.set_ylabel('Closure phase [deg]',size=14)
    ax1.set_title('Calibrated Closure Phase',size=14)
    for ii, tri in enumerate(tri_strings):
        ax1.annotate(tri, (bmaxlambda_cp[ii], cp[ii]), xytext=(bmaxlambda_cp[ii]+10000, cp[ii]))
        
    ax2.errorbar(bmaxlambda_sqvis, sqvis, yerr=sqvis_err, fmt='go')
    ax2.set_title('Calibrated Squared Visibility',size=16)
    ax2.set_xlabel(r'$B_{max}/\lambda$',size=14)
    ax2.set_ylabel('Squared visibility amplitude',size=14)
    for ii, bl in enumerate(bl_strings):
        ax2.annotate(bl, (bmaxlambda_sqvis[ii], sqvis[ii]), xytext=(bmaxlambda_sqvis[ii]+10000, sqvis[ii]))
    plt.savefig(outputfile+"_cp_sqv.png")

    # The above plots show the calibrated closure phases (left) and the
    # calibrated squared visibilities (right). Each quantity is plotted against
    # $B_{max}/\lambda$, the baseline length divided by the wavelength of the
    # observation. In the case of closure phases, where the triangle is formed
    # by three baselines, the longest one is selected. 
    # 
    # For a monochromatic observation of a point source, we would expect all 35
    # closure phases to be zero, and all 21 squared visibilities to be unity.
    # Asymmetries in the target caused by, e.g., an unresolved companion, cause
    # the closure phases and visibilities corresponding to the baselines
    # between affected sub-apertures to diverge from zero or unity. We can now
    # use the set of calibrated observables to model the most probable location
    # and contrast ratio of the companion. 

    # You can also use the dedicated tool from AMICAL to plot the data:

    plot = amical.show(calib_oifits, cmax=30)
    plt.savefig(outputfile+'_amical_show.png')

    bmax = 5.28 * u.meter
    wavel = 4.8e-6 * u.meter
    maxstep = wavel/(4*bmax) * u.rad
    stepsize = int(maxstep.to(u.mas)/u.mas)
    print('Using a step size of %i mas' % stepsize)

    param_candid = {'rmin': 10,  # inner radius of the grid
                    'rmax': 500,  # outer radius of the grid
                    'step': stepsize,  # grid sampling
                    'ncore': multiprocessing.cpu_count()  # core for multiprocessing
                    }
    # Perform the fit
    fit1 = amical.candid_grid(calib_oifits, **param_candid, diam=0, doNotFit=['diam*'], save=True, outputfile=outputfile)

    # plot the fitted model on data with residuals
    mod_v2, mod_cp, chi2 = amical.plot_model(calib_oifits, fit1['best'], save=True, outputfile=outputfile)


    # In the above output, CANDID provides a best-fit angular size for the
    # target star, 'best fit diameter (UD)', and the $\chi^2$ and n$\sigma$
    # (capped at 50$\sigma$) of the detection. It gives us estimates for the
    # binary separation ('sep'), position angle ('theta'), contrast ratio
    # ('CR'), and delta magnitudes ('dm'). 
    # 
    # It also produces plots of the squared visibilities and closure phases,
    # and plots the residual (difference between the data and the best-fit
    # model for each observable).
    # 
    # We can now compare these with our expected values from above:

    sep_fit, sep_unc = fit1['best']['sep'], fit1['uncer']['sep']
    theta_fit, theta_unc = fit1['best']['theta'], fit1['uncer']['theta']
    dm_fit, dm_unc = fit1['best']['dm'], fit1['uncer']['dm']

    print('             Expected      Model')
    print('Sep [mas]:   %.3f      %.3f +/- %.2f' % (sep, sep_fit, sep_unc))
    print('Theta [deg]: %.3f      %.3f +/- %.2f' % (theta, theta_fit, theta_unc))
    print('dm [mag]:    %.3f        %.3f +/- %.2f' % (dm, dm_fit, dm_unc))


    # Next, we will use CANDID to find the detection limit at different angular
    # separations. To do this, CANDID injects an additional companion at each
    # grid position with different flux ratios and estimates the number of
    # sigma for a theoretical detection at that point. It interpolates the flux
    # ratio values at 3$\sigma$ for all points in the grid to produce a
    # 3$\sigma$ detection map of the contrast (flux) ratio. 

    #  Find detection limits using the fake injection method
    cr_candid = amical.candid_cr_limit(
                calib_oifits, **param_candid, fitComp=fit1['comp'], save=True, outputfile=outputfile)


    # The first plot above shows the detection limit, in terms of contrast
    # ($\Delta$Mag), at each location in the search grid based on the
    # injection/detection of false companions. The second plot show an estimate
    # of the same detection limit with respect to the angular separation [mas]
    # from the primary target. For a complete description of the CANDID
    # algorithm, see [Galenne et al.
    # 2015](https://ui.adsabs.harvard.edu/link_gateway/2015A&A...579A..68G/doi:10.1051/0004-6361/201525917).

    # ### Visually compare the position
    # 
    # We can now look at an image with the faint companion artificially
    # brightened, and we see that the position of the primary star at the
    # center and its faint companion appear to match the position of the
    # companion detected on the above $\chi^2$ and $n\sigma$ maps output by
    # CANDID.
    ########################


if __name__ == "__main__":
    """ Developing use-bad-pixels-in-fitting-fringes call eg:
        python run_extract_calibrate_binaryfit_pos1.py -d /Users/anand/data/nis_019/implaneiadev/ -o 3  --firstfew 10
        python run_extract_calibrate_binaryfit_pos1.py -d /Users/anand/data/nis_019/implaneiadev/ -o 1  --firstfew 10
    Testing on noiseless:
    python run_extract_calibrate_binaryfit_pos1.py \
    -d /Users/anand/data/nis_019/implaneiadev/lower_contrast/pa0_sep200_con0.01_F380M_sky_81px_x11__F380M_81_flat_x11_noiseless_00_mir/  \
    -o 1  --firstfew 10
    Files: jw_cal_calints.fits and jw_tgt_calints.fits
    """
    main()
