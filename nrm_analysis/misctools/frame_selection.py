#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from scipy import stats
import copy
from astropy.stats import sigma_clip


import nrm_analysis.misctools.oifits as oifits
from notebooks.observable_tools import ObservableSet


def frame_select(calintsfn, nsigma=1, save_mtfs=True):
    """
    Takes a calints file and performs frame selection based on the FT of each integration.
    Integrations where the sum of the central 9 pixels of the MTF is more than nsigma from the mean
    are discarded. Returns list of good indices.
    """
    with fits.open(calintsfn) as hdu:
        data = hdu['SCI'].data
    imsz = data.shape
    if len(imsz) != 3:
        raise Exception('Image must be 3d multi-integration (calints file)')
    # maxlist = []
    # for j in range(imsz[0]):
    #     maxlist += [np.unravel_index(np.argmax(data[j]), data[j].shape)] 
    # maxlist = np.array(maxlist)
    # xh = min(imsz[1] - stats.mode(maxlist[:, 0]).mode, stats.mode(maxlist[:, 0]).mode - 4) # the bottom 4 rows are reference pixels
    # yh = min(imsz[2] - stats.mode(maxlist[:, 1]).mode, stats.mode(maxlist[:, 1]).mode - 0)
    # sh = min(xh, yh)
    # dupes = np.unique(maxlist,axis=0,return_inverse=True)[1]
    # maxlist = maxlist[dupes]
    # peak = stats.mode(maxlist).mode
    # find center from median image to be insensitive to CR hits
    medimage = np.median(data,axis=0)
    peak = np.where(medimage==medimage.max())
    peak0,peak1 = peak[0],peak[1]
    sh = min((imsz[1]-peak1),(imsz[2]-peak0))
    print('      Cropping all frames to %.0fx%.0f pixels' % (2*sh+1, 2*sh+1))
    centered_data = data[:,int(peak0-sh):int(peak0+sh+1),int(peak1-sh):int(peak1+sh+1)]
    # Code adapted from Joel SB's SAMpip
    mtf_ims = np.zeros_like(centered_data)
    peaks = np.zeros(imsz[0])
    for www in range(imsz[0]):
        im = np.abs(np.fft.fftshift(np.fft.ifft2(centered_data[www,:,:])))
        mtf_ims[www,:,:] = im
        # ind_peakx, ind_peaky = np.where(im == np.max(im))
        # assuming data has been properly centered, the mtf peak should be the center
        ind_peakx, ind_peaky = int(sh), int(sh)
        peaks[www] = np.sum(im[int(ind_peakx-1):int(ind_peakx+2), int(ind_peaky-1):int(ind_peaky+2)])
    [goodidxlist] = np.where((peaks >= np.mean(peaks)-np.std(peaks)*nsigma) & (peaks <= np.mean(peaks)+np.std(peaks)*nsigma))
    ngood = len(goodidxlist)
    nbad = imsz[0] - ngood
    perc_bad = nbad/imsz[0] * 100
    print('      %i/%i frames rejected with %.1f sigma threshold' %(nbad,imsz[0],nsigma))
    print('%i/%i integrations (%.2f%%) rejected for >%i-sigma outlier sum of central 3x3 FT pixels' % (nbad,imsz[0],perc_bad,nsigma))

    if save_mtfs==True:
        mtf_name = calintsfn.replace('.fits','_mtfs.fits')
        fits.writeto(mtf_name, mtf_ims, overwrite=True)
        print('MTFs saved to %s' % mtf_name)
    return goodidxlist


def median_difference(filename,nsigma=10,display=True):
    with fits.open(filename) as hduin:
        data = hduin['SCI'].data
    # check that data is 3d
    if len(data.shape) != 3:
        print('WARNING! DATA SHOULD BE IMAGE CUBE')
    median = np.median(data, axis=0)
    std = np.std(data,axis=0)
    # Make array for slice - median cube
    mediandiff = np.empty_like(data)
    mediandiff[:,:,:] = data - median
    # get locations with nsigma outliers from median
    outliers = np.argwhere(mediandiff > nsigma*std)
    badints = sorted(list(set(outliers[:,0])))
    outlier_dict = {}
    for bad in badints:
        pixels = []
        for out in outliers:
            if out[0] == bad:
                pixels.append([out[1],out[2]])
        outlier_dict[bad] = pixels
    print("Number of integrations with >%s-sigma outliers: %s" % (nsigma,len(badints)))
    for slc in outlier_dict:
        print('\tInt %i: %i pixels' % (slc,len(outlier_dict[slc])))
    if display & (len(badints) != 0):
        fig, axs = plt.subplots(len(badints),figsize=(5,5*len(badints)))
        for ii,slc in enumerate(outlier_dict):
            pixlist = np.array(outlier_dict[slc])
            maxs = np.array([max(pixlist[:,0])+3, max(pixlist[:,1])+3]) # xmax, ymax
            mins = np.array([min(pixlist[:,0])-2, min(pixlist[:,1])-2]) # xmin, ymin
            # adjust the edges for display
            toobig = np.where(np.array(maxs) > 80)
            toosmall = np.where(np.array(mins) < 0)
            maxs[toobig] = maxs[toobig] - 3
            mins[toosmall] = mins[toosmall] + 3
            xmin, xmax = mins[0], maxs[0]
            ymin, ymax =  mins[1], maxs[1]
            axs[ii].imshow(data[slc,xmin:xmax,ymin:ymax],origin='lower')
            axs[ii].set_title('Int %s: [%i:%i,%i:%i]' % (slc,xmin,xmin,ymin,ymax))
            axs[ii].axis('off')
        plt.show()
    bad_idx_list = list(outlier_dict.keys())
    print(bad_idx_list)
    allints = range(data.shape[0]) # this is nints
    goodidxlist = [idx for idx in allints if idx not in bad_idx_list]
    perc_bad = len(bad_idx_list)/len(allints) * 100
    print('%i/%i integrations (%.2f%%) rejected for >%i-sigma outlier cps and/or visamps' % (len(bad_idx_list),len(allints),perc_bad,nsigma))
    #return outlier_dict
    return goodidxlist

def clip_observables(oifitsfn, nsigma=4, plot=True):
    """
    Alternate form of frame selection: 
    do sigma-clipping on multi-integration observables. 
    Returns: a list of good indices to be given to clip_oifits
    """
    def get_badints(obsarray, nsigma, plot):
        badidxlist = []
        if plot:
            plt.figure(figsize=(25,25)) 
            nsubs = int(np.ceil(np.sqrt(len(obsarray))))
        n=0
        for oneobs in obsarray: # one observable 
            clipped_ma = sigma_clip(oneobs, sigma=nsigma, cenfunc='mean', maxiters=None, axis=None, masked=True)
            clipped = clipped_ma[~clipped_ma.mask]
            nclipped = len(obsarray) - len(clipped)
            badints = np.where(clipped_ma.mask == True)
            badidxlist.append(badints[0])
            if plot:
                ax = plt.subplot(nsubs, nsubs, n + 1)
                _, bins, patches = ax.hist(clipped,histtype='step',density=True,bins='auto',fill=True,alpha=.5)
                # plot gaussian (of full data,clipped data)
                mu, std = stats.norm.fit(clipped) 
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mu, std)
                labelstr = '%i' %  nsigma + r'$\sigma$ clipped:'+'\n'+ r'$\mu$ = %.2f' % mu + '\n'+'$\sigma$ = %.2f' % std
                plt.plot(x, p, ls='--',c='tab:blue', linewidth=2, label=labelstr)

                mu_unclip,std_unclip = stats.norm.fit(oneobs)
                badp = stats.norm.pdf(x, mu_unclip, std_unclip)
                labelstr2 = r'unclipped:'+'\n'+r'$\mu$ = %.2f' % mu_unclip + '\n'+'$\sigma$ = %.2f' % std_unclip
                plt.plot(x, badp, 'k--', linewidth=2, alpha=.25, label=labelstr2)
                
                str2 = '%i/%i ints' % (len(clipped),len(oneobs))
                ax.annotate(str2,xycoords='axes fraction',xy=(0.01,.95),size=10)
                ax.legend(fontsize=9,loc='upper right')
                ax.set_title(n+1)
                n+=1
            plt.tight_layout()
                
        badidxlist = sorted(list(set([bad for badidx in badidxlist for bad in badidx]))) 
        return badidxlist
    print(os.path.basename(oifitsfn))
    obsset = ObservableSet(oifitsfn)
    # clip based on both closure phases & visamps
    cps = obsset.oi.OI_T3.T3PHI
    visamps = obsset.oi.OI_VIS.VISAMP
    badints_cps = get_badints(cps, nsigma, plot)
    badints_visamps = get_badints(visamps, nsigma, plot)
    bothbadints = badints_cps + badints_visamps # join lists
    # remove duplicate ints and sort
    badidxlist = sorted(list(set(bothbadints))) 
    allints = range(len(cps[0])) # this is nints
    goodidxlist = [idx for idx in allints if idx not in badidxlist]
    perc_bad = len(badidxlist)/len(allints) * 100
    print('%i/%i integrations (%.2f%%) rejected for >%i-sigma outlier cps and/or visamps' % (len(badidxlist),len(allints),perc_bad,nsigma))
    # mean observable error bar reduction due to clipping
    old_cp_stds = obsset.stats.std_cps
    old_va_stds = obsset.stats.std_visamp
    new_cp_stds = np.std(cps[:,goodidxlist],axis=1)
    new_va_stds = np.std(visamps[:,goodidxlist],axis=1)
    
    cp_std_pchange = (new_cp_stds - old_cp_stds)/old_cp_stds * 100
    va_std_pchange = (new_va_stds - old_va_stds)/old_va_stds * 100
    print('\tMedian %% change in CP errorbars: %.2f%%' % np.median(cp_std_pchange))
    print('\tMedian %% change in visamp errorbars: %.2f%%' % np.median(va_std_pchange))
    #print(cp_std_pchange)
    print(badidxlist)
    return goodidxlist


def clip_oifits(oifitsfn, good_indices, method='med', suffix=''):
    """
    Takes an OIFITS filename and list of good integration indices and outputs
    updated OIFITS files using only those integrations.
    TO DO: save mtf peak sums, DC term, constant flux term somewhere in OIFITS header
    """
    indir, bn = os.path.split(oifitsfn)
    nrm_dct = oifits.load(oifitsfn)
    obsarr = nrm_dct['OI_VIS']['VISAMP'] # for checking observable array shape
    if suffix == '':
        suffix = 'trim'
    if (len(obsarr.shape)==1) | (obsarr.shape[1] == 1):
        raise Exception('Multi-integration oifits file expected (2d observable arrays)')
    print('Reading multi-integration OIFITS file...')
    # arrays to update
    namedict = {'OI_ARRAY':['PISTONS','PIST_ERR','PISTON_T','PISTON_C'],
                'OI_VIS':['VISAMP','VISAMPERR','VISPHI','VISPHIERR'],
                'OI_VIS2':['VIS2DATA','VIS2ERR'],
                'OI_T3':['T3AMP','T3AMPERR','T3PHI','T3PHIERR']}
    
    outdict_multi = copy.deepcopy(nrm_dct)
    for extname in namedict:
        for colname in namedict[extname]:
            try:
                #print(nrm_dct[extname][colname].shape)
                outarr = nrm_dct[extname][colname][:,good_indices]
                # print(extname, colname,nrm_dct[extname][colname].shape,'-->',outarr.shape)
                outdict_multi[extname][colname] = outarr
            except KeyError as e: # e.g. different PISTONS keywords present
                continue
    multi_outname = bn.replace('.oifits','_%s.oifits'%suffix)
    oifits.save(outdict_multi, filename=multi_outname, datadir=indir) # this saves the trimmed multi-oifits
    # save updated averaged oifits too
    outdict_avg = copy.deepcopy(outdict_multi)
    # default method in populate_NRM is median combination, apply that here too
    for extname in namedict:
        for colname in namedict[extname]:
            if 'ERR' in colname:
                # get the corresponding data column
                datacol = colname.replace('ERR','')
                if datacol == 'VIS2':
                    arr = outdict_multi[extname]['VIS2DATA']
                if datacol == 'PIST_': # handle different PISTONS keyword present, again
                    for eee in ['PISTONS','PISTON_T','PISTON_C']:
                        try:
                            arr = outdict_multi[extname][eee]
                        except KeyError:
                            continue
                outarr = np.std(arr, axis=1)
            else:
                try:
                    arr = outdict_multi[extname][colname]
                except KeyError:
                    continue
                if method=='med':
                    outarr = np.median(arr, axis=1)
                else:
                    outarr = np.mean(arr, axis=1)
            outdict_avg[extname][colname] = outarr

    avg_outname = bn.replace('multi_','').replace('.oifits','_%s.oifits'%suffix)
    oifits.save(outdict_avg,filename=avg_outname,datadir=indir)