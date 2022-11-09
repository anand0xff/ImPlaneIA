#! /usr/bin/env  python 
# Mathematica nb from Alex & Laurent
# anand@stsci.edu major reorg as LG++ 2018 01
# python3 required (int( (len(coeffs) -1)/2 )) because of  float int/int result change from python2

import numpy as np
import scipy.special
import numpy.linalg as linalg
import sys
from scipy.special import comb
import os, pickle
from uncertainties import unumpy  # pip install if you need

m = 1.0
mm = 1.0e-3 * m
um = 1.0e-6 * m


def scaling(img, photons):  # RENAME this function
    # img gives a perfect psf to count its total flux
    # photons is the desired number of photons (total flux in data)
    total = np.sum(img)
    print("total", total)
    return photons / total

    
def matrix_operations(img, model, flux = None, verbose=False, linfit=False, dqm=None):
    # meta-question: why & when do we use linfit?
    # least squares matrix operations to solve A x = b, where A is the model,
    # b is the data (image), and x is the coefficient vector we are solving for.
    # In 2-D data x = inv(At.A).(At.b)
    #
    # img 2d array of image data
    # dqm 2d bool array of bad pixel locations (same shape as 2d img), or  None (for all-good data)

    print("leastsqnrm.matrix_operations() - equally-weighted")

    flatimg = img.reshape(np.shape(img)[0] * np.shape(img)[1])
    flatdqm = dqm.reshape(np.shape(img)[0] * np.shape(img)[1])
    if verbose: 
        print(f'fringefitting.leastsqnrm.matrix_operations(): ', end='')
        print(f'\n\timg {img.shape:} \n\tdqm {dqm.shape:}', end='')
        print(f'\n\tL x W = {img.shape[0]:d} x {img.shape[1]:d} = {img.shape[0] * img.shape[1]:d}', end='')
        print(f'\n\tflatimg {flatimg.shape:}', end='')
        print(f'\n\tflatdqm {flatdqm.shape:}', end='')

    # Originally Alex had  nans coding bad pixels in the image.
    # Anand: re-use the nan terminology code but driven by bad pixel frame
    #        nanlist shoud get renamed eg donotuselist

    if verbose: print('\n\ttype(dqm)', type(dqm), end='')
    if dqm is not None: nanlist = np.where(flatdqm==True)  # where DO_NOT_USE up.
    else: nanlist = (np.array(()), ) # shouldn't occur w/MAST JWST data

    if verbose: 
        print(f'\n\ttype(nanlist) {type(nanlist):}, len={len(nanlist):}', end='')
        print(f'\n\tnumber of nanlist pixels: {len(nanlist[0]):d} items', end='') 
        print(f'\n\t{len(nanlist[0]):d} DO_NOT_USE pixels found in data slice',
          end='')
    else:
        print(f'\t{len(nanlist[0]):d} DO_NOT_USE pixels found in data slice')
          

    flatimg = np.delete(flatimg, nanlist)

    if verbose: print(f'\n\tflatimg {flatimg.shape:} after deleting {len(nanlist[0]):d}',
          end='')


    if flux is not None:
        flatimg = flux * flatimg / flatimg.sum()

    # A
    flatmodel_nan = model.reshape(np.shape(model)[0] * np.shape(model)[1], 
                                  np.shape(model)[2])
    flatmodel = np.zeros((len(flatimg), np.shape(model)[2]))
    if verbose: 
        print(f'\n\tflatmodel_nan {flatmodel_nan.shape:}', end='')
        print(f'\n\tflatmodel     {flatmodel.shape:}', end='')
        print(f'\n\tdifference    {flatmodel_nan.shape[0] - flatmodel.shape[0]:}', end='')
        print()
        print("flat model dimensions ", np.shape(flatmodel))
        print("flat image dimensions ", np.shape(flatimg))

    for fringe in range(np.shape(model)[2]):
        flatmodel[:,fringe] = np.delete(flatmodel_nan[:,fringe], nanlist)
    # At (A transpose)
    flatmodeltransp = flatmodel.transpose()
    # At.A (makes square matrix)
    modelproduct = np.dot(flatmodeltransp, flatmodel)
    # At.b
    data_vector = np.dot(flatmodeltransp, flatimg)
    # inv(At.A)
    inverse = linalg.inv(modelproduct)
    cond = np.linalg.cond(inverse)

    x = np.dot(inverse, data_vector)
    res = np.dot(flatmodel, x) - flatimg

    # put bad pixels back
    naninsert = nanlist[0] - np.arange(len(nanlist[0]))
    # calculate residuals with fixed but unused bad pixels as nans
    res = np.insert(res, naninsert, np.nan)
    res = res.reshape(img.shape[0], img.shape[1])

    if verbose:
        print('model flux', flux)
        print('data flux', flatimg.sum())
        print("flat model dimensions ", np.shape(flatmodel))
        print("model transpose dimensions ", np.shape(flatmodeltransp))
        print("flat image dimensions ", np.shape(flatimg))
        print("transpose * image data dimensions", np.shape(data_vector))
        print("flat img * transpose dimensions", np.shape(inverse))

    if linfit:
        try:
            from linearfit import linearfit

            # dependent variables
            M = np.mat(flatimg)

            # photon noise
            noise = np.sqrt(np.abs(flatimg))

            # this sets the weights of pixels fulfilling condition to zero
            weights = np.where(np.abs(flatimg)<=1.0, 0.0, 1.0/(noise**2))

            # uniform weight
            wy = weights
            S = np.mat(np.diag(wy));
            # matrix of independent variables
            C = np.mat(flatmodeltransp)

            # initialize object
            result = linearfit.LinearFit(M,S,C)

            # do the fit
            result.fit()

            # delete inverse_covariance_matrix to reduce size of pickled file
            result.inverse_covariance_matrix = []

            linfit_result = result
            print("Returned linearfit result")

        except ImportError:
            linfit_result = None
    #         if verbose:
            print("linearfit module not imported, no covariances saved.")
    else:
        linfit_result = None
        print("linearfit module not imported, no covariances saved.")

    return x, res, cond, linfit_result


#######################################################################

def weighted_operations(img, model,  verbose=False, dqm=None):
    # return x, res, condition_number (None=>no condition number yet), singvals
    #  x: solution vector
    #  res: residuals array, nan-flagged for bad dq values?
    #  cond: condition number not calculateds (no inversion done here, so not available)
    #  singvals: singular values returned by the SVD solution for the parameters
    #
    # meta-question: why & when do we use linfit?  I removed it here - anand 2022 Jan
    # least squares matrix operations to solve A x = b, where 
    #    A is the model, 
    #    b is the data (image), 
    #    x is the coefficient vector we are solving for. 
    #
    #    Solution 1:  equal weighting of data (matrix_operations()).
    #       x = inv(At.A).(At.b) 
    #
    #    Solution 2:  weighting data by Poisson variance (weighted_operations())
    #       x = inv(At.W.A).(At.W.b) 
    #       where W is a diagonal matrix of weights w_i, 
    #       weighting each data point i by the inverse of its variance:
    #          w_i = 1 / sigma_i^2
    #       For photon noise, the data, i.e. the image values b_i  have variance 
    #       proportional to b_i with an e.g. ADU to electrons coonversion factor.
    #       If this factor is the same for all pixels, we do not need to include
    #       it here (is that really true? Yes I think so because we're not
    #       normalizing wts here, just ascribing rel wts.).
    #
    # Possibly replace or campare with a MAD minimization using fast simplex
    # https://theoryl1.wordpress.com/2016/08/03/solve-weighted-least-squares-with-numpy/
    # Solve for x in Ax = b
    #
    # np.set_printoptions(formatter={'float': lambda x: '{:+.1e}'.format(x)}, linewidth=80)
    #
    # Ax = b
    # b: data vector nd long; nd=5
    # A: model matrix; np x nd matrix 4 x 5:  np=4 parameters, nd=5 data points.
    # x: parameter, vector np=4 long, unknown
    #
    # A=np.array([[3,1,4,2],[2,7,1,2],[1,6,1,8],[6,1,8,2],[1,4,1,4]])
    # print("A:", A.shape)
    # b = np.array([1.2,1.3,1.4,1.5,1.6])
    # print("b:", b.shape)
    # w = np.array([1,2,3,4,5])
    # print("w:", w.shape)
    # Aw = A * np.sqrt(w[:,np.newaxis])
    # print("Aw:", Aw.shape)
    # bw = w * np.sqrt(w)
    # x, r, rank, s = np.linalg.lstsq(Aw, bw, rcond=None)
    # print("x.shape:", x.shape)
    # print("x:", x)
    # print("r:", r)
    # print("rank:", rank)
    # print("s:", s)

    # Also a good summary at:
    #    https://math.stackexchange.com/questions/3094925/weighted-least-squares

    # Remove not-to-be-fit data from the flattened "img" data vector
    flatimg = img.reshape(np.shape(img)[0] * np.shape(img)[1])
    flatdqm = dqm.reshape(np.shape(img)[0] * np.shape(img)[1])

    if dqm is not None: nanlist = np.where(flatdqm==True)  # where DO_NOT_USE up.
    else: nanlist = (np.array(()), ) # shouldn't occur w/MAST JWST data

    # see original linearfit https://github.com/agreenbaum/ImPlaneIA: 
    # agreenbaum committed on May 21, 2017 1 parent 3e0fb8b
    # commit bf02eb52c5813cb5d77036174a7caba703f9d366
    # 
    flatimg = np.delete(flatimg, nanlist)  # DATA values

    # photon noise variance - proportional to ADU 
    # (for roughly uniform adu2electron factor)
    variance = np.abs(flatimg)  
    # this resets the weights of pixels with negative or unity values to zero
    # we ignore data with unity or lower values - weight it not-at-all..
    weights = np.where(flatimg <= 1.0, 0.0, 1.0/np.sqrt(variance))  # anand 2022 Jan

    print("fringefitting.leastsqnrm.weighted_operations:", len(nanlist[0]),
          "bad pixels skipped in weighted fringefitter")

    # A - but delete all pixels flagged by dq array
    flatmodel_nan = model.reshape(np.shape(model)[0] * np.shape(model)[1], 
                                  np.shape(model)[2])
    flatmodel = np.zeros((len(flatimg), np.shape(model)[2]))
    for fringe in range(np.shape(model)[2]):
        flatmodel[:,fringe] = np.delete(flatmodel_nan[:,fringe], nanlist)
    print(flatmodel.shape)

    # A.w 
    # Aw = A * np.sqrt(w[:,np.newaxis]) # w as a column vector
    Aw = flatmodel * weights[:,np.newaxis]
    # bw = b * np.sqrt(w)
    bw = flatimg * weights
    # x = np.linalg.lstsq(Aw, bw)[0]
    # resids are pixel value residuals, flattened to 1d vector
    x, rss, rank, singvals = np.linalg.lstsq(Aw, bw)

    #inverse = linalg.inv(Atww)
    #cond = np.linalg.cond(inverse)

    # actual residuals in image:  is this sign convention odd?
    # res = np.dot(flatmodel, x) - flatimg
    # changed here to data - model
    res = flatimg - np.dot(flatmodel, x) 

    # put bad pixels back
    naninsert = nanlist[0] - np.arange(len(nanlist[0]))
    # calculate residuals with fixed but unused bad pixels as nans
    res = np.insert(res, naninsert, np.nan)
    res = res.reshape(img.shape[0], img.shape[1])

    cond = None
    return x, res, cond, singvals # no condition number yet...


def deltapistons(pistons):
    # This function is used for comparison to calculate relative pistons from given pistons (only deltapistons are measured in the fit)
    N = len(pistons)
    # same alist as above to label holes
    alist = []
    for i in range(N - 1):
        for j in range(N - 1):
            if j + i + 1 < N:
                alist = np.append(alist, i)
                alist = np.append(alist, j + i + 1)
    alist = alist.reshape(len(alist)/2, 2)
    delta = np.zeros(len(alist))
    for q,r in enumerate(alist):
        delta[q] = pistons[r[0]] - pistons[r[1]]
    return delta


def tan2visibilities(coeffs, verbose=False):
    """
    Technically the fit measures phase AND amplitude, so to retrieve
    the phase we need to consider both sin and cos terms. Consider one fringe:
    A { cos(kx)cos(dphi) +  sin(kx)sin(dphi) } = 
    A(a cos(kx) + b sin(kx)), where a = cos(dphi) and b = sin(dphi)
    and A is the fringe amplitude, therefore coupling a and b
    In practice we measure A*a and A*b from the coefficients, so:
    Ab/Aa = b/a = tan(dphi)
    call a' = A*a and b' = A*b (we actually measure a', b')
    (A*sin(dphi))^2 + (A*cos(dphi)^2) = A^2 = a'^2 + b'^2

    Edit 10/2014: pistons now returned in units of radians!!
    Edit 05/2017: J. Sahlmann added support of uncertainty propagation
    """
    if type(coeffs[0]).__module__ != 'uncertainties.core':
        # if uncertainties not present, proceed as usual
        
        # coefficients of sine terms mulitiplied by 2*pi

        delta = np.zeros(int( (len(coeffs) -1)/2 ))  # py3
        amp = np.zeros(int( (len(coeffs) -1)/2 ))  # py3
        for q in range(int( (len(coeffs) -1)/2 )):  # py3
            delta[q] = (np.arctan2(coeffs[2*q+2], coeffs[2*q+1])) 
            amp[q] = np.sqrt(coeffs[2*q+2]**2 + coeffs[2*q+1]**2)
        if verbose:
            print("shape coeffs", np.shape(coeffs))
            print("shape delta", np.shape(delta))

        # returns fringe amplitude & phase
        return amp, delta
    
    else:
        #         propagate uncertainties
        qrange = np.arange(int( (len(coeffs) -1)/2 ))  # py3
        fringephase = unumpy.arctan2(coeffs[2*qrange+2], coeffs[2*qrange+1])
        fringeamp = unumpy.sqrt(coeffs[2*qrange+2]**2 + coeffs[2*qrange+1]**2)
        return fringeamp, fringephase


def fixeddeltapistons(coeffs, verbose=False):
    delta = np.zeros(int( (len(coeffs) -1)/2 ))  # py3
    for q in range(int( (len(coeffs) -1)/2 )):  # py3
        delta[q] = np.arcsin((coeffs[2*q+1] + coeffs[2*q+2]) / 2) / (np.pi*2.0)
    if verbose:
        print("shape coeffs", np.shape(coeffs))
        print("shape delta", np.shape(delta))

    return delta    


def populate_antisymmphasearray(deltaps, N=7):
    if type(deltaps[0]).__module__ != 'uncertainties.core':
        fringephasearray = np.zeros((N,N))
    else:
        fringephasearray = unumpy.uarray(np.zeros((N,N)),np.zeros((N,N)))    
    step=0
    n=N-1
    for h in range(n):
        """
        fringephasearray[0,q+1:] = coeffs[0:6]
        fringephasearray[1,q+2:] = coeffs[6:11]
        fringephasearray[2,q+3:] = coeffs[11:15]
        fringephasearray[3,q+4:] = coeffs[15:18]
        fringephasearray[4,q+5:] = coeffs[18:20]
        fringephasearray[5,q+6:] = coeffs[20:]
        """
        fringephasearray[h, h+1:] = deltaps[step:step+n]
        step= step+n
        n=n-1
    fringephasearray = fringephasearray - fringephasearray.T
    return fringephasearray


def populate_symmamparray(amps, N=7):

    if type(amps[0]).__module__ != 'uncertainties.core':
        fringeamparray = np.zeros((N,N))
    else:
        fringeamparray = unumpy.uarray(np.zeros((N,N)),np.zeros((N,N)))
        
    step=0
    n=N-1
    for h in range(n):
        fringeamparray[h,h+1:] = amps[step:step+n]
        step = step+n
        n=n-1
    fringeamparray = fringeamparray + fringeamparray.T
    return fringeamparray


def phases_and_amplitudes(solution_coefficients, N=7):

    #     number of solution coefficients
    Nsoln = len(solution_coefficients)    
    
    # normalise by intensity
    soln = np.array([solution_coefficients[i]/solution_coefficients[0] for i in range(Nsoln)])

    # compute fringe quantitites
    fringeamp, fringephase = tan2visibilities( soln )    
    
#     import pdb
#     pdb.set_trace()

    # compute closure phases
    if type(solution_coefficients[0]).__module__ != 'uncertainties.core':
        redundant_closure_phases = redundant_cps(np.array(fringephase), N=N)
    else:
        redundant_closure_phases, fringephasearray = redundant_cps(np.array(fringephase), N=N)
    
    # compute closure amplitudes
    redundant_closure_amplitudes = return_CAs(np.array(fringephase), N=N)

    return fringephase, fringeamp, redundant_closure_phases, redundant_closure_amplitudes


def redundant_cps(deltaps, N = 7):
    fringephasearray = populate_antisymmphasearray(deltaps, N=N)
    if type(deltaps[0]).__module__ != 'uncertainties.core':
        cps = np.zeros(int(comb(N,3)))
    else:
        cps = unumpy.uarray( np.zeros(np.int(comb(N,3))),np.zeros(np.int(comb(N,3))) )    
    nn=0
    for kk in range(N-2):
        for ii in range(N-kk-2):
            for jj in range(N-kk-ii-2):
                cps[nn+jj] = fringephasearray[kk, ii+kk+1] \
                       + fringephasearray[ii+kk+1, jj+ii+kk+2] \
                       + fringephasearray[jj+ii+kk+2, kk]
            nn = nn+jj+1
    if type(deltaps[0]).__module__ != 'uncertainties.core':
        return cps
    else:
        return cps, fringephasearray

        
def closurephase(deltap, N=7):
    # N is number of holes in the mask
    # 7 and 10 holes available (JWST & GPI)

    # p is a triangular matrix set up to calculate closure phases
    if N == 7:
        p = np.array( [ deltap[:6], deltap[6:11], deltap[11:15], \
                deltap[15:18], deltap[18:20], deltap[20:] ] )
    elif N == 10:
        p = np.array( [ deltap[:9], deltap[9:17], deltap[17:24], \
                deltap[24:30], deltap[30:35], deltap[35:39], \
                deltap[39:42], deltap[42:44], deltap[44:] ] )
        
    else:
        print("invalid hole number")

    # calculates closure phases for general N-hole mask (with p-array set up properly above)
    cps = np.zeros(int((N - 1)*(N - 2)/2)) #py3
    for l1 in range(N - 2):
        for l2 in range(N - 2 - l1):
            cps[int(l1*((N + (N-3) -l1) / 2.0)) + l2] = \
                p[l1][0] + p[l1+1][l2] - p[l1][l2+1]
    return cps


def return_CAs(amps, N=7):
    fringeamparray = populate_symmamparray(amps, N=N)            
    nn=0
    
    if type(amps[0]).__module__ != 'uncertainties.core':
        CAs = np.zeros(int(comb(N,4)))
    else:
        CAs = unumpy.uarray( np.zeros(np.int(comb(N,4))),np.zeros(np.int(comb(N,4))) )
        
    for ii in range(N-3):
        for jj in range(N-ii-3):
            for kk in range(N-jj-ii-3):
                for ll  in range(N-jj-ii-kk-3):
                    CAs[nn+ll] = fringeamparray[ii,jj+ii+1] \
                               * fringeamparray[ll+ii+jj+kk+3,kk+jj+ii+2] \
            / (fringeamparray[ii,kk+ii+jj+2]*fringeamparray[jj+ii+1,ll+ii+jj+kk+3])
                nn=nn+ll+1
    return CAs
