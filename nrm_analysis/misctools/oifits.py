#! /usr/bin/env python

"""
@author: Anthony Soulain (University of Sydney), Rachel Cooper (STScI), Anand Sivaramakrishnan (STScI)


--------------------------------------------------------------------
implaneIA software
--------------------------------------------------------------------

OIFITS related function.

-------------------------------------------------------------------- 
"""

import datetime
import os
import copy
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.simbad import Simbad
from matplotlib import pyplot as plt
from termcolor import cprint

plt.close('all')


list_color = ['#00a7b5', '#afd1de', '#055c63', '#ce0058', '#8a8d8f', '#f1b2dc']

def rad2mas(rad):
    return rad / u.marcsec.to(u.rad)


def GetWavelength(ins, filt):
    """ Get wavelengths information from using instrument and filter informations."""
    dic_filt = {'JWST': {'F277W': [2.776, 0.715],
                         'F380M': [3.828, 0.205],
                         'F430M': [4.286, 0.202],
                         'F480M': [4.817, 0.298]
                         }
                }

    wl = dic_filt[ins][filt][0]*1e-6
    e_wl = dic_filt[ins][filt][1]*1e-6

    return wl, e_wl


def Format_STAINDEX_V2(tab):
    """ Converts sta_index to save oifits in the appropriate format."""
    sta_index = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        if np.min(tab) == 0:
            line = np.array([ap1, ap2]) + 1 # RAC 2/2021
        else:
            line = np.array([ap1, ap2])
        sta_index.append(line)
    return sta_index


def Format_STAINDEX_T3(tab):
    """ Converts sta_index to save oifits in the appropriate format."""
    sta_index = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        ap3 = int(x[2])
        if np.min(tab) == 0:
            line = np.array([ap1, ap2, ap3]) + 1
        else:
            line = np.array([ap1, ap2, ap3])
        sta_index.append(line)
    return sta_index


def ApplyFlag(data, unit='arcsec'):
    """ Apply flag and convert to the appropriate units."""

    wl = data['OI_WAVELENGTH']['EFF_WAVE']
    uv_scale = {'m': 1,
                'rad': 1/wl,
                'arcsec': 1/wl/rad2mas(1e-3),
                'lambda': 1/wl/1e6}

    U = data['OI_VIS2']['UCOORD']*uv_scale[unit]
    V = data['OI_VIS2']['VCOORD']*uv_scale[unit]

    flag_v2 = np.invert(data['OI_VIS2']['FLAG'])
    V2 = data['OI_VIS2']['VIS2DATA'][flag_v2]
    e_V2 = data['OI_VIS2']['VIS2ERR'][flag_v2] * 1
    sp_freq_vis = data['OI_VIS2']['BL'][flag_v2] * uv_scale[unit]
    flag_cp = np.invert(data['OI_T3']['FLAG'])
    cp = data['OI_T3']['T3PHI'][flag_cp]
    e_cp = data['OI_T3']['T3PHIERR'][flag_cp]
    sp_freq_cp = data['OI_T3']['BL'][flag_cp] * uv_scale[unit]
    bmax = 1.2*np.max(np.sqrt(U**2+V**2))

    return U, V, bmax, V2, e_V2, cp, e_cp, sp_freq_vis, sp_freq_cp, wl, data['info']['FILT']


def save(dic, filename=None, datadir=None, verbose=False):
    """
    Save dictionnary formatted data into a proper OIFITS (version 2) format file.

    Parameters:
    -----------

    `dic` {dict}:
        Dictionnary containing all extracted data (keys: 'OI_VIS2', 'OI_VIS', 'OI_T3', 'OI_WAVELENGTH', 'info'),\n
    `filename` {str}:
        By default None, the filename is constructed using informations included in the input dictionnary ('info'),\n
    """
    if dic is None:
        cprint('\nError save oifits : Wrong data format!', on_color='on_red')
        return None

    if datadir is None:
        datadir = 'Saveoifits/'
    if datadir[-1] != '/':
        datadir = datadir + '/'

    if not os.path.exists(datadir):
        print('### Create %s directory to save all requested Oifits ###' % datadir)
        os.system('mkdir %s' % datadir)
    
    if type(filename) != str:
        try:
            if len(dic['info']['MJD']) > 1:
                filename = '%s_%s_%s_%s_%s.oifits' % (dic['info']['TARGET'].replace(' ', ''),
                                                             dic['info']['INSTRUME'],
                                                             dic['info']['MASK'],
                                                             dic['info']['FILT'],
                                                             dic['info']['MJD'][0]) # if loaded from oifits it is a list
        except TypeError:
            filename = '%s_%s_%s_%s_%s.oifits' % (dic['info']['TARGET'].replace(' ', ''),
                                                     dic['info']['INSTRUME'],
                                                     dic['info']['MASK'],
                                                     dic['info']['FILT'],
                                                     dic['info']['MJD'])

    # ------------------------------
    #       Creation OIFITS
    # ------------------------------
    if verbose:
        print("\n\n### Init creation of OI_FITS (%s) :" % (filename))

    hdulist = fits.HDUList()
    hdu = fits.PrimaryHDU()
    hdu.header['DATE'] = datetime.datetime.now().strftime(
        format='%F')  # , 'Creation date'
    hdu.header['ORIGIN'] = 'STScI'
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS']
    hdu.header['CONTENT'] = 'OIFITS2'
    hdu.header['TELESCOP'] = dic['info']['TELESCOP']
    hdu.header['INSTRUME'] = dic['info']['INSTRUME']
    hdu.header['OBSERVER'] = dic['info']['OBSERVER']
    hdu.header['OBJECT'] = dic['info']['OBJECT']
    hdu.header['INSMODE'] = dic['info']['INSMODE']
    hdu.header['FILT'] = dic['info']['FILT']
    hdu.header['ARRNAME'] = dic['info']['ARRNAME'] # Anand 9/2020
    hdu.header['MASK'] = dic['info']['ARRNAME'] # Anand 9/2020
    hdu.header['PA'] = dic['info']['PA'] # RC 1/2021
    # name of calibrator if applicable. RC 1/2021
    try:
        hdu.header['CALIB'] = dic['info']['CALIB']
    except KeyError:
        pass

    hdulist.append(hdu)
    # ------------------------------
    #        OI Wavelength
    # ------------------------------

    if verbose:
        print('-> Including OI Wavelength table...')
    data = dic['OI_WAVELENGTH']


    col1 = fits.Column(name='EFF_WAVE', format='1E',
                    unit='METERS', array=[data['EFF_WAVE']])
    col2 = fits.Column(name='EFF_BAND', format='1E',
                    unit='METERS', array=[data['EFF_BAND']])

    coldefs = fits.ColDefs([col1,col2])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    # Header
    hdu.header['EXTNAME'] = 'OI_WAVELENGTH'
    hdu.header['OI_REVN'] = 2  # , 'Revision number of the table definition'
    # 'Name of detector, for cross-referencing'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdulist.append(hdu)  # Add current HDU to the final fits file.

    # ------------------------------
    #          OI Target
    # ------------------------------
    if verbose:
        print('-> Including OI Target table...')

    name_star = dic['info']['TARGET']

    customSimbad = Simbad()
    customSimbad.add_votable_fields('propermotions', 'sptype', 'parallax')

    # Add informations from Simbad:
    if name_star == 'UNKNOWN':
        ra, dec, spectyp = [0], [0], ['unknown']
        pmra, pmdec, plx = [0], [0], [0]
    else:
        try:
            query = customSimbad.query_object(name_star)
            coord = SkyCoord(query['RA'][0]+' '+query['DEC']
                             [0], unit=(u.hourangle, u.deg))
            ra, dec = [coord.ra.deg], [coord.dec.deg]
            spectyp, plx = query['SP_TYPE'], query['PLX_VALUE']
            pmra, pmdec = query['PMRA'], query['PMDEC']
        except TypeError:
            ra, dec, spectyp = [0], [0], ['unknown']
            pmra, pmdec, plx = [0], [0], [0]


    col1 = fits.Column(name='TARGET_ID', format='1I', array=[1])
    col2 = fits.Column(name='TARGET', format='16A', array=[name_star])
    col3 = fits.Column(name='RAEP0', format='1D', unit='DEGREES', array=ra)
    col4 = fits.Column(name='DECEP0', format='1D', unit='DEGREES', array=dec)
    col5 = fits.Column(name='EQUINOX', format='1E', unit='YEARS', array=[2000])
    col6 = fits.Column(name='RA_ERR', format='1D', unit='DEGREES', array=[0])
    col7 = fits.Column(name='DEC_ERR', format='1D', unit='DEGREES', array=[0])
    col8 = fits.Column(name='SYSVEL', format='1D', unit='M/S', array=[0])
    col9 = fits.Column(name='VELTYP', format='8A', array=['UNKNOWN'])
    col10 = fits.Column(name='VELDEF', format='8A', array=['OPTICAL'])
    col11 = fits.Column(name='PMRA', format='1D', unit='DEG/YR', array=pmra)
    col12 = fits.Column(name='PMDEC', format='1D', unit='DEG/YR', array=pmdec)
    col13 = fits.Column(name='PMRA_ERR', format='1D', unit='DEG/YR', array=[0])
    col14 = fits.Column(name='PMDEC_ERR', format='1D', unit='DEG/YR', array=[0])
    col15 = fits.Column(name='PARALLAX', format='1E', unit='DEGREES', array=plx)
    col16 = fits.Column(name='PARA_ERR', format='1E', unit='DEGREES', array=[0])
    col17 = fits.Column(name='SPECTYP', format='16A', array=spectyp)

    coldefs = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,
                            col10,col11,col12,col13,col14,col15,col16,col17])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header['EXTNAME'] = 'OI_TARGET'
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdulist.append(hdu)

    # ------------------------------
    #           OI Array
    # ------------------------------

    if verbose:
        print('-> Including OI Array table...')
    try:
        staxy = dic['info']['STAXY'] # these are the mask hole xy-coords as built (ctrs_inst)
    except KeyError:
        staxy = dic['OI_ARRAY']['STAXY']
    try:
        ctrs_eqt = dic['info']['CTRS_EQT']
    except KeyError:
        ctrs_eqt = dic['OI_ARRAY']['CTRS_EQT']

    N_ap = len(staxy)

    tel_name = ['A%i' % x for x in np.arange(N_ap)+1]
    sta_name = tel_name
    diameter = [0] * N_ap

    staxyz = []
    for x in staxy:
        a = list(x)
        line = [a[0], a[1], 0]
        staxyz.append(line)



    sta_index = np.arange(N_ap) + 1

    pscale = dic['info']['PSCALE']/1000.  # arcsec
    isz = dic['info']['ISZ']  # Size of the image to extract NRM data
    fov = [pscale * isz] * N_ap
    fovtype = ['RADIUS'] * N_ap


    col1 = fits.Column(name='TEL_NAME', format='16A', array=tel_name)
    col2 = fits.Column(name='STA_NAME', format='16A', array=sta_name)
    col3 = fits.Column(name='STA_INDEX', format='1I', array=sta_index)
    col4 = fits.Column(name='DIAMETER', unit='METERS', format='1E', array=diameter)
    col5 = fits.Column(name='STAXYZ', unit='METERS', format='3D', array=staxyz)
    col6 = fits.Column(name='FOV', unit='ARCSEC', format='1D', array=fov)
    col7 = fits.Column(name='FOVTYPE', format='6A', array=fovtype)
    col8 = fits.Column(name='CTRS_EQT', unit='METERS', format='2D', array=ctrs_eqt) # for debugging

    coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header['EXTNAME'] = 'OI_ARRAY'
    hdu.header['ARRAYX'] = float(0)
    hdu.header['ARRAYY'] = float(0)
    hdu.header['ARRAYZ'] = float(0)
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['FRAME'] = 'SKY'
    hdu.header['PSCALE'] = pscale * 1000.  # [mas] RAC 9/2020
    hdu.header['ISZ'] = isz  # RAC 9/2020
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'

    hdulist.append(hdu)

    # ------------------------------
    #           OI VIS
    # ------------------------------

    if verbose:
        print('-> Including OI Vis table...')

    data = dic['OI_VIS']
    npts = len(dic['OI_VIS']['VISAMP'])

    sta_index = Format_STAINDEX_V2(data['STA_INDEX'])
    some_keys = ['TARGET_ID','TIME','MJD','INT_TIME']
    for akey in some_keys:
        try:
            if len(data[akey]) > 1:
                data[akey] = data[akey][0]
                if len(data[akey]) > 1:
                    data[akey] = data[akey][0]
        except TypeError:
            pass
    if len(data['VISAMP'].shape) == 1: # figure out if it's multi-slice or not
        nslice = 1
    else:
        nslice = data['VISAMP'].shape[1] # this would cause an error if not multi-slice
    col1 = fits.Column(name='TARGET_ID', format='1I',
                    array=[data['TARGET_ID']]*npts)
    col2 = fits.Column(name='TIME', format='1D', unit='SECONDS',
                    array=[data['TIME']]*npts)
    col3 = fits.Column(name='MJD', unit='DAY', format='1D',
                    array=[data['MJD']]*npts)
    col4 = fits.Column(name='INT_TIME', format='1D', unit='SECONDS',
                    array=[data['INT_TIME']]*npts)
    col5 = fits.Column(name='VISAMP', format='%dD'%nslice, array=data['VISAMP'])
    col6 = fits.Column(name='VISAMPERR', format='%dD'%nslice, array=data['VISAMPERR'])
    col7 = fits.Column(name='VISPHI', format='%dD'%nslice, unit='DEGREES',
                    array=data['VISPHI'])
    col8 = fits.Column(name='VISPHIERR', format='%dD'%nslice, unit='DEGREES',
                    array=data['VISPHIERR'])
    col9 = fits.Column(name='UCOORD', format='1D',
                    unit='METERS', array=data['UCOORD'])
    col10 = fits.Column(name='VCOORD', format='1D',
                    unit='METERS', array=data['VCOORD'])
    col11 = fits.Column(name='STA_INDEX', format='2I', array=sta_index)
    col12 = fits.Column(name='FLAG', format='1L', array=data['FLAG'])

    coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9,
                            col10, col11, col12])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['EXTNAME'] = 'OI_VIS'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS'], 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    # ------------------------------
    #           OI VIS2
    # ------------------------------
    if verbose:
        print('-> Including OI Vis2 table...')

    data = dic['OI_VIS2']
    npts = len(dic['OI_VIS2']['VIS2DATA'])

    some_keys = ['TARGET_ID', 'TIME', 'MJD', 'INT_TIME']
    for akey in some_keys:
        try:
            if len(data[akey]) > 1:
                data[akey] = data[akey][0]
                if len(data[akey]) > 1:
                    data[akey] = data[akey][0]
        except TypeError:
            pass
    col1 = fits.Column(name='TARGET_ID', format='1I',
                    array=[data['TARGET_ID']]*npts)
    col2 = fits.Column(name='TIME', format='1D', unit='SECONDS',
                    array=[data['TIME']]*npts)
    col3 = fits.Column(name='MJD', unit='DAY', format='1D',
                    array=[data['MJD']]*npts)
    col4 = fits.Column(name='INT_TIME', format='1D', unit='SECONDS',
                    array=[data['INT_TIME']]*npts)
    col5 = fits.Column(name='VIS2DATA', format='%dD'%nslice, array=data['VIS2DATA'])
    col6 = fits.Column(name='VIS2ERR', format='%dD'%nslice, array=data['VIS2ERR'])
    col7 = fits.Column(name='UCOORD', format='1D',
                    unit='METERS', array=data['UCOORD'])
    col8 = fits.Column(name='VCOORD', format='1D',
                    unit='METERS', array=data['VCOORD'])
    col9 = fits.Column(name='STA_INDEX', format='2I', array=sta_index)
    col10 = fits.Column(name='FLAG', format='1L', array=data['FLAG'])

    coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9,
                            col10])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header['EXTNAME'] = 'OI_VIS2'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS'], 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    # ------------------------------
    #           OI T3
    # ------------------------------
    if verbose:
        print('-> Including OI T3 table...')

    data = dic['OI_T3']
    npts = len(dic['OI_T3']['T3PHI'])

    sta_index = Format_STAINDEX_T3(data['STA_INDEX'])
    some_keys = ['TARGET_ID', 'TIME', 'MJD', 'INT_TIME']
    for akey in some_keys:
        try:
            if len(data[akey]) > 1:
                data[akey] = data[akey][0]
                if len(data[akey]) > 1:
                    data[akey] = data[akey][0]
        except TypeError:
            pass

    col1 = fits.Column(name='TARGET_ID', format='1I', array=[1]*npts)
    col2 = fits.Column(name='TIME', format='1D', unit='SECONDS', array=[0]*npts)
    col3 = fits.Column(name='MJD', format='1D', unit='DAY',
                    array=[data['MJD']]*npts)
    col4 = fits.Column(name='INT_TIME', format='1D', unit='SECONDS',
                    array=[data['INT_TIME']]*npts)
    col5 = fits.Column(name='T3AMP', format='%dD'%nslice, array=data['T3AMP'])
    col6 = fits.Column(name='T3AMPERR', format='%dD'%nslice, array=data['T3AMPERR'])
    col7 = fits.Column(name='T3PHI', format='%dD'%nslice, unit='DEGREES',
                    array=data['T3PHI'])
    col8 = fits.Column(name='T3PHIERR', format='%dD'%nslice, unit='DEGREES',
                    array=data['T3PHIERR'])
    col9 = fits.Column(name='U1COORD', format='1D',
                    unit='METERS', array=data['U1COORD'])
    col10 = fits.Column(name='V1COORD', format='1D',
                    unit='METERS', array=data['V1COORD'])
    col11 = fits.Column(name='U2COORD', format='1D',
                    unit='METERS', array=data['U2COORD'])
    col12 = fits.Column(name='V2COORD', format='1D',
                    unit='METERS', array=data['V2COORD'])
    col13 = fits.Column(name='STA_INDEX', format='3I', array=sta_index)
    col14 = fits.Column(name='FLAG', format='1L', array=data['FLAG'])

    coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9,
                            col10, col11, col12, col13, col14])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header['EXTNAME'] = 'OI_T3'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdu.header['ARRNAME'] = dic['info']['MASK'] # Anand 9/2020
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS'], 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    # ------------------------------
    #          Save file
    # ------------------------------
    #print(os.path.join(datadir,filename))
    hdulist.writeto(os.path.join(datadir,filename), overwrite=True)
    cprint('\n\n### OIFITS CREATED (%s).' % filename, 'cyan')
    del(hdu)
    del(hdulist)



def load(filename, target=None, ins=None, mask=None, include_vis=True):
    """ Load oifits file and create the dictionary format to be read and plotted.

    Parameters
    ----------
    `filename` {str}:
        Name of the oifits file,\n
    `target` {str}:
        If target name is not included in the header, use `target` instead,\n
    `ins` {str}:
        If instrument name is not included in the header, use `ins` instead,\n
    `mask` {str}:
        If mask name not included in the header, use `mask` instead,\n
    `include_vis` {boolean}:
        If True, include visibilities amplitude and phase in the oifits (default: True),\n
    """
    with fits.open(filename, mode='readonly', memmap=False) as hdulist:
        fitsHandler = copy.deepcopy(hdulist)

        hdr = fitsHandler[0].header

        dic = {}
        dic['info'] = {}
        try:
            dic['info']['TARGET'] = hdr['OBJECT']
        except KeyError:
            dic['info']['TARGET'] = target
        try:
            dic['info']['OBJECT'] = hdr['OBJECT']
        except KeyError:
            dic['info']['OBJECT'] = None
        try:
            dic['info']['INSTRUME'] = hdr['INSTRUME']
        except KeyError:
            dic['info']['INSTRUME'] = ins
        try:
            dic['info']['MASK'] = hdr['MASK']
        except KeyError:
            dic['info']['MASK'] = mask
        try:
            dic['info']['FILT'] = hdr['FILT']
        except KeyError:
            dic['info']['FILT'] = None
        try:
            dic['info']['DATE-OBS'] = hdr['DATE-OBS']
        except KeyError:
            dic['info']['DATE-OBS'] = None
        try:
            dic['info']['TELESCOP'] = hdr['TELESCOP']
        except KeyError:
            dic['info']['TELESCOP'] = None  # try to get it from somewhere else?
        try:
            dic['info']['OBSERVER'] = hdr['OBSERVER']
        except KeyError:
            dic['info']['OBSERVER'] = None
        try:
            dic['info']['INSMODE'] = hdr['INSMODE']
        except KeyError:
            dic['info']['INSMODE'] = None
        try:
            dic['info']['PA'] = hdr['PA']
        except KeyError:
            dic['info']['PA'] = None

        for hdu in fitsHandler[1:]:
            # RAC 9/2020
            # try to read in info from the OI_ARRAY required for re-saving
            if hdu.header['EXTNAME'] == 'OI_ARRAY':
                try:
                    dic['info']['PSCALE'] = hdu.header['PSCALE']
                except KeyError:
                    continue
                try:
                    dic['info']['ISZ'] = hdu.header['ISZ']
                except KeyError:
                    continue

                # make staxy from staxyz array (remove last column)
                staxyz = hdu.data['STAXYZ']
                staxy = np.delete(staxyz, -1, 1)
                dic['OI_ARRAY'] = {'STAXYZ': staxyz,
                                   'STAXY': staxy,
                                   'CTRS_EQT': hdu.data['CTRS_EQT']
                                   }

            if hdu.header['EXTNAME'] == 'OI_WAVELENGTH':
                dic['OI_WAVELENGTH'] = {'EFF_WAVE': hdu.data['EFF_WAVE'],
                                        'EFF_BAND': hdu.data['EFF_BAND'],
                                        }

            if hdu.header['EXTNAME'] == 'OI_VIS2':
                dic['OI_VIS2'] = {'VIS2DATA': hdu.data['VIS2DATA'],
                                  'VIS2ERR': hdu.data['VIS2ERR'],
                                  'UCOORD': hdu.data['UCOORD'],
                                  'VCOORD': hdu.data['VCOORD'],
                                  'STA_INDEX': hdu.data['STA_INDEX'],
                                  'MJD': hdu.data['MJD'],
                                  'INT_TIME': hdu.data['INT_TIME'],
                                  'TIME': hdu.data['TIME'],
                                  'TARGET_ID': hdu.data['TARGET_ID'],
                                  'FLAG': np.array(hdu.data['FLAG']),
                                  }
                # these are in every extension, but take them from here
                dic['info']['MJD'] = hdu.data['MJD'][0]
                dic['info']['ARRNAME'] = hdu.header['ARRNAME']
                try:
                    dic['OI_VIS2']['BL'] = hdu.data['BL']
                except KeyError:
                    dic['OI_VIS2']['BL'] = (
                        hdu.data['UCOORD']**2 + hdu.data['VCOORD']**2)**0.5

            if hdu.header['EXTNAME'] == 'OI_VIS':
                dic['OI_VIS'] = {'TARGET_ID': hdu.data['TARGET_ID'],
                                 'TIME': hdu.data['TIME'],
                                 'MJD': hdu.data['MJD'],
                                 'INT_TIME': hdu.data['INT_TIME'],
                                 'VISAMP': hdu.data['VISAMP'],
                                 'VISAMPERR': hdu.data['VISAMPERR'],
                                 'VISPHI': hdu.data['VISPHI'],
                                 'VISPHIERR': hdu.data['VISPHIERR'],
                                 'UCOORD': hdu.data['UCOORD'],
                                 'VCOORD': hdu.data['VCOORD'],
                                 'STA_INDEX': hdu.data['STA_INDEX'],
                                 'FLAG': hdu.data['FLAG'],
                                 }
                try:
                    dic['OI_VIS']['BL'] = hdu.data['BL']
                except KeyError:
                    dic['OI_VIS']['BL'] = (
                        hdu.data['UCOORD']**2 + hdu.data['VCOORD']**2)**0.5

            if hdu.header['EXTNAME'] == 'OI_T3':
                u1 = hdu.data['U1COORD']
                u2 = hdu.data['U2COORD']
                v1 = hdu.data['V1COORD']
                v2 = hdu.data['V2COORD']
                u3 = -(u1+u2)
                v3 = -(v1+v2)
                bl_cp = []
                for k in range(len(u1)):
                    B1 = np.sqrt(u1[k]**2+v1[k]**2)
                    B2 = np.sqrt(u2[k]**2+v2[k]**2)
                    B3 = np.sqrt(u3[k]**2+v3[k]**2)
                    bl_cp.append(np.max([B1, B2, B3]))  # rad-1
                bl_cp = np.array(bl_cp)

                dic['OI_T3'] = {'T3PHI': hdu.data['T3PHI'],
                                'T3PHIERR': hdu.data['T3PHIERR'],
                                'T3AMP': hdu.data['T3AMP'],
                                'T3AMPERR': hdu.data['T3AMPERR'],
                                'U1COORD': hdu.data['U1COORD'],
                                'V1COORD': hdu.data['V1COORD'],
                                'U2COORD': hdu.data['U2COORD'],
                                'V2COORD': hdu.data['V2COORD'],
                                'STA_INDEX': hdu.data['STA_INDEX'],
                                'MJD': hdu.data['MJD'],
                                'FLAG': hdu.data['FLAG'],
                                'TARGET_ID': hdu.data['TARGET_ID'],
                                'TIME': hdu.data['TIME'],
                                'INT_TIME': hdu.data['INT_TIME'],
                                }
                try:
                    dic['OI_T3']['BL'] = hdu.data['FREQ'] # why?
                except KeyError:
                    dic['OI_T3']['BL'] = bl_cp
    del(fitsHandler)

    return dic


def show(inputList, diffWl=False, vmin=0, vmax=1.05, cmax=180, setlog=False,
         unit='arcsec', unit_cp='deg'):
    """ Show oifits data of a multiple dataset (loaded with oifits.load) or oifits filename).

    Parameters:
    -----------
    `inputList` {list or str or dict}:
        Single or list of dictionnaries (from `oifits.load` or `ObservablesFromText`) or
        oifits filename,\n
    `diffWl` {bool}:
        If True, differentiate the file (wavelenghts) by color,\n
    `vmin`, `vmax` {float}:
        Minimum and maximum visibilities (default: 0, 1.05),\n
    `cmax` {float}:
        Maximum closure phase [deg] (default: 180),\n
    `setlog` {bool}:
        If True, the visibility curve is plotted in log scale,\n
    `unit` {str}:
        Unit of the sp. frequencies (default: 'arcsec'),\n
    `unit_cp` {str}:
        Unit of the closure phases (default: 'deg'),\n
    """

    if type(inputList) is not list:
        inputList = [inputList]

    if type(inputList[0]) is str:
        l_dic = [load(x) for x in inputList]
        print('Inputs are oifits filename.')
    elif type(inputList[0]) is dict:
        l_dic = inputList
        print('Inputs are dict from oifits.load or ObservablesFromText.')

    # return None

    dic_color = {}
    i_c = 0
    for dic in l_dic:
        filt = dic['info']['FILT']
        if filt not in dic_color.keys():
            dic_color[filt] = list_color[i_c]
            i_c += 1

    fig = plt.figure(figsize=(16, 5.5))
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=4)
    ax3 = plt.subplot2grid((2, 6), (1, 2), colspan=4)

    # Plot plan UV
    # -------
    l_bmax, l_band_al = [], []
    for dic in l_dic:
        tmp = ApplyFlag(dic)
        U = tmp[0]
        V = tmp[1]
        band = tmp[10]
        wl = tmp[9]
        label = '%2.2f $\mu m$ (%s)' % (wl, band)
        if diffWl:
            c1, c2 = dic_color[band], dic_color[band]
            if band not in l_band_al:
                label = '%2.2f $\mu m$ (%s)' % (wl*1e6, band)
        else:
            c1, c2 = '#00adb5', '#fc5185'
        l_bmax.append(tmp[2])
        l_band_al.append(band)

        ax1.scatter(U, V, s=50, c=c1, label=label,
                    edgecolors='#364f6b', marker='o', alpha=1)
        ax1.scatter(-1*np.array(U), -1*np.array(V), s=50, c=c2,
                    edgecolors='#364f6b', marker='o', alpha=1)

    Bmax = np.max(l_bmax)
    ax1.axis([Bmax, -Bmax, -Bmax, Bmax])
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.patch.set_facecolor('#f7f9fc')
    ax1.patch.set_alpha(1)
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    if diffWl:
        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(
            *sorted(zip(labels, handles), key=lambda t: t[0]))
        ax1.legend(handles, labels, loc='best', fontsize=9)
        # ax1.legend(loc='best')

    unitlabel = {'m': 'm',
                 'rad': 'rad$^{-1}$',
                 'arcsec': 'arcsec$^{-1}$',
                 'lambda': 'M$\lambda$'}

    ax1.set_xlabel(r'U [%s]' % unitlabel[unit])
    ax1.set_ylabel(r'V [%s]' % unitlabel[unit])
    ax1.grid(alpha=0.2)

    # Plot V2
    # -------
    max_f_vis = []
    for dic in l_dic:
        tmp = ApplyFlag(dic, unit='arcsec')
        V2 = tmp[3]
        e_V2 = tmp[4]
        sp_freq_vis = tmp[7]
        max_f_vis.append(np.max(sp_freq_vis))
        band = tmp[10]
        if diffWl:
            mfc = dic_color[band]
        else:
            mfc = '#00adb5'

        ax2.errorbar(sp_freq_vis, V2, yerr=e_V2, linestyle="None", capsize=1, mfc=mfc, ecolor='#364f6b', mec='#364f6b',
                     marker='.', elinewidth=0.5, alpha=1, ms=9)

    ax2.hlines(1, 0, 1.2*np.max(max_f_vis),
               lw=1, color='k', alpha=.2, ls='--')

    ax2.set_ylim([vmin, vmax])
    ax2.set_xlim([0, 1.2*np.max(max_f_vis)])
    ax2.set_ylabel(r'$V^2$')
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.patch.set_facecolor('#f7f9fc')
    ax2.patch.set_alpha(1)
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    # ax2.set_xticklabels([])

    if setlog:
        ax2.set_yscale('log')
    ax2.grid(which='both', alpha=.2)

    # Plot CP
    # -------

    if unit_cp == 'rad':
        conv_cp = np.pi/180.
        h1 = np.pi
    else:
        conv_cp = 1
        h1 = np.rad2deg(np.pi)

    cmin = -cmax

    max_f_cp = []
    for dic in l_dic:
        tmp = ApplyFlag(dic, unit='arcsec')
        cp = tmp[5]*conv_cp
        e_cp = tmp[6]*conv_cp
        sp_freq_cp = tmp[8]
        max_f_cp.append(np.max(sp_freq_cp))
        band = tmp[10]
        if diffWl:
            mfc = dic_color[band]
        else:
            mfc = '#00adb5'

        ax3.errorbar(sp_freq_cp, cp, yerr=e_cp, linestyle="None", capsize=1, mfc=mfc, ecolor='#364f6b', mec='#364f6b',
                     marker='.', elinewidth=0.5, alpha=1, ms=9)
    ax3.hlines(h1, 0, 1.2*np.max(max_f_cp),
               lw=1, color='k', alpha=.2, ls='--')
    ax3.hlines(-h1, 0, 1.2*np.max(max_f_cp),
               lw=1, color='k', alpha=.2, ls='--')
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.patch.set_facecolor('#f7f9fc')
    ax3.patch.set_alpha(1)
    ax3.xaxis.set_ticks_position('none')
    ax3.yaxis.set_ticks_position('none')
    ax3.set_xlabel('Spatial frequency [cycle/arcsec]')
    ax3.set_ylabel('Clos. $\phi$ [%s]' % unit_cp)
    ax3.axis([0, 1.2*np.max(max_f_cp), cmin*conv_cp, cmax*conv_cp])
    ax3.grid(which='both', alpha=.2)

    plt.subplots_adjust(top=0.974,
                        bottom=0.091,
                        left=0.04,
                        right=0.99,
                        hspace=0.127,
                        wspace=0.35)

    plt.show(block=False)
    return fig
