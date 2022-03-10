

import unittest, os, glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u


import sys
#sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../'))
#sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))
# print(sys.path)

#import nrm_analysis
import nrm_analysis.nrm_core as nrm_core
import nrm_analysis.InstrumentData as InstrumentData
import nrm_analysis.find_affine2d_parameters as FAP
import nrm_analysis.misctools.utils as utils



arcsec2rad = u.arcsec.to(u.rad)

holeshape='hex'
oversample = 1
n_image = 77

rotd_true = 50.5 # actual rotation used in the affine2d object when creating simulated data
rotd_search = (49.5, 49.9, 50.4, 51.1, 51.5)
scale_search = (0.975, 0.985, 0.995, 1.005, 1.015) # avoid the exact hit of your simulated data's pixel scale, i.e. 1.000

class FindPassAffine2dTestCase(unittest.TestCase):
    def setUp(self):
    
        # setup parameters for simulation
        verbose = 1
        overwrite = 1

        monochromatic_wavelength_m = np.array([(1.0, 4.3e-6),]) 
        mask = 'MASK_NRM'
        filter = 'F430M'
        pixel = 0.0656  # arcsec
        filter_name = 'Monochromatic '+np.str(monochromatic_wavelength_m)

        self.filter = filter
        self.filter_name = filter_name
        self.monochromatic_wavelength_m = monochromatic_wavelength_m

        # directory containing the test data
        datadir = os.path.join(os.path.dirname(__file__),'test_data/find_pass_affine')
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        self.datadir = datadir

        # file containing the test data
        imagefn = datadir+"/simulated_image.fits"
        # use millidegrees and integers
        imagefn = imagefn.replace(".fits","_truerotmd{0:+05d}.fits".format(int(1000.0*float(rotd_true))))
        self.imagefn = imagefn

        # savedir for reduced quantities
        self.savedir = imagefn.replace(".fits","_truerotmd{0:+05d}.fits".format(int(1000.0*float(rotd_true))))
        self.savedir = self.datadir+'/'

        print('\n')
        print('  >> >> >> >> set-up  self.imagefn %s' % self.imagefn)
        print('  >> >> >> >> set-up  self.datadir %s' % self.datadir)
        print('  >> >> >> >> set-up  self.savedir %s' % self.savedir)


        # Create test data on disk
        
        print("avoidhexsingularity:  ", utils.avoidhexsingularity(rotd_true))
        affine = utils.Affine2d(rotradccw=np.pi*utils.avoidhexsingularity(rotd_true)/180.0, 
                                name="{0:.3}".format(float(rotd_true)))
        affine.show(label="Creating PSF data with Affine2d")
        #
        from nrm_analysis.fringefitting.LG_Model import NRM_Model
        jw = NRM_Model(mask='jwst',holeshape="hex", affine2d=affine)
        jw.set_pixelscale(pixel*arcsec2rad)
        jw.simulate(fov=n_image, 
                    bandpass=monochromatic_wavelength_m, 
                    over=oversample)
        # PSF without oversampling
        fits.writeto(imagefn, jw.psf, overwrite=True)
        header = fits.getheader(imagefn)
        header['PIXELSCL'] = (pixel, "arcseconds")
        header['FILTER'] = filter_name
        header['PUPIL'] = mask 
        header = utils.affinepars2header(header, affine)
        fits.update(imagefn, jw.psf, header=header)
        self.simulated_image = jw.psf.copy()
        self.affine = affine # Affine2d parameters used to create test simulated_image ("true" value)
        del jw
        del affine

                
    def test_find_pass_affine_rot(self):
        
        header = fits.getheader(self.imagefn)   
        filename = os.path.basename(self.imagefn)
        datadir = os.path.dirname(self.imagefn)
        print('\n')
        print('  >> >> >> >> testing self.imagefn %s' % self.imagefn)
        print('  >> >> >> >> testing self.datadir %s' % self.datadir)
        print('  >> >> >> >> testing self.savedir %s' % self.savedir)

        fn = filename.replace('.fits','')
        print('  >> >> >> >> fn %s' % fn)

        print("driver rotd_search:", rotd_search)
        print("driver rotd_true:", rotd_true)
        mx, my, sx,sy, xo,yo, = (1.0,1.0, 0.0,0.0, 0.0,0.0)
        psf_offset = (0.0,0.0)

        # FIND ROTATION FROM IMAGE DATA CREATED USING KNOWN ROTATION AND SCALE
        aff_best_rot = FAP.find_rotation(self.simulated_image,psf_offset,
                                         rotd_search, mx, my, sx,sy, xo,yo,
                                         header['PIXELSCL']*arcsec2rad, 
                                         n_image, self.monochromatic_wavelength_m, oversample, holeshape, 
                                         outdir=None)
        print("rotd_measured:",  aff_best_rot)
        aff_best_rot.show(label="Measured ")
        # you have to re-initialize an InstrumentData.NIRISS object with the 
        # best rotation affine parameter now in order to use it in further data reduction

        if 'Monochromatic' in self.filter_name:
            nirissdata = InstrumentData.NIRISS(filt=self.filter, affine2d=aff_best_rot)
            nirissdata.wls = [self.monochromatic_wavelength_m]
            print((nirissdata.wls))

        ff = nrm_core.FringeFitter(nirissdata, oversample=oversample,
                      savedir=self.savedir, datadir=datadir,
                      npix=n_image, interactive=False,
                      hold_centering=True)
        print(('FringeFitter oversampling: %d' % ff.oversample))
    
        threads = 1
        ff.fit_fringes([filename])

        CP_file = sorted(glob.glob(os.path.join(datadir,'%s/%s*.txt' % (filename.split('.')[0],'CPs_') )));
        print("CP_file: ", type(CP_file), len(CP_file))
        CP = Table.read(CP_file[0],format='ascii.no_header',names=({'closure_phase'}))

        #         perform the actual test
        self.assertTrue(np.mean(np.array(CP['closure_phase'])) < 1e-7, 
            'Simulated closure phases of point source are non-zero')  # Avoiding hex singularity in hextransformEE.py causes this sort of noise in exactly-XYaligned PSF 2018 AS


    def test_find_pass_affine_scl(self):
        
        header = fits.getheader(self.imagefn)   
        filename = os.path.basename(self.imagefn)
        datadir = os.path.dirname(self.imagefn)
        print('\n')
        print('  >> >> >> >> testing self.imagefn %s' % self.imagefn)
        print('  >> >> >> >> testing self.datadir %s' % self.datadir)
        print('  >> >> >> >> testing self.savedir %s' % self.savedir)

        fn = filename.replace('.fits','')
        print('  >> >> >> >> fn %s' % fn)

        #mx, my, sx,sy, xo,yo, = (1.0,1.0, 0.0,0.0, 0.0,0.0)

        # FIND SCALE FROM IMAGE DATA CREATED USING KNOWN ROTATION AND SCALE
        # we use the known rotation for the affine2d
        aff_best_scl = FAP.find_scale(   self.simulated_image,
                                         self.affine, #  contains how the test image data was created
                                         scale_search,
                                         header['PIXELSCL']*arcsec2rad, 
                                         n_image, self.monochromatic_wavelength_m, oversample, holeshape, 
                                         outdir=None)
        # you have to re-initialize an InstrumentData.NIRISS object with the
        # best pscale_mas & pscale_rad parameter now in order to use it in further data reduction
        aff_best_scl.show(label="aff_best_scl")

        if 'Monochromatic' in self.filter_name:
            nirissdata = InstrumentData.NIRISS(filt=self.filter, affine2d=aff_best_scl)
            nirissdata.wls = [self.monochromatic_wavelength_m]
            print((nirissdata.wls))
        # naughty coding - adjust attributes to suit quick coding.  Just for testing...
        nirissdata.pscale_mas = header['PIXELSCL'] * 1000  # set
        nirissdata.pscale_rad = nirissdata.pscale_mas/1000. * arcsec2rad # propagate to convenience attribute
    
        ff = nrm_core.FringeFitter(nirissdata, oversample=oversample,
                      savedir=self.savedir, datadir=datadir,
                      npix=n_image, interactive=False,
                      hold_centering=True)
        print(('FringeFitter oversampling: %d' % ff.oversample))
    
        threads = 1
        ff.fit_fringes([filename])

        CP_file = sorted(glob.glob(os.path.join(datadir,'%s/%s*.txt' % (filename.split('.')[0],'CPs_') )));
        print("CP_file: ", type(CP_file), len(CP_file))
        CP = Table.read(CP_file[0],format='ascii.no_header',names=({'closure_phase'}))

        #         perform the actual test
        self.assertTrue(np.mean(np.array(CP['closure_phase'])) < 1e-7, 
            'Simulated closure phases of point source are non-zero')  # Avoiding hex singularity in hextransformEE.py causes this sort of noise in exactly-XYaligned PSF 2018 AS

if __name__ == '__main__':
    unittest.main()
