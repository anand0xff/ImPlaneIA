

import unittest, os, glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u



import sys
# sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../'))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))


# print(sys.path)

from nrm_analysis import nrm_core, InstrumentData



arcsec2rad = u.arcsec.to(u.rad)

oversample = 11
n_image = 77

class FringeFittingTestCase(unittest.TestCase):
    def setUp(self):
    
        # setup parameters for simulation
        verbose = 1
        overwrite = 1

        monochromatic_wavelength_m = np.array([(1.0, 4.3e-6),])
        mask = 'MASK_NRM'
        filter = 'F430M'
        pixelscale_arcsec = 0.0656 
        filter_name = 'Monochromatic '+np.str(monochromatic_wavelength_m)

        self.filter = filter
        self.filter_name = filter_name
        self.monochromatic_wavelength_m = monochromatic_wavelength_m
        
        # directory containing the test data
        data_dir = os.path.join(os.path.dirname(__file__),'test_data/fringe_fitting')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        out_dir = data_dir

        name_seed = 'PSF_NIRISS_%s_%s'%(mask,filter)

        psf_image_name = name_seed + '_reference.fits'
        psf_image = os.path.join(data_dir,psf_image_name)
        psf_image_without_oversampling = os.path.join(data_dir,psf_image_name.replace('.fits','_without_oversampling.fits'))

        if (not os.path.isfile(psf_image_without_oversampling)) | (overwrite): 
            from nrm_analysis.fringefitting.LG_Model import NRM_Model
            jw = NRM_Model(mask='jwst',holeshape="hex")
            jw.set_pixelscale(pixelscale_arcsec*arcsec2rad)
            jw.simulate(fov=n_image, 
                bandpass=monochromatic_wavelength_m, 
                over=oversample)
            print("FringeFittingTestCase.SetUp: simulation oversampling is", oversample)
            
            # optional writing of oversampled image
            if 0:
                fits.writeto(psf_image,jw.psf_over, clobber=True)
                header = fits.getheader(psf_image)
                header['PIXELSCL'] = pixelscale_arcsec/oversample
                header['FILTER'] = filter_name
                header['PUPIL'] = mask 
                fits.update(psf_image,jw.psf_over/10000./28., header=header)

            # PSF without oversampling
            fits.writeto(psf_image_without_oversampling,jw.psf, clobber=True)
            header = fits.getheader(psf_image_without_oversampling)
            header['PIXELSCL'] = pixelscale_arcsec
            header['FILTER'] = filter_name
            header['PUPIL'] = mask 
            fits.update(psf_image_without_oversampling,jw.psf, header=header)
    
    
        # list of files produced for target
        file_list = glob.glob(os.path.join(data_dir,'*%s*.fits' % 'without_oversampling' ));
            
        self.simulated_image = file_list[0]
        self.psf_image_without_oversampling = psf_image_without_oversampling
                
    def test_fringe_fitting(self):
    
        image_file_name = self.simulated_image        
        
        print(('testing on %s' % image_file_name))
        header = fits.getheader(image_file_name)   
        file_name = os.path.basename(image_file_name)
        data_dir = os.path.dirname(image_file_name)
        save_dir = data_dir + '/'

        if 'Monochromatic' in self.filter_name:
            nirissdata = InstrumentData.NIRISS(filt=self.filter)
            nirissdata.wls = [self.monochromatic_wavelength_m]
            print((nirissdata.wls))
                
    
        nirissdata.pscale_mas = header['PIXELSCL'] * 1000
        print(('measure_fringes: pixelscale %f'%header['PIXELSCL'])) 
        nirissdata.pscale_rad = nirissdata.pscale_mas/1000. * arcsec2rad
    
    
        ff = nrm_core.FringeFitter(nirissdata, oversample=oversample, \
                      savedir=save_dir, datadir=data_dir, \
                      npix=n_image, interactive=False)
        print(('FringeFitter oversampling: %d' % ff.oversample))
    
        threads = 1
        ff.fit_fringes([file_name])

        CP_file = sorted(glob.glob(os.path.join(data_dir,'%s/%s*.txt' % (file_name.split('.')[0],'CPs_') )));
        CP = Table.read(CP_file[0],format='ascii.no_header',names=({'closure_phase'}))

        #         perform the actual test
        print("Closure phases standard deviation (radians)", (np.array(CP['closure_phase'])).std())
        print("Closure phases standard deviation (degrees)", (np.array(CP['closure_phase'])).std()*180.0/np.pi)
        self.assertTrue(np.mean(np.array(CP['closure_phase'])) < 1e-7, 'Simulated closure phases of point source are non-zero')  # Avoiding hex singularity in hextransformEE.py causes this sort of noise in exactly-XYaligned PSF 2018 AS

if __name__ == '__main__':
    unittest.main()
