# nrm_analysis README (WIP) #

Original package development led by Alexandra Greenbaum following legacy code by Greenbaum, Anand Sivaramakrishnan, and Laurent Pueyo. Further design changes and contributions from Sivaramakrishnan, Deepashri Thatte, Johannes Sahlmann, Anthony Soulain, Rachel Cooper.

The package implements the algorithm in [2015ApJ...798...68G](https://ui.adsabs.harvard.edu/abs/2015ApJ...798...68G/abstract)

It extracts interferometric observables from JWST MIRISS aperture masking interferometry (AMI) images/interferograms.  The principal task writes out multiple-slice and averaged oifits files.  

The subdirectory nrm_analysis contains the code.  scripts/**exampleDriver.py** is a current, maintained, working sample script **(TBD)**. 

### Major changes from [Alex Greenbaum's ImPlaneIA ](https://github.com/agreenbaum/ImPlaneIA)
Calibration of a target's raw oifits by another observation's raw oifits is supported by this package, and model-fitting has been removed. Original format text files of observables are still written out, along with oifits files.

The scripts directory contains scripts that were used during code dvelopment.  Some calls may have changed so they might not all work.  Reorg required at some point.   

* TBD 1: Put an exampledDriver.py, including calibrating two observations against each other, into scripts.  
* TBD 2: Put small sample data into a cleanly-defined supporting data directory.  Existing bad-pixel-free and with-bad-pixels AB Dor and a caibrator are good examples.
* Create a requirements file for use by pip

**WebbPSF, pysynphot, and supporting data files**  
[Routines from WebbPSF's Poppy. and supporting JWST filter data](https://webbpsf.readthedocs.io/en/latest/installation.html).  Installing WebbPSF (just using conda or pip) by itself, and [pysynphot](https://webbpsf.readthedocs.io/en/latest/installation.html#synphot-install), and supporting JWST filter, aperture, and spectral data ensure better ImPlaneIA model fidelity.
In bash an environment variable  - e.g.:
**WEBBPSF_PATH=/Users/mylogin/anaconda3/envs/py38/share/webbpsf-data**  
informs ImPlaneIA where WebbPSF's data files live.  Without these files your ImPlaneIA will default to top-hat filter proiles and nominal primary object spectra, and use slow integration methods over filters.  Pysynphot uses the environment variable where a wide range of stellar and other objects' spectra are stored:  
**PYSYN_CDBS=/Users/mylogin/data/cdbs**. 

