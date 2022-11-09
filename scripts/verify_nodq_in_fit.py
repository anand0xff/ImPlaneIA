#! /usr/bin/env python
import sys
import os
import numpy as np
from astropy.io import fits
import string
import nrm_analysis
from nrm_analysis.misctools.implane2oifits import calibrate_oifits

np.set_printoptions(precision=4, linewidth=160)
np.set_printoptions(formatter={'float': lambda x: '{:+.2e}'.format(x)}, linewidth=80)

if __name__ == "__main__":

    oidir = "/Users/anand/data/nis_019/implaneiadev/Saveoif_ov5/"
    oi_abdor = "jw01093001001_01101_00005_nis.oifits"
    oi_37093 = "jw01093004001_01101_00005_nis.oifits"
    oi_36805 = "jw01093006001_01101_00005_nis.oifits"

    # Produce a single calibrated OIFITS file for each  pair
    print("************  Running calibrate ***************")
    calibrate_oifits(oidir+oi_abdor, oidir+oi_37093, oifdir=oidir)
    calibrate_oifits(oidir+oi_abdor, oidir+oi_36805, oifdir=oidir)
    calibrate_oifits(oidir+oi_37093, oidir+oi_36805, oifdir=oidir)
    calibrate_oifits(oidir+oi_36805, oidir+oi_37093, oifdir=oidir)
