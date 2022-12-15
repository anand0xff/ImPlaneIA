import numpy as np
import os
from astropy.io import fits
from munch import munchify as dict2class
import matplotlib.pyplot as plt
from nrm_analysis.misctools import oifits
from nrm_analysis.misctools import utils
from astropy.io import ascii
from astropy.table import Table

# ---------------------------------------------------------
# Class ObservableSet for handling observables stored in OIFITS files
# ---------------------------------------------------------
class Stats:
    def ___init__(self):
        pass


class Geometry:
    def ___init__(self):
        pass


class ObservableSet:
    """
    Takes an OIFITS file produced by ImPlaneIA (either averaged observables or
    multi-integration observable tables) and reads contents into oi extension.
    If 'multi' (multi-int oifits) stats are calculated and added to stats sub-class
    Otherwise they can be accessed in the extensions of the oi attribute as they
    are stored in the OIFITS, e.g: closure_phases = self.oi.OI_T3.T3PHI
    Mask geometry info is stored in geometry sub-class for convenience.

    """

    def __init__(self, filename):
        self.fn = os.path.basename(filename)
        # read in multi-oifits extensions, make it one attribute of the class
        self.oi = dict2class(oifits.load(filename))
        self.stats = Stats()
        # Populate the geometry sub-class
        self.geometry = Geometry()
        staxyz = self.oi.OI_ARRAY.STAXYZ
        self.geometry.ctrs_inst = np.delete(
            staxyz, -1, 1
        )  # remove the Z column of all zeros
        self.geometry.ctrs_eqt = self.oi.OI_ARRAY.CTRS_EQT
        self.geometry.t3_bl = self.oi.OI_T3.BL
        self.geometry.t3_idx = self.oi.OI_T3.STA_INDEX
        self.geometry.vis_bl = self.oi.OI_VIS.BL
        self.geometry.vis_idx = self.oi.OI_VIS.STA_INDEX
        self.geometry.quad_idx, self.geometry.quad_bl = self.calc_quads()

        # construct strings (useful for labeling plots)
        bl_idx = self.geometry.vis_idx
        tri_idx = self.geometry.t3_idx
        bl_strings = []
        for idx in bl_idx:
            bl_strings.append(str(idx[0]) + '_' + str(idx[1]))
        tri_strings = []
        for idx in tri_idx:
            tri_strings.append(str(idx[0]) + '_' + str(idx[1]) + '_' + str(idx[2]))
        self.geometry.vis_idx_strings = bl_strings
        self.geometry.t3_idx_strings = tri_strings
        # If this is a multi-int oifits, calculate observable statistics
        # and add as attributes in the Stats subclass
        # otherwise access directly
        # judge based on shape
        if len(self.oi.OI_T3.T3PHI.shape) == 2:
            self.multi = True
            self.calc_stats()
        else:
            self.multi = False  # not a multi-integration OIFITS

    def calc_stats(self):  # (assess internal data quality)
        cps = self.oi.OI_T3.T3PHI  # closure phases
        camp = self.oi.OI_T3.T3AMP  # closure amplitudes
        visamp = self.oi.OI_VIS.VISAMP  # visibility amplitudes
        vispha = self.oi.OI_VIS.VISPHI  # visibility phases
        sqvis = self.oi.OI_VIS2.VIS2DATA  # squared visibility amplitudes
        # calculate medians across integrations and add as attributes
        self.stats.med_cps = np.median(cps, axis=1)
        self.stats.med_camp = np.median(camp, axis=1)
        self.stats.med_visamp = np.median(visamp, axis=1)
        self.stats.med_visphi = np.median(vispha, axis=1)
        self.stats.med_sqvis = np.median(sqvis, axis=1)
        # calculate stdev across integrations (35 values for cps, 21 for vis)
        self.stats.std_cps = np.std(cps, axis=1)
        self.stats.std_camp = np.std(camp, axis=1)
        self.stats.std_visamp = np.std(visamp, axis=1)
        self.stats.std_visphi = np.std(vispha, axis=1)
        self.stats.std_sqvis = np.std(sqvis, axis=1)

    def calc_quads(self):
        """returns int array of quad hole indices (0-based),
        and float array of longest baseline among four holes
        """
        nholes = self.geometry.ctrs_eqt.shape[0]
        qlist = []
        for i in range(nholes):
            for j in range(nholes):
                for k in range(nholes):
                    for q in range(nholes):
                        if i < j and j < k and k < q:
                            qlist.append((i, j, k, q))
        qarray = np.array(qlist).astype(np.int)
        uvwlist = []
        # foreach row of 3 elts...
        for quad in qarray:
            uvwlist.append(
                (
                    self.geometry.ctrs_eqt[quad[0]] - self.geometry.ctrs_eqt[quad[1]],
                    self.geometry.ctrs_eqt[quad[1]] - self.geometry.ctrs_eqt[quad[2]],
                    self.geometry.ctrs_eqt[quad[2]] - self.geometry.ctrs_eqt[quad[3]],
                )
            )
        uvwarray = np.array(uvwlist)
        v1coord = uvwarray[:, 0, 0]
        u1coord = uvwarray[:, 0, 1]
        v2coord = uvwarray[:, 1, 0]
        u2coord = uvwarray[:, 1, 1]
        v3coord = uvwarray[:, 2, 0]
        u3coord = uvwarray[:, 2, 1]
        v4coord = -(u2coord + u3coord)
        u4coord = -(v2coord + v3coord)

        n_bispect = len(v1coord)
        bl_camp = []
        for ii in range(n_bispect):
            b1 = np.sqrt(u1coord[ii] ** 2 + v1coord[ii] ** 2)
            b2 = np.sqrt(u2coord[ii] ** 2 + v2coord[ii] ** 2)
            b3 = np.sqrt(u3coord[ii] ** 2 + v3coord[ii] ** 2)
            b4 = np.sqrt(u4coord[ii] ** 2 + v4coord[ii] ** 2)
            bl_camp.append(np.max([b1, b2, b3, b4]))
        bl_camp = np.array(bl_camp)
        return qarray, bl_camp

    def plot_observables(self, saveplot=True, odir="./", annotate=True, outname=None):
        """
        Basic plot of closure phases and squared visbility vs. baseline length
        for averaged observables
        Optionally label each point with the hole indices that form the baseline, or three hole
        indices for closure phases
        Outname should be a mnemonic string; 'observables_plot.png' will be appended otherwise
        the input filename will be used
        """
        t3_bl = self.geometry.t3_bl
        vis_bl = self.geometry.vis_bl
        t3_idx_str = self.geometry.t3_idx_strings
        vis_idx_str = self.geometry.vis_idx_strings
        if self.multi:
            cps = self.stats.med_cps
            cperr = self.stats.std_cps
            sqvis = self.stats.med_sqvis
            sqvis_err = self.stats.std_sqvis
        else:
            cps = self.oi.OI_T3.T3PHI
            cperr = self.oi.OI_T3.T3PHIERR
            sqvis = self.oi.OI_VIS2.VIS2DATA
            sqvis_err = self.oi.OI_VIS2.VIS2ERR
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        ax1.errorbar(t3_bl, cps, yerr=cperr, fmt="go")
        ax2.errorbar(vis_bl, sqvis, yerr=sqvis_err, fmt="go")
        ax1.set_xlabel(r"$B_{max}$", size=14)
        ax1.set_ylabel("Closure phase [deg]", size=14)
        ax1.set_title("Closure Phase", size=16)
        ax2.set_title("Squared Visibility", size=16)
        ax2.set_xlabel(r"$B_{max}$", size=14)
        ax2.set_ylabel("Squared Visibility", size=14)
        plt.suptitle(self.fn)
        ax1.set_ylim([-3.5, 3.5])  # closure phase y limits
        ax2.set_ylim([0.8, 1.1])  # sqv y limits
        if annotate:
            # label each point
            for ii, tri in enumerate(t3_idx_str):
                ax1.annotate(tri, (t3_bl[ii], cps[ii]), xytext=(t3_bl[ii] + 0.05, cps[ii]))
            for ii, bl in enumerate(vis_idx_str):
                ax2.annotate(bl, (vis_bl[ii], sqvis[ii]), xytext=(vis_bl[ii] + 0.05, sqvis[ii]))
        if saveplot:
            if outname is not None:
                plotname = os.path.join(odir, outname + "_observables_plot.png")
            else:
                plotname = os.path.join(odir, self.fn + "_observables_plot.png")
            plt.savefig(plotname)
        else:
            plt.show()

    def plot_observables_allints(self, saveplot=True, odir="./"):
        """
        Plot closure phases and visibility amplitudes vs. baseline length
        for all integrations of an observable set.
        Integrations different colors (really not useful)
        """
        if self.multi:
            t3_bl = self.geometry.t3_bl
            all_cps = self.oi.OI_T3.T3PHI
            vis_bl = self.geometry.vis_bl
            all_visamps = self.oi.OI_VIS.VISAMP
            nints = all_cps.shape[1]
            colormap = plt.cm.gist_ncar
            plt.gca().set_prop_cycle(
                plt.cycler("color", plt.cm.jet(np.linspace(0, 1, nints)))
            )
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

            for ii in np.arange(nints):
                ax1.plot(t3_bl, all_cps[:, ii], ".")
                ax2.plot(vis_bl, all_visamps[:, ii], ".")
            ax1.set_xlabel(r"$B_{max}$", size=14)
            ax1.set_ylabel("Closure phase [deg]", size=14)
            ax1.set_title("Closure Phase", size=16)
            ax2.set_title("Visibility Amplitude", size=16)
            ax2.set_xlabel(r"$B_{max}$", size=14)
            ax2.set_ylabel("Visibility Amplitude", size=14)
            plt.suptitle(self.fn)

            if saveplot:
                plotname = os.path.join(odir, self.fn + "_observables_allints.png")
                plt.savefig(plotname)
            else:
                plt.show()
        else:
            print(
                "This is not a multi-integration OIFITS file, use plot_observables function instead."
            )

    def plot_observables_ramp(self, saveplot=True, odir="./"):
        """
        Plot observables vs integration  number (going up the ramp)
        Look for wacky integrations (to possibly discard?)
        """
        if self.multi:
            all_cps = self.oi.OI_T3.T3PHI
            all_visamps = self.oi.OI_VIS.VISAMP
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
            nints = all_cps.shape[1]
            ncps = all_cps.shape[0]
            nvis = all_visamps.shape[0]
            nints_list = np.arange(nints)
            for i in nints_list:
                ax1.plot([i] * ncps, all_cps[:, i], "b.")
                ax2.plot([i] * nvis, all_visamps[:, i], "b.")
            ax1.set_xlabel("Integration number", size=14)
            ax1.set_ylabel("Closure phase [deg]", size=14)
            ax1.set_title("Closure Phase", size=16)
            ax2.set_title("Visibility Amplitude", size=16)
            ax2.set_xlabel("Integration number", size=14)
            ax2.set_ylabel("Visibility Amplitude", size=14)
            plt.suptitle(self.fn)
            if saveplot:
                plotname = os.path.join(odir, self.fn + "_observables_ramp.png")
                plt.savefig(plotname)
            else:
                plt.show()
        else:
            print(
                "This is not a multi-integration OIFITS file, use plot_observables function instead."
            )


#---------------------------------------------------------
#Class for comparing observables from two files already loaded into ObservableSet
#---------------------------------------------------------
class ObsComp:
    def __init__(self, obsset1, obsset2, odir):
        self.obsset1 = obsset1
        self.obsset2 = obsset2
        self.odir = odir
        # statistics for each observable set have been calculated and added as attributes
        print("Comparing %s to %s" % (obsset1.fn, obsset2.fn))
        print("")

    def compare_calibrators(self, savetxt=False, outname=None):
        self.outname = outname
        if savetxt and outname is None:
            raise Exception("Table file basename is required to save tables")
        if outname is not None:
            print(outname)
        ## zip each with their baselines, triples, or quads and sort and print (with confidence)
        # plot in UV space: larger visamps with larger pts, smaller w smaller pts, label with distance from 1?
        # check each against 7e-3 radians --> degrees (~.4 deg)
        # CPs: use Ireland estimate for achievable contrast with this calibratois not data
        # for median values of important observables:
        # add as attributes for later plotting
        self.visamp_rat = (
            self.obsset1.stats.med_visamp / self.obsset2.stats.med_visamp
        )  # ratio, 21 values
        self.cp_diff = (
            self.obsset1.stats.med_cps - self.obsset2.stats.med_cps
        )  # difference, 35 values
        self.camp_diff = (
            self.obsset1.stats.med_camp - self.obsset2.stats.med_camp
        )  # 35 values
        # confidence: stdevs in quadrature, beware when visamps are close to 0
        self.visamp_conf = np.sqrt(
            self.obsset1.stats.std_visamp ** 2 + self.obsset2.stats.std_visamp ** 2
        )
        # visphi_conf = np.sqrt(self.obsset1.stats.std_visphi**2 + self.obsset2.stats.std_visphi**2)
        self.cp_conf = np.sqrt(
            self.obsset1.stats.std_cps ** 2 + self.obsset2.stats.std_cps ** 2
        )
        self.camp_conf = np.sqrt(
            self.obsset1.stats.std_camp ** 2 + self.obsset2.stats.std_camp ** 2
        )

        ## MAKE SORTED TABLES:
        # get baselines/triples/quads (should be same for both obs sets)
        t3_bl = self.obsset1.geometry.t3_bl
        t3_idx = self.obsset1.geometry.t3_idx
        vis_bl = self.obsset1.geometry.vis_bl
        vis_idx = self.obsset1.geometry.vis_idx
        quad_bl = self.obsset1.geometry.quad_bl
        quad_idx = self.obsset1.geometry.quad_idx
        # reformat hole indices for printing
        t3_idx_str = [
            str(idx[0]) + "_" + str(idx[1]) + "_" + str(idx[2]) for idx in t3_idx
        ]
        vis_idx_str = [str(idx[0]) + "_" + str(idx[1]) for idx in vis_idx]
        quad_idx_str = [
            str(idx[0]) + "_" + str(idx[1]) + "_" + str(idx[2]) + "_" + str(idx[3])
            for idx in quad_idx
        ]
        # sort by baseline length and display
        sorted_cp = sorted(
            zip(t3_idx_str, t3_bl, self.cp_diff, self.cp_conf), key=lambda x: x[1]
        )
        sorted_camp = sorted(
            zip(quad_idx_str, quad_bl, self.camp_diff, self.camp_conf),
            key=lambda x: x[1],
        )
        sorted_visamp = sorted(
            zip(vis_idx_str, vis_bl, self.visamp_rat, self.visamp_conf),
            key=lambda x: x[1],
        )

        # make last column of strings. Meant to be human-readable, not to be read back in for use
        # also less precision for baselines
        t3_idx_str_sorted = [each[0] for each in sorted_cp]
        t3_bl_str = ["%.3f" % each[1] for each in sorted_cp]
        cp_diff_str = ["%.3f +/- %.2f" % (each[2], each[3]) for each in sorted_cp]
        cp_table = Table(
            [t3_idx_str_sorted, t3_bl_str, cp_diff_str],
            names=["Hole Indices", "Baseline [m]", "Closure phase Cal1-Cal2"],
        )
        cp_table.pprint_all()
        print("\n")
        quad_idx_str_sorted = [each[0] for each in sorted_camp]
        quad_bl_str = ["%.3f" % each[1] for each in sorted_camp]
        ca_diff_str = ["%.3f +/- %.2f" % (each[2], each[3]) for each in sorted_camp]
        ca_table = Table(
            [quad_idx_str_sorted, quad_bl_str, ca_diff_str],
            names=["Hole Indices", "Baseline [m]", "Closure amplitude Cal1-Cal2"],
        )
        ca_table.pprint_all()
        print("\n")
        vis_idx_str_sorted = [each[0] for each in sorted_visamp]
        vis_bl_str = ["%.3f" % each[1] for each in sorted_visamp]
        vis_ratio_str = ["%.3f +/- %.2f" % (each[2], each[3]) for each in sorted_visamp]
        vis_table = Table(
            [vis_idx_str_sorted, vis_bl_str, vis_ratio_str],
            names=["Hole Indices", "Baseline [m]", "Visbility amplitude Cal1/Cal2"],
        )
        vis_table.pprint_all()
        print("\n")
        if savetxt:
            cptable_fn = os.path.join(self.odir, outname + "_cp_diff.dat")
            catable_fn = os.path.join(self.odir, outname + "_ca_diff.dat")
            visamptable_fn = os.path.join(self.odir, outname + "_visamp_ratio.dat")

            ascii.write(
                cp_table,
                cptable_fn,
                format="fixed_width_two_line",
                delimiter_pad=" ",
                overwrite=True,
            )
            ascii.write(
                ca_table,
                catable_fn,
                format="fixed_width_two_line",
                delimiter_pad=" ",
                overwrite=True,
            )
            ascii.write(
                vis_table,
                visamptable_fn,
                format="fixed_width_two_line",
                delimiter_pad=" ",
                overwrite=True,
            )
        # return cp_table, ca_table, vis_table

    def plot_calibrator(self, saveplot=True):
        """
        Plot "calibrated" calibrators CPs, Sq vis, vs baseline (or spatial frequency?)
        """
        cps = self.cp_diff
        cp_err = self.cp_conf
        vas = self.visamp_rat
        va_err = self.visamp_conf

        t3_bl = self.obsset1.geometry.t3_bl
        t3_idx = self.obsset1.geometry.t3_idx
        vis_bl = self.obsset1.geometry.vis_bl
        vis_idx = self.obsset1.geometry.vis_idx
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

        ax1.errorbar(t3_bl, cps, yerr=cp_err, fmt="bo")
        ax2.errorbar(vis_bl, vas, yerr=va_err, fmt="bo")
        ax1.set_xlabel("Baseline [m]", size=14)
        ax1.set_ylabel("Closure phase [deg]", size=14)
        ax1.set_title("Closure Phase", size=16)
        ax2.set_title("Visibility Amplitude", size=16)
        ax2.set_xlabel("Baseline [m]", size=14)
        ax2.set_ylabel("Visibility Amplitude", size=14)
        ax1.set_ylim([-3.5, 3.5])  # closure phase y limits
        ax2.set_ylim([0.8, 1.1])  # sqv y limits
        plt.suptitle(self.outname)
        if saveplot:
            plotname = os.path.join(self.odir, self.outname + "_cp_va_plot.png")
            plt.savefig(plotname)
        else:
            plt.show()