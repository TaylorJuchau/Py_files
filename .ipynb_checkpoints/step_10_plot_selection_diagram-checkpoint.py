"""
now we use the list of reliable miri sources and explore the selection of the final candidates
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rc('text', usetex=True)

from matplotlib.patches import ConnectionPatch

import os
import shutil
from phangs_data_access import phot_access, helper_func, phangs_info, phot_tools, spec_access, sample_access, spec_tools
from phangs_visualizer import plotting_tools
from phangs_visualizer.multi_panel_visualizer import MultiPanelVisualizer
from phangs_visualizer.phot_visualizer import PhotVisualizer

from astropy.visualization import SqrtStretch, SinhStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import sigma_clipped_stats
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.table import Table
from astropy.io import fits

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture


from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
import matplotlib.ticker as ticker

# compute a 10 micron excess
# estimating the flux which should be detected in accordance to f770w and f1130w
# Therefore we are measuring the wavelength weighted EW of the 10 micron filter
# get the piviot wavelength of both bands
pivot_wave_f770w = helper_func.ObsTools.get_jwst_band_wave(band='F770W', instrument='miri', wave_estimator='pivot_wave', unit='angstrom')
pivot_wave_f1000w = helper_func.ObsTools.get_jwst_band_wave(band='F1000W', instrument='miri', wave_estimator='pivot_wave', unit='angstrom')
pivot_wave_f1130w = helper_func.ObsTools.get_jwst_band_wave(band='F1130W', instrument='miri', wave_estimator='pivot_wave', unit='angstrom')

min_wave_f770w = helper_func.ObsTools.get_jwst_band_wave(band='F770W', instrument='miri', wave_estimator='min_wave', unit='angstrom')
min_wave_f1000w = helper_func.ObsTools.get_jwst_band_wave(band='F1000W', instrument='miri', wave_estimator='min_wave', unit='angstrom')
min_wave_f1130w = helper_func.ObsTools.get_jwst_band_wave(band='F1130W', instrument='miri', wave_estimator='min_wave', unit='angstrom')

max_wave_f770w = helper_func.ObsTools.get_jwst_band_wave(band='F770W', instrument='miri', wave_estimator='max_wave', unit='angstrom')
max_wave_f1000w = helper_func.ObsTools.get_jwst_band_wave(band='F1000W', instrument='miri', wave_estimator='max_wave', unit='angstrom')
max_wave_f1130w = helper_func.ObsTools.get_jwst_band_wave(band='F1130W', instrument='miri', wave_estimator='max_wave', unit='angstrom')

w_eff_f1000w = helper_func.ObsTools.get_jwst_band_wave(band='F1000W', instrument='miri', wave_estimator='w_eff', unit='angstrom')
# calculate the weighted continuum flux
weight_f770w = (pivot_wave_f1000w - pivot_wave_f770w) / (pivot_wave_f1130w - pivot_wave_f770w)
weight_f1130w = (pivot_wave_f1130w - pivot_wave_f1000w) / (pivot_wave_f1130w - pivot_wave_f770w)

si_emit_dict = {
'MWC 480': {'spec_name': '83501201_sws.fit', 'folder_name': '4_se', 'star_type': 'Herbig Ae/Be', 'scale': 10000, 'dist': 162,
            'ra': '04 58 46.10', 'dec': '+29 50 37.5', 'spec_type': '4.SE::'},
'HD 104237': {'spec_name': '10400424_sws.fit', 'folder_name': '4_se', 'star_type': 'Herbig Ae/Be', 'scale': 10000, 'dist': 106.6,
              'ra': '12 00 05.98', 'dec': '-78 11 33.7', 'spec_type': '4.SE:', 'note': 'F'},
# 'Pe 2-8': {'spec_name': '48800628_sws.fit', 'folder_name': '4_se', 'star_type': ' ', 'scale': 100, 'dist': 1,
#            'ra': '15 23 42.86', 'dec': '-57 09 23.3', 'spec_type': '4.SECe'},
'HD 144432': {'spec_name': '45000284_sws.fit', 'folder_name': '4_se', 'star_type': 'late A to early F-type', 'scale': 10000, 'dist': 160,
              'ra': '16 06 58.04', 'dec': '-27 43 08.4', 'spec_type': '4.SE:'},
# 'MWC 247': {'spec_name': '08402942_sws.fit', 'folder_name': '4_se', 'star_type': ' ', 'scale': 100, 'dist': 1,
#             'ra': '17 04 36.23', 'dec': '-33 59 18.9', 'spec_type': '4.SE:e'},
# 'MWC 270': {'spec_name': '46900969_sws.fit', 'folder_name': '4_se', 'star_type': ' ', 'scale': 100, 'dist': 1,
#             'ra': '17 45 57.71', 'dec': '-30 12 00.4', 'spec_type': '4.SEC:e'},
'MWC 275': {'spec_name': '32901191_sws.fit', 'folder_name': '4_se', 'star_type': 'Herbig Ae', 'scale': 10000, 'dist': 122,
            'ra': '17 56 21.35', 'dec': '-21 57 19.5', 'spec_type': '4.SEu:'},
'IRAS 18062+2410': {'spec_name': '46000275_sws.fit', 'folder_name': '4_se', 'star_type': 'hot post-AGB', 'scale': 1000, 'dist': 7326.134,
                    'ra': '18 08 20.15', 'dec': '+24 10 43.9', 'spec_type': '4.SE:'},
'IRAS 18095+2704': {'spec_name': '31101819_sws.fit', 'folder_name': '4_se', 'star_type': 'F-supergiant/protoplanetary nebular', 'scale': 100, 'dist': 7244,
                    'ra': '18 11 30.60', 'dec': '+27 05 16.0', 'spec_type': '4.SEC'},
'AC Her': {'spec_name': '10600514_sws.fit', 'folder_name': '4_se', 'star_type': 'RV Tauri variable', 'scale': 1000, 'dist': 1280,
           'ra': '18 30 16.30', 'dec': '+21 52 00.0', 'spec_type': '4.SEC'},
'NGC 6790': {'spec_name': '13401107_sws.fit', 'folder_name': '4_se', 'star_type': 'PN', 'scale': 1000, 'dist': 5.7 * 1e3,
             'ra': '19 22 57.00', 'dec': '+01 30 46.5', 'spec_type': '4.SE:e'},
# 'Vy 2-2': {'spec_name': '32002528_sws.fit', 'folder_name': '4_se', 'star_type': ' ', 'scale': 100, 'dist': 1,
#            'ra': '19 24 21.88', 'dec': '+09 53 54.8', 'spec_type': '4.SECe'},
'IRC +10420': {'spec_name': '12801311_sws.fit', 'folder_name': '4_se', 'star_type': 'yellow hypergiant', 'scale': 10, 'dist': 5000,
               'ra': '19 26 47.99', 'dec': '+11 21 16.8', 'spec_type': '4.SEC'},
'IRAS 20572+4919': {'spec_name': '40300736_sws.fit', 'folder_name': '4_se', 'star_type': 'F-Type', 'scale': 10000, 'dist': 500,
                    'ra': '20 58 55.60', 'dec': '+49 31 13.0', 'spec_type': '4.SE:'},
'Hb 12': {'spec_name': '43700330_sws.fit', 'folder_name': '4_se', 'star_type': 'nested planetary nebula', 'scale': 100, 'dist': 14250, # 'scale': 100, 'dist': 2240,
          'ra': '23 26 14.68', 'dec': '+58 10 54.7', 'spec_type': '4.SECe'},

'HV 888': {'spec_name': 'cassis_yaaar_spcfw_6015488t.fits', 'folder_name': 'rsg', 'star_type': 'RSG', 'scale': 100, 'dist': 49.97 * 1e3,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},
'eta Car': {'spec_name': '23701861_sws.fit', 'folder_name': 'eta_car', 'star_type': 'miscellaneous', 'scale': 1, 'dist': 2300,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},
'WR 124': {'spec_name': '72500754_sws.fit', 'folder_name': 'wr', 'star_type': 'WR', 'scale': 1, 'dist': 6400,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},
'WR 140': {'spec_name': '35200913_sws.fit', 'folder_name': 'wr', 'star_type': 'WR', 'scale': 100, 'dist': 1518,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},
'WR 136': {'spec_name': '38102211_sws.fit', 'folder_name': 'wr', 'star_type': 'WR', 'scale': 100, 'dist': 2100,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},
'WR 34': {'spec_name': '45701204_sws.fit', 'folder_name': 'wr', 'star_type': 'WR', 'scale': 100, 'dist': 5000,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},
'WR 70': {'spec_name': '43400604_sws.fit', 'folder_name': 'wr', 'star_type': 'WR', 'scale': 100, 'dist': 4100,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},
'WR 147': {'spec_name': '04800954_sws.fit', 'folder_name': 'wr', 'star_type': 'WR', 'scale': 1, 'dist': 630,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},

'WR 24': {'spec_name': 'cassis_yaaar_spcfw_9269248t.fits', 'folder_name': 'wr', 'star_type': 'WR', 'scale': 1, 'dist': 4200,
               'ra': ' ', 'dec': ' ', 'spec_type': ' '},

'AG CAR': {'spec_name': '04000652_sws.fit', 'folder_name': 'ag_car', 'star_type': 'LBV', 'scale': 100, 'dist': 5200,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},
'P Cyg': {'spec_name': '03201129_sws.fit', 'folder_name': 'p_cyg', 'star_type': 'LBV', 'scale': 100, 'dist': 1610,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},
'R136a': {'spec_name': 'cassis_yaaar_sptfc_12081408_3t.fits', 'folder_name': 'r136a', 'star_type': 'VMS', 'scale': 1, 'dist': 49970,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},
'HD 15558': {'spec_name': 'cassis_yaaar_spcfw_27581440t.fits', 'folder_name': 'o_type', 'star_type': 'O-type', 'scale': 1, 'dist': 1700,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},

'NGC 3606': {'spec_name': 'cassis_yaaar_opt_12080384_6.fits', 'folder_name': 'ngc3603', 'star_type': 'massive HII region', 'scale': 1000, 'dist': 6900,
              'ra': ' ', 'dec': ' ', 'spec_type': ' '},
}

filter_f770w = np.genfromtxt('data/miri_filter/JWST_MIRI.F770W.dat')
wave_response_f770w = filter_f770w[:, 0] * 1e-4
tras_response_f770w = filter_f770w[:, 1]
tras_response_f770w_norm = tras_response_f770w / np.sum(tras_response_f770w)

filter_f1000w = np.genfromtxt('data/miri_filter/JWST_MIRI.F1000W.dat')
wave_response_f1000w = filter_f1000w[:, 0] * 1e-4
tras_response_f1000w = filter_f1000w[:, 1]
tras_response_f1000w_norm = tras_response_f1000w / np.sum(tras_response_f1000w)

filter_f1130w = np.genfromtxt('data/miri_filter/JWST_MIRI.F1130W.dat')
wave_response_f1130w = filter_f1130w[:, 0] * 1e-4
tras_response_f1130w = filter_f1130w[:, 1]
tras_response_f1130w_norm = tras_response_f1130w / np.sum(tras_response_f1130w)


def get_star_mag_excess(star_name):

    # get the stellar spectrum
    hdu = fits.open('data/' + si_emit_dict[star_name]['folder_name'] + '/' + si_emit_dict[star_name]['spec_name'])
    wave = hdu[0].data[:, 0]
    flux = hdu[0].data[:, 1]
    flux *= 1e3
    sort = np.argsort(wave)
    wave = wave[sort]
    flux = flux[sort]
    dist_star_mpc = si_emit_dict[star_name]['dist'] * 1e-6

    interp_func_star = interp1d(wave, flux)

    star_func_f770w = interp_func_star(wave_response_f770w)
    star_func_f1000w = interp_func_star(wave_response_f1000w)
    star_func_f1130w = interp_func_star(wave_response_f1130w)

    flux_star_f770w = np.sum(star_func_f770w * tras_response_f770w_norm)
    flux_star_f1000w = np.sum(star_func_f1000w * tras_response_f1000w_norm)
    flux_star_f1130w = np.sum(star_func_f1130w * tras_response_f1130w_norm)


    mag_ab_f1000w_star = helper_func.UnitTools.conv_mjy2ab_mag(flux=flux_star_f1000w)

    abs_mag_ab_f1000w_star = helper_func.UnitTools.conv_mag2abs_mag(mag=mag_ab_f1000w_star, dist=dist_star_mpc)

    weighted_continuum_flux_star = weight_f770w * flux_star_f770w + weight_f1130w * flux_star_f1130w

    weighted_delta_10mu_flux_star = flux_star_f1000w / weighted_continuum_flux_star

    # plt.plot(wave, flux)
    #
    # plt.plot(wave_response_f770w, star_func_f770w)
    # plt.plot(wave_response_f1000w, star_func_f1000w)
    # plt.plot(wave_response_f1130w, star_func_f1130w)
    #
    # plt.scatter(pivot_wave_f770w * 1e-4, flux_star_f770w)
    # plt.scatter(pivot_wave_f1000w * 1e-4, flux_star_f1000w)
    # plt.scatter(pivot_wave_f1130w * 1e-4, flux_star_f1130w)
    # # con
    # plt.show()
    #
    return abs_mag_ab_f1000w_star, weighted_delta_10mu_flux_star


abs_mag_ab_eta_car, weighted_delta_10mu_flux_eta_car = get_star_mag_excess(star_name='eta Car')
abs_mag_ab_yhg, weighted_delta_10mu_flux_yhg = get_star_mag_excess(star_name='IRC +10420')
abs_mag_ab_rsg, weighted_delta_10mu_flux_rsg = get_star_mag_excess(star_name='HV 888')


# load sample access to get properties of individual targets
phangs_sample = sample_access.SampleAccess()
# PHANGS list of HST and JWST covered targets
target_list = phangs_info.phangs_jwst_treasury_1_galaxy_list
# add M51
target_list.append('ngc5194')


target_name_src_detect_comb = np.load('data_output/step_4_combo_cand/target_name_src_detect_comb.npy')
id_src_detect_comb = np.load('data_output/step_4_combo_cand/id_src_detect_comb.npy')
ra_src_detect_comb = np.load('data_output/step_4_combo_cand/ra_src_detect_comb.npy')
dec_src_detect_comb = np.load('data_output/step_4_combo_cand/dec_src_detect_comb.npy')

flux_f770w_apert_corr_comb = np.load('data_output/step_4_combo_cand/flux_f770w_apert_corr_comb.npy')
flux_err_f770w_apert_corr_comb = np.load('data_output/step_4_combo_cand/flux_err_f770w_apert_corr_comb.npy')
flux_f1000w_apert_corr_comb = np.load('data_output/step_4_combo_cand/flux_f1000w_apert_corr_comb.npy')
flux_err_f1000w_apert_corr_comb = np.load('data_output/step_4_combo_cand/flux_err_f1000w_apert_corr_comb.npy')
flux_f1130w_apert_corr_comb = np.load('data_output/step_4_combo_cand/flux_f1130w_apert_corr_comb.npy')
flux_err_f1130w_apert_corr_comb = np.load('data_output/step_4_combo_cand/flux_err_f1130w_apert_corr_comb.npy')
flux_f770w_profile_comb = np.load('data_output/step_4_combo_cand/flux_f770w_profile_comb.npy')
flux_err_f770w_profile_comb = np.load('data_output/step_4_combo_cand/flux_err_f770w_profile_comb.npy')
flux_f1000w_profile_comb = np.load('data_output/step_4_combo_cand/flux_f1000w_profile_comb.npy')
flux_err_f1000w_profile_comb = np.load('data_output/step_4_combo_cand/flux_err_f1000w_profile_comb.npy')
flux_f1130w_profile_comb = np.load('data_output/step_4_combo_cand/flux_f1130w_profile_comb.npy')
flux_err_f1130w_profile_comb = np.load('data_output/step_4_combo_cand/flux_err_f1130w_profile_comb.npy')

mask_hst_cc_hum_match_src_detect_comb = np.load('data_output/step_4_combo_cand/mask_hst_cc_hum_match_src_detect_comb.npy')
mask_hst_cc_ml_match_src_detect_comb = np.load('data_output/step_4_combo_cand/mask_hst_cc_ml_match_src_detect_comb.npy')
mask_pah_pop3_match_src_detect_comb = np.load('data_output/step_4_combo_cand/mask_pah_pop3_match_src_detect_comb.npy')
mask_pah_pop2_match_src_detect_comb = np.load('data_output/step_4_combo_cand/mask_pah_pop2_match_src_detect_comb.npy')
mask_hassani_match_src_detect_comb = np.load('data_output/step_4_combo_cand/mask_hassani_match_src_detect_comb.npy')

cluster_class_hum_comb = np.load('data_output/step_4_combo_cand/cluster_class_hum_comb.npy')
cluster_class_ml_comb = np.load('data_output/step_4_combo_cand/cluster_class_ml_comb.npy')
cluster_class_ml_qual_comb = np.load('data_output/step_4_combo_cand/cluster_class_ml_qual_comb.npy')


color_vi_vega_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/color_vi_vega_hum_src_detect_comb.npy')
color_ub_vega_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/color_ub_vega_hum_src_detect_comb.npy')
color_vi_err_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/color_vi_err_hum_src_detect_comb.npy')
color_ub_err_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/color_ub_err_hum_src_detect_comb.npy')
detect_nuv_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/detect_nuv_hum_src_detect_comb.npy')
detect_u_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/detect_u_hum_src_detect_comb.npy')
detect_b_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/detect_b_hum_src_detect_comb.npy')
detect_v_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/detect_v_hum_src_detect_comb.npy')
detect_i_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/detect_i_hum_src_detect_comb.npy')
abs_v_mag_vega_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/abs_v_mag_vega_hum_src_detect_comb.npy')
age_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/age_hum_src_detect_comb.npy')
mstar_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/mstar_hum_src_detect_comb.npy')
ebv_hum_src_detect_comb = np.load('data_output/step_4_combo_cand/ebv_hum_src_detect_comb.npy')

color_vi_vega_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/color_vi_vega_ml_src_detect_comb.npy')
color_ub_vega_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/color_ub_vega_ml_src_detect_comb.npy')
color_vi_err_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/color_vi_err_ml_src_detect_comb.npy')
color_ub_err_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/color_ub_err_ml_src_detect_comb.npy')
detect_nuv_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/detect_nuv_ml_src_detect_comb.npy')
detect_u_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/detect_u_ml_src_detect_comb.npy')
detect_b_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/detect_b_ml_src_detect_comb.npy')
detect_v_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/detect_v_ml_src_detect_comb.npy')
detect_i_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/detect_i_ml_src_detect_comb.npy')
abs_v_mag_vega_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/abs_v_mag_vega_ml_src_detect_comb.npy')

age_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/age_ml_src_detect_comb.npy')
mstar_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/mstar_ml_src_detect_comb.npy')
ebv_ml_src_detect_comb = np.load('data_output/step_4_combo_cand/ebv_ml_src_detect_comb.npy')

rsg_src_detect_comb = np.load('data_output/step_4_combo_cand/rsg_src_detect_comb.npy')
oagb_src_detect_comb = np.load('data_output/step_4_combo_cand/oagb_src_detect_comb.npy')
cagb_src_detect_comb = np.load('data_output/step_4_combo_cand/cagb_src_detect_comb.npy')
be_src_detect_comb = np.load('data_output/step_4_combo_cand/be_src_detect_comb.npy')
wr_src_detect_comb = np.load('data_output/step_4_combo_cand/wr_src_detect_comb.npy')
cpn_src_detect_comb = np.load('data_output/step_4_combo_cand/cpn_src_detect_comb.npy')


ci_f555w_src_detect_comb = np.load('data_output/step_7_explore_opt/ci_f555w_src_detect_comb.npy')
detect_flag_f555w_src_detect_comb = np.load('data_output/step_7_explore_opt/detect_flag_f555w_src_detect_comb.npy')
flux_f555w_src_detect_comb = np.load('data_output/step_7_explore_opt/flux_f555w_src_detect_comb.npy')
flux_err_f555w_src_detect_comb = np.load('data_output/step_7_explore_opt/flux_err_f555w_src_detect_comb.npy')

ci_f814w_src_detect_comb = np.load('data_output/step_7_explore_opt/ci_f814w_src_detect_comb.npy')
detect_flag_f814w_src_detect_comb = np.load('data_output/step_7_explore_opt/detect_flag_f814w_src_detect_comb.npy')
flux_f814w_src_detect_comb = np.load('data_output/step_7_explore_opt/flux_f814w_src_detect_comb.npy')
flux_err_f814w_src_detect_comb = np.load('data_output/step_7_explore_opt/flux_err_f814w_src_detect_comb.npy')

ci_f200w_src_detect_comb = np.load('data_output/step_7_explore_opt/ci_f200w_src_detect_comb.npy')
detect_flag_f200w_src_detect_comb = np.load('data_output/step_7_explore_opt/detect_flag_f200w_src_detect_comb.npy')
flux_f200w_src_detect_comb = np.load('data_output/step_7_explore_opt/flux_f200w_src_detect_comb.npy')
flux_err_f200w_src_detect_comb = np.load('data_output/step_7_explore_opt/flux_err_f200w_src_detect_comb.npy')

ci_f1000w_src_detect_comb = np.load('data_output/step_7_explore_opt/ci_f1000w_src_detect_comb.npy')
detect_flag_f1000w_src_detect_comb = np.load('data_output/step_7_explore_opt/detect_flag_f1000w_src_detect_comb.npy')
flux_f1000w_src_detect_comb = np.load('data_output/step_7_explore_opt/flux_f1000w_src_detect_comb.npy')
flux_err_f1000w_src_detect_comb = np.load('data_output/step_7_explore_opt/flux_err_f1000w_src_detect_comb.npy')

no_artifacts_mask = np.load('data_output/step_8_inspection_masks/no_artifacts_mask.npy')


mag_vega_f770w_comb = helper_func.UnitTools.conv_mjy2vega(flux=flux_f770w_apert_corr_comb, telescope='jwst', instrument='miri', band='F770W')
mag_vega_f1000w_comb = helper_func.UnitTools.conv_mjy2vega(flux=flux_f1000w_apert_corr_comb, telescope='jwst', instrument='miri', band='F1000W')
mag_vega_f1130w_comb = helper_func.UnitTools.conv_mjy2vega(flux=flux_f1130w_apert_corr_comb, telescope='jwst', instrument='miri', band='F1130W')

mag_ab_f770w_comb = helper_func.UnitTools.conv_mjy2ab_mag(flux=flux_f770w_apert_corr_comb)
mag_ab_f1000w_comb = helper_func.UnitTools.conv_mjy2ab_mag(flux=flux_f1000w_apert_corr_comb)
mag_ab_f1130w_comb = helper_func.UnitTools.conv_mjy2ab_mag(flux=flux_f1130w_apert_corr_comb)

mag_err_f770w_comb = helper_func.UnitTools.conv_mjy_err2vega_err(flux=flux_f770w_apert_corr_comb, flux_err=flux_err_f770w_apert_corr_comb)
mag_err_f1000w_comb = helper_func.UnitTools.conv_mjy_err2vega_err(flux=flux_f1000w_apert_corr_comb, flux_err=flux_err_f1000w_apert_corr_comb)
mag_err_f1130w_comb = helper_func.UnitTools.conv_mjy_err2vega_err(flux=flux_f1130w_apert_corr_comb, flux_err=flux_err_f1130w_apert_corr_comb)

abs_mag_ab_f1000w_comb = np.zeros(len(mag_ab_f1000w_comb))
for target in target_list:
    mask_target = target_name_src_detect_comb == target
    abs_mag_ab_f1000w_comb[mask_target] = helper_func.UnitTools.conv_mag2abs_mag(
        mag=mag_ab_f1000w_comb[mask_target], dist=phangs_sample.get_target_dist(target=target))


# calculate 10 micron excess
weighted_continuum_flux_comb = weight_f770w * flux_f770w_apert_corr_comb + weight_f1130w * flux_f1130w_apert_corr_comb
weighted_continuum_flux_err_comb = np.sqrt((weight_f770w * flux_err_f770w_apert_corr_comb) ** 2 +
                                     (weight_f1130w * flux_err_f1130w_apert_corr_comb) ** 2)

excess_10mu_comb = flux_f1000w_apert_corr_comb / weighted_continuum_flux_comb

excess_10mu_err_comb = np.sqrt(
    (flux_err_f1000w_apert_corr_comb / weighted_continuum_flux_comb) ** 2 +
    ((flux_f1000w_apert_corr_comb * weighted_continuum_flux_err_comb/ (weighted_continuum_flux_comb**2))) ** 2)



# get masks
#
# parameters
# threshold_10mu_emission = 1.24
threshold_10mu_emission = 1.2739759201154451
#
# signal-to-noise
sn_mid_ir_mask = (((flux_f770w_apert_corr_comb/ flux_err_f770w_apert_corr_comb) > 3) &
                  ((flux_f1000w_apert_corr_comb/ flux_err_f1000w_apert_corr_comb) > 3) &
                  ((flux_f1130w_apert_corr_comb/ flux_err_f1130w_apert_corr_comb) > 3))

good_data_points_mask = sn_mid_ir_mask * no_artifacts_mask

print('N total sources ', len(good_data_points_mask),
      ' N enough S/N ', sum(sn_mid_ir_mask),
      ' N artifacts ',  sum(np.invert(no_artifacts_mask)),
      ' N usable data points ', sum(good_data_points_mask))

# masks HST star clusters
mask_star_cluster_hum = detect_v_hum_src_detect_comb == 1
mask_star_cluster_ml = detect_v_ml_src_detect_comb == 1

mask_hst_cc = mask_star_cluster_hum + mask_star_cluster_ml
print('mask_star_cluster_hum ', sum(mask_star_cluster_hum))
print('mask_star_cluster_ml ', sum(mask_star_cluster_ml))
print('mask_hst_cc ', sum(mask_hst_cc))

# mask no optical counterparts
no_opt_counterpart = (((flux_f555w_src_detect_comb / flux_err_f555w_src_detect_comb) < 3) &
                      ((flux_f814w_src_detect_comb / flux_err_f814w_src_detect_comb) < 3))

# print(sum(no_opt_counterpart))
# exit()

only_i_band_mask = (((flux_f555w_src_detect_comb / flux_err_f555w_src_detect_comb) < 3) &
                      ((flux_f814w_src_detect_comb / flux_err_f814w_src_detect_comb) > 3))



# star masks
oagb_star_mask = oagb_src_detect_comb * good_data_points_mask * np.invert(mask_hst_cc)
cagb_star_mask = cagb_src_detect_comb * good_data_points_mask * np.invert(mask_hst_cc)
cpn_star_mask = cpn_src_detect_comb * good_data_points_mask * np.invert(mask_hst_cc)
rsg_star_mask = rsg_src_detect_comb * good_data_points_mask * np.invert(mask_hst_cc)



# mask the final selection
significant_10mu_excess = (excess_10mu_comb > (excess_10mu_err_comb * 3 + threshold_10mu_emission))

bright_10mu_mask = abs_mag_ab_f1000w_comb < -10.0

# pre_selection_mask =   significant_10mu_excess * bright_10mu_mask * good_data_points_mask



interesting_star_cluster_mask = (((mstar_hum_src_detect_comb > 1e5) & ((cluster_class_hum_comb == 1) | (cluster_class_hum_comb == 2))) +
                                 ((mstar_ml_src_detect_comb > 1e5) & ((cluster_class_ml_comb == 1) | (cluster_class_ml_comb == 2))))
# interesting_star_cluster_mask = (((mstar_hum_src_detect_comb > 1e5) & ((cluster_class_hum_comb == 1) | (cluster_class_hum_comb == 2))))
# interesting_star_cluster_mask = (mstar_hum_src_detect_comb > 1e5)


sample_selection_mask = significant_10mu_excess * good_data_points_mask * interesting_star_cluster_mask

if not os.path.isdir('data_output/step_10_sample_selection_mask'): os.makedirs('data_output/step_10_sample_selection_mask')
np.save('data_output/step_10_sample_selection_mask/sample_selection_mask.npy', sample_selection_mask)


print(target_name_src_detect_comb[sample_selection_mask])
print(id_src_detect_comb[sample_selection_mask])
print(age_hum_src_detect_comb[sample_selection_mask])
print(age_ml_src_detect_comb[sample_selection_mask])
print(sum(sample_selection_mask))



# get sed of example clusters
# mask_pah_example = good_data_points_mask & (excess_10mu_comb < 0.4) & (abs_mag_ab_f1000w_comb < -17)
mask_pah_example = good_data_points_mask & (excess_10mu_comb < 0.4) & (abs_mag_ab_f1000w_comb < -15.5) & ((abs_mag_ab_f1000w_comb > -16.5))
mask_10mu_example = id_src_detect_comb == 1185


# get the band wavelength
mean_wave_f770w = helper_func.ObsTools.get_jwst_band_wave(band='F770W', instrument='miri', wave_estimator='mean_wave', unit='mu')
mean_wave_f1000w = helper_func.ObsTools.get_jwst_band_wave(band='F1000W', instrument='miri', wave_estimator='mean_wave', unit='mu')
mean_wave_f1130w = helper_func.ObsTools.get_jwst_band_wave(band='F1130W', instrument='miri', wave_estimator='mean_wave', unit='mu')

min_wave_f770w = helper_func.ObsTools.get_jwst_band_wave(band='F770W', instrument='miri', wave_estimator='min_wave', unit='mu')
min_wave_f1000w = helper_func.ObsTools.get_jwst_band_wave(band='F1000W', instrument='miri', wave_estimator='min_wave', unit='mu')
min_wave_f1130w = helper_func.ObsTools.get_jwst_band_wave(band='F1130W', instrument='miri', wave_estimator='min_wave', unit='mu')

max_wave_f770w = helper_func.ObsTools.get_jwst_band_wave(band='F770W', instrument='miri', wave_estimator='max_wave', unit='mu')
max_wave_f1000w = helper_func.ObsTools.get_jwst_band_wave(band='F1000W', instrument='miri', wave_estimator='max_wave', unit='mu')
max_wave_f1130w = helper_func.ObsTools.get_jwst_band_wave(band='F1130W', instrument='miri', wave_estimator='max_wave', unit='mu')

# # get fluxes
# phot_visual_access = PhotVisualizer(
#     target_name='ngc1365',
#     phot_miri_target_name='ngc1365', miri_data_ver='v1p1p1',
#     nircam_data_ver='v1p1p1')
# phot_visual_access.load_phangs_bands(band_list=['F770W', 'F1000W', 'F1130W'], flux_unit='mJy')
# wcs = phot_visual_access.miri_bands_data['F770W_wcs_img']
# standard_apert_rad_ref_arcsec = phot_tools.ApertTools.get_standard_ap_rad_arcsec(obs='miri', band='F770W',
#                                                                                  wcs=wcs)
# bkg_rad_in_ref_arcsec, bkg_rad_out_ref_arcsec = phot_tools.ApertTools.get_standard_bkg_annulus_rad_arcsec(
#     obs='miri', band='F770W', wcs=wcs)
# phot_table_pah_example, _ = phot_visual_access.compute_complete_photometry(
#     ra_list=ra_src_detect_comb[mask_pah_example], dec_list=dec_src_detect_comb[mask_pah_example], idx_list=None,
#     roi_arcsec=standard_apert_rad_ref_arcsec,
#     bkg_roi_rad_in_arcsec=bkg_rad_in_ref_arcsec,
#     bkg_roi_rad_out_arcsec=bkg_rad_out_ref_arcsec,
#     band_list=None,
#     plot_flag=False,
#     plot_output_path='plot_output/step_12_sample_photometry/f200w_centered/',
#     verbose_flag=False
# )
# phot_table_10mu_example, _ = phot_visual_access.compute_complete_photometry(
#     ra_list=ra_src_detect_comb[mask_10mu_example], dec_list=dec_src_detect_comb[mask_10mu_example], idx_list=None,
#     roi_arcsec=standard_apert_rad_ref_arcsec,
#     bkg_roi_rad_in_arcsec=bkg_rad_in_ref_arcsec,
#     bkg_roi_rad_out_arcsec=bkg_rad_out_ref_arcsec,
#     band_list=None,
#     plot_flag=False,
#     plot_output_path='',
#     verbose_flag=False
# )
#
#
# img_zoom_in_example_pah_emitter, wcs_zoom_in_example_pah_emitter = (
#     phot_visual_access.get_rgb_zoom_in(
#         ra=ra_src_detect_comb[mask_pah_example], dec=dec_src_detect_comb[mask_pah_example], cutout_size=(5, 5),
#         band_red='F770W', band_green='F1000W', band_blue='F1130W'))
#
# img_zoom_in_example_10mu_emitter, wcs_zoom_in_example_10mu_emitter = (
#     phot_visual_access.get_rgb_zoom_in(
#         ra=ra_src_detect_comb[mask_10mu_example], dec=dec_src_detect_comb[mask_10mu_example], cutout_size=(5, 5),
#         band_red='F770W', band_green='F1000W', band_blue='F1130W'))
#
# zoom_in_dict = {
#     'img_zoom_in_example_pah_emitter': img_zoom_in_example_pah_emitter,
#     'wcs_zoom_in_example_pah_emitter': wcs_zoom_in_example_pah_emitter,
#     'img_zoom_in_example_10mu_emitter': img_zoom_in_example_10mu_emitter,
#     'wcs_zoom_in_example_10mu_emitter': wcs_zoom_in_example_10mu_emitter,
# }
#
#
# if not os.path.isdir('data_output/step_10_phot_example_selection_plot/'):
#     os.makedirs('data_output/step_10_phot_example_selection_plot/')
#
# np.save('data_output/step_10_phot_example_selection_plot/zoom_in_dict.npy', zoom_in_dict)
#
#
# phot_table_pah_example.write('data_output/step_10_phot_example_selection_plot/phot_table_pah_example.fits', overwrite=True)
# phot_table_10mu_example.write('data_output/step_10_phot_example_selection_plot/phot_table_10mu_example.fits', overwrite=True)


zoom_in_dict = np.load('data_output/step_10_phot_example_selection_plot/zoom_in_dict.npy', allow_pickle=True).item()
phot_table_pah_example = Table.read('data_output/step_10_phot_example_selection_plot/phot_table_pah_example.fits')
phot_table_10mu_example = Table.read('data_output/step_10_phot_example_selection_plot/phot_table_10mu_example.fits')

##################
#### Plotting ####
##################
#
# parameters
x_lim = (0.11, 6.9)
y_lim = (-5.1, -25.9)


fig = plt.figure(figsize=(20, 15))
fontsize_labels = 27
fontsize_large = 35

# add axis
ax_cmd = fig.add_axes((0.075, 0.06, 0.92, 0.935))

# contours from histogram
color_bins = np.linspace(0.1, 0.7, 15)
mag_bins = np.linspace(-13, -9, 15)
hist_cmd, xedges, yedges = np.histogram2d(x=excess_10mu_comb[good_data_points_mask],
                                          y=abs_mag_ab_f1000w_comb[good_data_points_mask],
                                          bins=(color_bins, mag_bins))
kernel = Gaussian2DKernel(x_stddev=0.5)
hist_cmd = convolve(hist_cmd, kernel)
hist_cmd /= np.sum(hist_cmd)

norm = plotting_tools.ColorBarTools.compute_cbar_norm(vmin_vmax=(0.001, np.max(hist_cmd)*1),
                                                      cutout_list=None, log_scale=False)
# levels = [0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.024, 0.026, 0.028, 0.03, 0.032, 0.034, 0.036, 0.038, 0.040]
levels = [0.01, 0.014, 0.018, 0.024, 0.028, 0.032, 0.036, 0.040]
ax_cmd.contourf(hist_cmd.T, origin='lower', aspect='equal', cmap='Greys_r', norm=norm, levels=levels,
                extent=(color_bins[0], color_bins[-1], mag_bins[0], mag_bins[-1]),
                locator=ticker.MaxNLocator(prune='lower'), zorder=10)
contours = ax_cmd.contour(hist_cmd.T, levels=levels, origin='lower', aspect='equal', cmap='Greys_r', norm=norm,
                         extent=(color_bins[0], color_bins[-1], mag_bins[0], mag_bins[-1]))

# get the contours
contour_collection = contours.allsegs[0]
x_cont = []
y_cont = []
for point in contour_collection[0]:
    x_cont.append(point[0])
    y_cont.append(point[1])
x_cont = np.array(x_cont)
y_cont = np.array(y_cont)
mask_outside_hull = np.invert(helper_func.GeometryTools.check_points_in_polygon(
    x_point=excess_10mu_comb, y_point=abs_mag_ab_f1000w_comb, x_data_hull=x_cont, y_data_hull=y_cont))


ax_cmd.errorbar(excess_10mu_comb[mask_outside_hull * good_data_points_mask],
                abs_mag_ab_f1000w_comb[mask_outside_hull * good_data_points_mask],
                xerr=excess_10mu_err_comb[mask_outside_hull * good_data_points_mask],
                yerr=mag_err_f1000w_comb[mask_outside_hull * good_data_points_mask],
                fmt='.', elinewidth=1, mfc='dimgrey', mec='dimgrey', ecolor='dimgrey', mew=1, ms=13,
                zorder=1)


### plot PAH emitter Rodriguez
color_bins_pop3 = np.linspace(0.1, 0.7, 15)
mag_bins_pop3 = np.linspace(-14, -9, 15)
hist_pop3, xedges, yedges = np.histogram2d(
    x=excess_10mu_comb[mask_pah_pop3_match_src_detect_comb * good_data_points_mask],
    y=abs_mag_ab_f1000w_comb[mask_pah_pop3_match_src_detect_comb * good_data_points_mask],
    bins=(color_bins_pop3, mag_bins_pop3))
hist_pop3 = convolve(hist_pop3, kernel)
hist_pop3 /= np.sum(hist_pop3)
# levels_pop3 = [0.01, 0.014, 0.018, 0.024, 0.028, 0.032, 0.036, 0.040]
levels_pop3 = [0.01, 0.018, 0.026, 0.034, 0.042]
ax_cmd.contour(hist_pop3.T, origin='lower', aspect='equal', cmap='spring', levels=levels_pop3,
               extent=(color_bins_pop3[0], color_bins_pop3[-1], mag_bins_pop3[0], mag_bins_pop3[-1]), zorder=20,
               linewidths=4)
ax_cmd.plot([], [], color=mpl.colormaps['spring'](0), linewidth=4, label='3.3 $\mu$m PAH emitters \n (Rodriguez+25)')


# plot stars as reference
# plotting_tools.color_list_set2
ax_cmd.scatter(weighted_delta_10mu_flux_eta_car, abs_mag_ab_eta_car,
               color='gold', marker='*', s=1000, zorder=20, label=r'$\eta$ Car')
ax_cmd.scatter(weighted_delta_10mu_flux_yhg, abs_mag_ab_yhg,
               color='gold', marker='d', s=700, zorder=20, label='YHG IRC+10420')
ax_cmd.scatter(weighted_delta_10mu_flux_rsg, abs_mag_ab_rsg,
               color='gold', marker='^', s=700, zorder=20, label=r'RSG HV$\,$888')

ax_cmd.scatter(excess_10mu_comb[cagb_star_mask*mask_outside_hull],
               abs_mag_ab_f1000w_comb[cagb_star_mask*mask_outside_hull],
               color='aqua', marker='*', s=300, zorder=10, label='C-AGB (Hassani+2025)')
ax_cmd.scatter(excess_10mu_comb[oagb_star_mask*mask_outside_hull],
               abs_mag_ab_f1000w_comb[oagb_star_mask*mask_outside_hull],
               color='pink', marker='*', s=300, zorder=10, label='O-AGB (Hassani+2025)')
ax_cmd.scatter(excess_10mu_comb[cpn_star_mask*mask_outside_hull],
               abs_mag_ab_f1000w_comb[cpn_star_mask*mask_outside_hull],
               color='blue', marker='*', s=300, zorder=10, label='C PNe (Hassani+2025)')

ax_cmd.scatter(excess_10mu_comb[rsg_star_mask*mask_outside_hull],
               abs_mag_ab_f1000w_comb[rsg_star_mask*mask_outside_hull],
               color='purple', marker='*', s=300, zorder=10, label='RSG (Hassani+2025)')





ax_cmd.scatter(excess_10mu_comb[no_opt_counterpart * good_data_points_mask * mask_outside_hull],
               abs_mag_ab_f1000w_comb[no_opt_counterpart * good_data_points_mask * mask_outside_hull],
               color='k', marker='o', facecolor='None', s=180, zorder=0)
ax_cmd.scatter(excess_10mu_comb[mask_hst_cc * good_data_points_mask * mask_outside_hull],
               abs_mag_ab_f1000w_comb[mask_hst_cc * good_data_points_mask * mask_outside_hull],
               color='k', marker='D', facecolor='None', s=180, zorder=0)

ax_cmd.scatter(excess_10mu_comb[sample_selection_mask],
               abs_mag_ab_f1000w_comb[sample_selection_mask],
               color='tab:red', marker='D', facecolor='None', linewidth=4, s=200, zorder=20)

# print(sum(no_opt_counterpart * good_data_points_mask * mask_outside_hull))
# exit()



# plot selection
ax_cmd.plot([threshold_10mu_emission, threshold_10mu_emission], [-17.5, 0], color='k', linewidth=3, linestyle='--')

ax_cmd.set_xlim(x_lim)
ax_cmd.set_xscale('log')
ax_cmd.set_ylim(y_lim)


ax_cmd.legend(frameon=False, loc=2, bbox_to_anchor=(0.35, 0.99), fontsize=fontsize_labels)

ax_cmd.set_xlabel(r'E$_{\rm 10 \mu m}$', fontsize=fontsize_labels)
ax_cmd.set_ylabel(r'M$_{\rm 10 \mu m}$ [AB mag]', fontsize=fontsize_labels)
ax_cmd.tick_params(axis='x', which='minor', width=2, length=10, right=True, top=True, direction='in',
                   labelsize=fontsize_labels)
ax_cmd.tick_params(axis='x', which='major', width=3, length=15, right=True, top=True, direction='in',
                   labelsize=fontsize_labels)
ax_cmd.tick_params(axis='y', which='both', width=3, length=15, right=True, top=True, direction='in',
                   labelsize=fontsize_labels)




# zoom in parts


ax_sed_example_pah_emitter = fig.add_axes((0.13, 0.71, 0.25, 0.23))

ax_sed_example_pah_emitter.errorbar(
    mean_wave_f770w, phot_table_pah_example['F770W_flux'],
    xerr=[[mean_wave_f770w - min_wave_f770w], [max_wave_f770w - mean_wave_f770w]],
    yerr=phot_table_pah_example['F770W_flux_err'],
fmt='o', color='k', ms=20)
ax_sed_example_pah_emitter.errorbar(
    mean_wave_f1000w, phot_table_pah_example['F1000W_flux'],
    xerr=[[mean_wave_f1000w - min_wave_f1000w], [max_wave_f1000w - mean_wave_f1000w]],
    yerr=phot_table_pah_example['F1000W_flux_err'],
fmt='o', color='k', ms=20)
ax_sed_example_pah_emitter.errorbar(
    mean_wave_f1130w, phot_table_pah_example['F1130W_flux'],
    xerr=[[mean_wave_f1130w - min_wave_f1130w], [max_wave_f1130w - mean_wave_f1130w]],
    yerr=phot_table_pah_example['F1130W_flux_err'],
fmt='o', color='k', ms=20)

# ax_sed_example_pah_emitter.plot([mean_wave_f770w, mean_wave_f1130w],
#                                  [phot_table_pah_example['F770W_flux'],
#                                   phot_table_pah_example['F1130W_flux']],
#                                  linewidth=2, linestyle='--', color='k')

ax_sed_example_pah_emitter.set_xscale('log')
ax_sed_example_pah_emitter.set_yscale('log')

ax_sed_example_pah_emitter.set_xticks([7, 8, 9, 10, 11])
ax_sed_example_pah_emitter.set_xticklabels(['7', '8', '9', '10', '11'])

ax_sed_example_pah_emitter.set_ylim(0.9, 15.1)

ax_sed_example_pah_emitter.tick_params(axis='both', which='both', width=2, length=10, right=True, top=True, direction='in', labelsize=fontsize_labels)

ax_sed_example_pah_emitter.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fontsize_labels)
ax_sed_example_pah_emitter.set_ylabel(r'flux [mjy]', fontsize=fontsize_labels, loc='top')


ax_img_example_pah_emitter = fig.add_axes((0.08, 0.69, 0.15, 0.15), projection=zoom_in_dict['wcs_zoom_in_example_pah_emitter'])
ax_img_example_pah_emitter.imshow(zoom_in_dict['img_zoom_in_example_pah_emitter'])
plotting_tools.WCSPlottingTools.arr_axis_params(ax_img_example_pah_emitter, ra_tick_label=False, dec_tick_label=False,
                        ra_axis_label=' ', dec_axis_label=' ')

con_nucleus_1 = ConnectionPatch(
    xyA=(zoom_in_dict['img_zoom_in_example_pah_emitter'].shape[0] / 2,
         -3), coordsA=ax_img_example_pah_emitter.transData,
    xyB=(excess_10mu_comb[mask_pah_example][0] - 0.02, abs_mag_ab_f1000w_comb[mask_pah_example][0]), coordsB=ax_cmd.transData,
    arrowstyle="simple", connectionstyle="arc3,rad=0.3", color='k', linewidth=3, shrinkA=5, mutation_scale=20)
fig.add_artist(con_nucleus_1)








ax_sed_example_10mu_emitter = fig.add_axes((0.71, 0.69, 0.25, 0.23))

ax_sed_example_10mu_emitter.errorbar(
    mean_wave_f770w, phot_table_10mu_example['F770W_flux'],
    xerr=[[mean_wave_f770w - min_wave_f770w], [max_wave_f770w - mean_wave_f770w]],
    yerr=phot_table_10mu_example['F770W_flux_err'],
fmt='o', color='k', ms=20)
ax_sed_example_10mu_emitter.errorbar(
    mean_wave_f1000w, phot_table_10mu_example['F1000W_flux'],
    xerr=[[mean_wave_f1000w - min_wave_f1000w], [max_wave_f1000w - mean_wave_f1000w]],
    yerr=phot_table_10mu_example['F1000W_flux_err'],
fmt='o', color='k', ms=20)
ax_sed_example_10mu_emitter.errorbar(
    mean_wave_f1130w, phot_table_10mu_example['F1130W_flux'],
    xerr=[[mean_wave_f1130w - min_wave_f1130w], [max_wave_f1130w - mean_wave_f1130w]],
    yerr=phot_table_10mu_example['F1130W_flux_err'],
fmt='o', color='k', ms=20)

# ax_sed_example_10mu_emitter.plot([mean_wave_f770w, mean_wave_f1130w],
#                                  [phot_table_10mu_example['F770W_flux'],
#                                   phot_table_10mu_example['F1130W_flux']],
#                                  linewidth=2, linestyle='--', color='k')

ax_sed_example_10mu_emitter.set_xscale('log')
ax_sed_example_10mu_emitter.set_yscale('log')

ax_sed_example_10mu_emitter.set_xticks([7, 8, 9, 10, 11])
ax_sed_example_10mu_emitter.set_xticklabels(['7', '8', '9', '10', '11'])

ax_sed_example_10mu_emitter.set_ylim(0.21, 13)

ax_sed_example_10mu_emitter.tick_params(axis='both', which='both', width=2, length=10, right=True, top=True, direction='in', labelsize=fontsize_labels)

ax_sed_example_10mu_emitter.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fontsize_labels)
ax_sed_example_10mu_emitter.set_ylabel(r'flux [mjy]', fontsize=fontsize_labels)


ax_img_example_10mu_emitter = fig.add_axes((0.665, 0.82, 0.15, 0.15), projection=zoom_in_dict['wcs_zoom_in_example_10mu_emitter'])
ax_img_example_10mu_emitter.imshow(zoom_in_dict['img_zoom_in_example_10mu_emitter'])
plotting_tools.WCSPlottingTools.arr_axis_params(ax_img_example_10mu_emitter, ra_tick_label=False, dec_tick_label=False,
                        ra_axis_label=' ', dec_axis_label=' ')

con_nucleus_1 = ConnectionPatch(
    xyA=(zoom_in_dict['img_zoom_in_example_10mu_emitter'].shape[0] / 3,
         -3), coordsA=ax_img_example_10mu_emitter.transData,
    xyB=(excess_10mu_comb[mask_10mu_example][0] - 0.2, abs_mag_ab_f1000w_comb[mask_10mu_example][0] - 0.1),
    coordsB=ax_cmd.transData,
    arrowstyle="simple", connectionstyle="arc3,rad=0.4", color='k', linewidth=3, shrinkA=5, mutation_scale=20)
fig.add_artist(con_nucleus_1)


ax_img_example_10mu_emitter.scatter([], [], color='tab:red', marker='D', facecolor='None', linewidth=4, s=200, label='This Work')
ax_img_example_10mu_emitter.scatter([], [], color='k', marker='o', facecolor='None', s=180, zorder=0, label='Star clusters \n (Maschmann+24)')
ax_img_example_10mu_emitter.scatter([], [], color='k', marker='D', facecolor='None', s=180, zorder=0, label='No Opt. counterpart')

ax_img_example_10mu_emitter.legend(frameon=True, loc=2, bbox_to_anchor=(0.6, -1.5), fontsize=fontsize_labels)


fig.savefig('plot_output/excess_mag_final.png')
fig.savefig('plot_output/excess_mag_final.pdf')


