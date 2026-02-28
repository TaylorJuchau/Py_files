import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from astropy.table import Table
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import os
import sys

home_directory = "/d/ret1/Taylor/jupyter_notebooks/Research" 
parent_dir = Path(home_directory).resolve() 
os.chdir(parent_dir) #TJ change working directory to be the parent directory

from Py_files.Basic_analysis import * #TJ import basic functions from custom package

#05/21/2025
#Need to get uncertainty using stochastic variation in each flux
#Calculate EW for each line

def pull_vacuum_data_from_NIST():
    '''grabs the rest wavelengths from NIST. You must go to https://physics.nist.gov/PhysRefData/ASD/lines_form.html and manually change the following:
    Spectrum : HI
    Output Wavelengths : micrometers
    Lower: 0.96
    Upper: 28.095
    Format output: ASCII
    Wavelength in: vacuum all
    Level information: uncheck everything except configurations
    retrieve info, and copy url to this function below

    -------------

    Parameters
    -------------
    None
    
    Returns
    -------------
    array of rest wavelengths and array of two value lists representing [transition from, transition to
    '''   
    
    
    url = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=Hydrogen+I&output_type=0&low_w=0.96&upp_w=28.095&unit=2&de=0&plot_out=0&I_scale_type=1&format=1&line_out=0&remove_js=on&en_unit=0&output=0&bibrefs=1&page_size=100&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&submit=Retrieve+Data"
    #TJ this one uses air rest wavelengths, just to check
    #url = 'https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=Hydrogen+I&output_type=0&low_w=0.96&upp_w=28.095&unit=2&de=0&plot_out=0&I_scale_type=1&format=1&line_out=0&remove_js=on&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=4&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&submit=Retrieve+Data'
    # Fetch the data
    response = requests.get(url)
    response.raise_for_status()  # Check for errors
    
    # Parse HTML table
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('pre')  # NIST outputs data in <pre> tags for ASCII format
    data_lines = table.text.split('\n')
    transitions = []
    wavelengths = []
    
    for line in data_lines:
        parts = line.split('|')
        
        # Only process lines that look like data (i.e., contain many fields)
        if len(parts) > 8:
    
            try:
                lower_bound = int(parts[7])
                upper_bound = int(parts[8])
                wavelengths.append(float(parts[2]))
            except ValueError:
                continue
            
            transitions.append([upper_bound, lower_bound])
    return wavelengths, transitions

def fit_voigt_to_all(WL, flux, flux_unc, trans_wl, transitions, guess_Z = 0.001534, show_plots = False):
    '''Fits voigt profiles to all features

    -------------

    Parameters
    -------------
    WL :  type = array - array of wavelengths including features
    flux :  type = array - flux array representing flux at each wavelength
    flux unc : type = array - uncertainty in flux
    trans_wl : type = array - array with values representing the rest wavelengths of all transitions we are interested in getting fluxes for
    transitions : type = list of two integers - first entry in upper level, second entry is lower level
    guess_Z (optional, defaults to 0.001534) : type = float - estimate for redshift to correct rest wavelengths to observed wavelengths
    
    Returns
    -------------
    list of lists where the first entry in each index is the x-values, the second is the y-values describing the voigt function that fit the feature
    '''   
    voigt_funcs = [] #TJ initialize array of x,y data for each voigt function
    #TJ no longer needed
    #feature_WL, feature_flux, feature_name, feature_idx, feature_unc = get_feature_WL_and_flux(WL, flux, flux_unc, feature_mask, notes)
    for i in range(len(transitions)):
        estimated_obs_wl = trans_wl[i]*(1+guess_Z) #TJ correct for approx redshift
        center_idx = np.argmin(np.abs(WL - estimated_obs_wl)) #TJ assign the center index as the closest wavelength to the expected wavelength
        continuum, cont_std = get_continuum_around(WL, flux, center_idx) #TJ get continuum and continuum stddev
        idx_range = range(center_idx-3,center_idx+4)
        plt_range = range(center_idx-10,center_idx+11)
        x_data = WL[idx_range] #TJ generate the x data as the 20 nearest datapoints
        y_data = flux[idx_range] - continuum #TJ correct y-data for the net above continuum
        flux_uncertainty = flux_unc[idx_range] #TJ assign uncertainty array
        # Initial guesses
        amp_guess = max(flux[center_idx-1:center_idx+1]-continuum) if max(flux[center_idx-1:center_idx+1]-continuum) > 0 else 0
        mean_guess = WL[center_idx]
        width_guess = WL[center_idx+1] - WL[center_idx] if (1 > (WL[center_idx+1] - WL[center_idx]) > 0) else 0.001
        lower_amp = max(flux[center_idx-1:center_idx+1] - continuum) * 0.9
        upper_amp = max(flux[center_idx-1:center_idx+1] - continuum) * 1.1
        upper_amp = upper_amp if upper_amp > 0 else np.inf  # Handle non-positive cases
        
        bounds = (
            [lower_amp, WL[center_idx-2], 0, 0],  # Lower bounds
            [upper_amp, WL[center_idx+2], 1, 1]    # Upper bounds
        )
        
        params, cov = curve_fit(
            voigt,
            x_data,
            y_data,
            p0=[amp_guess, mean_guess, width_guess, width_guess],
            bounds=bounds,
            maxfev=5000
        )
        xrange = np.linspace(min(WL[plt_range]),max(WL[plt_range]), 1000) #TJ define high resolution xrange for plotting
        fitted = voigt(xrange, *params) #TJ create the fitted y-data
        total_feature_flux = np.trapz(fitted, xrange) #TJ integrate over fitted voigt to get total flux
        this_features_snr = params[0]/cont_std #TJ snr is just amp divided by the noise in continuum
        center_WL = params[1] #TJ assign center of the feature for redshift/velocity calculations
        this_feature_flux = flux[idx_range] 
        this_features_unc = flux_unc[idx_range]
        residuals = this_feature_flux - voigt(WL[idx_range], *params)
        chi2 = np.sum((residuals / this_features_unc)**2)
        dof = len(y_data) - len(params)
        reduced_chi2 = chi2 / dof
        if ((this_features_snr > 4) & (reduced_chi2/params[0] < 10)):
            tag = 2
        elif this_features_snr > 4:
            tag = 1
        else:
            tag = 0
        voigt_funcs.append([[xrange, fitted], total_feature_flux, center_WL, this_features_snr, chi2, reduced_chi2, [*params], tag])
        if show_plots:
            plt.plot(WL[plt_range], flux[plt_range]-continuum, label='Continuum-Subtracted', color='purple')
            plt.axvline(x=WL[center_idx])
            if tag == 2:
                plt.plot(xrange, fitted, color='blue', label=f'idx:{i} {transitions[i]}')
            elif tag == 1:
                plt.plot(xrange, fitted, color='green', label=f'idx:{i} chi2:{reduced_chi2:.2f} {transitions[i]}')
            else:
                plt.plot(xrange, fitted, color='red', label=f'idx:{i} {transitions[i]}')
            plt.legend()
            plt.show()
    return voigt_funcs


def get_feature_statistics(voigts, rest_wl_array, transitions):
    c = 2.99792458e+8
    fluxes = []
    center_wl = []
    velocities = []
    z_temp = []
    for i, feature in enumerate(voigts):
        fluxes.append(feature[1])
        center_wl.append(feature[2])
        rest = rest_wl_array[i]*(1e-6)
        obs = feature[2]*(1e-6)
        velocity = c*(obs-rest)/rest
        velocities.append(velocity)
        z_temp.append(((obs-rest)/rest))
    z = np.nanmedian(z_temp)
    return fluxes, center_wl, velocities, z



def get_z_v(index, trans_wl):
    c = 2.99792458e+8
    obs = feature_centers[index]
    rest = trans_wl[index]
    
    velocity = c*(obs-rest)/rest
    z = (obs-rest)/rest
    return velocity, z

def get_good_statistics(voigts, rest_wl_array, transitions):
    c = 2.99792458e+8
    fluxes = []
    center_wl = []
    velocities = []
    z_temp = []

    for i, feature in enumerate(voigts):
        if voigts[i][-1] < 1:
            continue
        fluxes.append(feature[1])
        center_wl.append(feature[2])
        rest = rest_wl_array[i]*(1e-6)
        obs = feature[2]*(1e-6)
        velocity = c*(obs-rest)/rest
        velocities.append(velocity)
        z_temp.append(((obs-rest)/rest))
    z = np.nanmedian(z_temp)
    return fluxes, center_wl, velocities, z