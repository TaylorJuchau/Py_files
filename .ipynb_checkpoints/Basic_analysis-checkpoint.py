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
import astropy.units as u
from astropy.wcs import WCS
from astropy.constants import c
from astropy.io import fits
from astropy.visualization import simple_norm, imshow_norm
from photutils.aperture import CircularAperture, aperture_photometry
from spectral_cube import SpectralCube
from astropy.coordinates import SkyCoord
test = 'list of lists'

def try_float(x):
    '''Try to convert item to float, if that fails, leave it as the type that it is, likely a string
    -------------
    Parameters
    -------------
    x : type = variable - item to be converted to float if possible
    
    Returns
    -------------
    item passed as argument, converted to float if it can be
    '''
    try:
        return float(x)
    except ValueError:
        return x


def gaussian_func(x, amplitude, xmean, stddev):
    '''classic gaussian profile

    -------------

    Parameters
    -------------
    x :  type = float - value to be passed on the x-axis to get a y-axis value
    amplitude :  type = float - maximum height of the gaussian
    xmean : type = float - line center
    stddev : type = float - standard deviation of the gaussian
    
    Returns
    -------------
    A single y-value based on the given x-value and other parameters
    '''
    return (amplitude * np.exp(-0.5 * ((x - xmean) / stddev)**2))


def voigt(x, amp, center, sigma, gamma):
    '''classic voigt profile

    -------------

    Parameters
    -------------
    x :  type = float - value to be passed on the x-axis to get a y-axis value
    amp :  type = float - maximum height of the voigt
    center : type = float - line center
    sigma : type = float - standard deviation of the Gaussian contribution
    gamma : type = float - Full Width Half Max of the Lorenzian contribution

    Returns
    -------------
    A single y-value based on the given x-value and other parameters
    '''
    profile = voigt_profile(x - center, sigma, gamma)
    return amp * profile / np.max(profile)


import numpy as np


def get_continuum_around(wavelength_array, flux_array, feature_index, window_size=25, iqr_mult=1.5):
    '''Calculates the surrounding continuum around a feature using robust statistics.

    Parameters
    ----------
    wavelength_array : array-like
        Array of wavelengths
    flux_array : array-like
        Array of flux values (must match wavelength_array length)
    feature_index : int
        Index of the feature center in the arrays
    window_size : int, optional
        Number of indices to search for continuum (default: 25)
    iqr_mult : float, optional
        Multiplier for IQR outlier rejection (default: 1.5)

    Returns
    -------
    mean_cont : float
        Robust average continuum flux
    stdev : float
        Robust standard deviation of continuum
    '''
    
    n = len(wavelength_array)
    start = max(0, feature_index - window_size)
    end = min(n, feature_index + window_size + 1)
    
    # Create continuum window (excluding feature core)
    feature_window = slice(max(0, feature_index-2), min(n, feature_index+3))
    cont_window = np.r_[slice(start, feature_window.start), 
                       slice(feature_window.stop, end)]
    window_fluxes = flux_array[cont_window]
    
    # Handle edge cases
    if len(window_fluxes) < 3:  # Need at least 3 points for meaningful stats
        return np.nan, np.nan
    
    # Calculate robust continuum bounds
    q25, q75 = np.nanpercentile(window_fluxes, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - iqr_mult * iqr
    upper_bound = q75 + iqr_mult * iqr
    
    # Filter valid continuum points
    good_flux = window_fluxes[(window_fluxes >= lower_bound) & 
                            (window_fluxes <= upper_bound) & 
                            ~np.isnan(window_fluxes)]
    
    return np.nanmean(good_flux), np.nanstd(good_flux)

def assign_feature_weights(wavelength_array, flux_array, continuum_array, sigma_cont, extra_cont_points=2, feature_weight=10, continuum_weight=1):
    """
    Returns an array of weights for each data point in the wavelength_array.
    
    Points above (continuum + sigma_cont) are assigned feature_weight.
    First and last extra_cont_points are also assigned feature_weight to anchor baseline.
    """
    weights = np.full_like(flux_array, continuum_weight, dtype=float)
    
    # Identify feature indices
    feature_indices = np.where(flux_array > (continuum_array + sigma_cont))[0]
    
    # Assign higher weights to feature
    weights[feature_indices] = feature_weight
    
    # Anchor: assign extra points before and after feature
    if len(feature_indices) > 0:
        first = feature_indices[0]
        last = feature_indices[-1]
        
        # Assign higher weight to N points before the feature starts
        start_anchor = max(0, first - extra_cont_points)
        weights[start_anchor:first] = feature_weight
        
        # Assign higher weight to N points after the feature ends
        end_anchor = min(len(weights), last + extra_cont_points + 1)
        weights[last+1:end_anchor] = feature_weight

    return weights


def fit_voigt_to(wavelength_of_feature, tolerance, wavelength_array, flux_array, type = True, units = 1e+6, show_plot = False, feature_idx_width = 6):
    '''Fits voigt profile to feature nearest to given wavelength.


    need to add backup trial

    -------------

    Parameters
    -------------
    wavelength_of_feature : type = float - wavelength closest 
    tolerance : type = float - number of units (u argument) that the center of the feature can be and still achieve a tag of 2
    wavelength_array :  type = float - array of wavelengths including features
    flux_array : type = list - flux density array at each corresponding wavelength
    type : type = boolean : True if emission feature, False if absorption
    u (optional, defaults to 1e+6 (microns)) : type = float - unit to convert to meters
    Show_plot (optional, defaults to false) : type = boolean - show plot of fit?
    feature_idx_width (optional, defaults to 6) : type = int - number of indexes on each side of the feature's center to fit to


    Returns
    -------------
    [xrange, fitted] : type = list - plotting datapoints in [x,y] format
    total_feature_flux : type = float - integrated flux in units of {flux}
    center_WL, 
    this_features_snr, 
    chi2, reduced_chi2, 
    [*params], 
    tag : type = boolean : 0 representing bad fit or feature not found, 1 representing decent fit, no warnings triggered
    '''   
    if len(wavelength_array) != len(flux_array):
        print(f'wavelength and flux array must be same length, instead len(wavelength) = {len(wavelength)}, but len(flux array) = {len(flux_array)}')
        return None
    
    voigt_func = [] #TJ initialize array of x,y data for each voigt function
    
    center_idx = np.argmin(np.abs(wavelength_array - wavelength_of_feature)) #TJ assign the center index as the closest wavelength to the expected wavelength
    continuum, cont_std = get_continuum_around(wavelength_array, flux_array, center_idx) #TJ get continuum and continuum stddev
    idx_range = range(int(center_idx-np.floor(feature_idx_width/2)),int(center_idx+np.ceil(feature_idx_width/2)))
    plt_range = range(min(idx_range)-feature_idx_width, max(idx_range)+feature_idx_width)
    weights = assign_feature_weights(wavelength_array[idx_range], flux_array[idx_range], continuum, cont_std)
    x_data = wavelength_array[idx_range] #TJ generate the x data as the 20 nearest datapoints
    y_data = flux_array[idx_range] - continuum #TJ correct y-data for the net above continuum
    #flux_uncertainty = flux_unc[idx_range] #TJ assign uncertainty array
    # Initial guesses
    amp_guess = max(flux_array[center_idx-1:center_idx+1]-continuum) if type else min((flux_array[center_idx-1:center_idx+1]-continuum))
    mean_guess = wavelength_array[center_idx]
    half_max = amp_guess / 2
    indices_above_half = np.where(y_data > half_max)[0]

    if len(indices_above_half) >= 2:
        fwhm = x_data[indices_above_half[-1]] - x_data[indices_above_half[0]]
        sigma_guess = fwhm / 2.355
    else:
        sigma_guess = wavelength_array[center_idx+1] - wavelength_array[center_idx]  # fallback
    gamma_guess = sigma_guess / 2
    amp_bounds = [amp_guess*0.75, amp_guess*1.25] 
    
    bounds = ([min(amp_bounds), wavelength_array[center_idx-tolerance], 0, 0], [max(amp_bounds), wavelength_array[center_idx+tolerance], np.inf, np.inf])
    
    params, cov = curve_fit(voigt, x_data, y_data, p0=[amp_guess, mean_guess, sigma_guess, gamma_guess], bounds=bounds, sigma=1/weights, absolute_sigma=False, maxfev=20000)

    xrange = np.linspace(min(wavelength_array[plt_range]),max(wavelength_array[plt_range]), len(wavelength_array[plt_range])*100) #TJ define high resolution xrange for plotting
    fitted = voigt(xrange, *params) #TJ create the fitted y-data
    total_feature_flux = np.trapz(fitted, xrange) #TJ integrate over fitted voigt to get total flux
    this_features_snr = params[0]/cont_std #TJ snr is just amp divided by the noise in continuum
    center_WL = params[1] #TJ assign center of the feature for redshift/velocity calculations
    this_feature_flux = flux_array[idx_range]
    #this_features_unc = flux_unc[idx_range]
    residuals = this_feature_flux - voigt(wavelength_array[idx_range], *params)
    #chi2 = np.sum((residuals / this_features_unc)**2)
    #dof = len(y_data) - len(params)
    #reduced_chi2 = chi2 / dof

    if this_features_snr > 4:
        tag = 1
    else:
        tag = 0
    voigt_func = [[xrange, fitted], total_feature_flux, center_WL, this_features_snr, [*params], tag]
    if show_plot:
        plt.plot(wavelength_array[plt_range], flux_array[plt_range]-continuum, label='Continuum-Subtracted', color='purple')
        plt.axvline(x=voigt_func[2], label = f'center_WL={round(params[1]*u.m.to(u.angstrom))}A')
        if tag == 1:
            plt.plot(xrange, fitted, color='blue', label=f'fitted')
        else:
            plt.plot(xrange, fitted, color='red', label=f'poorly fitted')
        plt.legend()
        plt.show()
    return voigt_func

def get_feature_statistics(rest_wl_array, transitions):
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


def get_IFU_spectrum(IFU_filepath, loc, radius, replace_negatives = False):
    '''extract spectrum from IFU file with aperature of radius, centered at ra,dec = loc
    -------------
    
    Parameters
    -------------
    IFU_filepath : type = str - string to location of IFU fits file
    loc : type = list - ra, dec in degrees or SkyCoord object
    radius : type = float - radius of aperture, must have units attached (like u.deg or u.arcsecond)
    replace_negatives (optional, defaults to nothing) : type = float : replace negative fluxes with this float times the smallest positive flux value, specify             as None to leave as negative values
    Returns
    -------------
    structured array with entries for "wavelength" and "intensity"
    '''   
    #fake_missing_header_info(IFU_filepath) #TJ run this if needed
    hdul = fits.open(IFU_filepath)
    header = hdul['SCI'].header
    wcs = WCS(header)
    cube = SpectralCube.read(IFU_filepath, hdu='SCI')

    # === CONVERT RA/DEC TO PIXEL COORDINATES ===
    # Create SkyCoord object for spatial coordinates
    if type(loc) == list:
        spatial_coords = SkyCoord(ra=loc[0]*u.deg, dec=loc[1]*u.deg)
    elif type(loc) == SkyCoord:
        spatial_coords = loc
    else:
        print('loc is not a list of ra, dec and it is not a SkyCoord object.')
        return None
    
    # Convert spatial coordinates to pixels
    x, y = wcs.celestial.all_world2pix(spatial_coords.ra.deg, 
                                      spatial_coords.dec.deg, 0)
    
    # === BUILD APERTURE ===
    if (header['CDELT2'] != header['CDELT1']) and (header['CDELT2'] != -header['CDELT1']):
        print('pixels are not square! function revisit get_IFU_spectrum() function to fix')
        print(header['CDELT2'], 'in y vs ', header['CDELT1'], ' in x')
        return None
    cdelt = np.abs(header['CDELT2']) * u.deg
    pixel_scale = cdelt.to(u.arcsec)  # arcsec/pixel
    pix_area = header['PIXAR_SR'] #TJ pixel area in steradians
    radius = radius.to(u.arcsec)
    radius_pix = (radius / pixel_scale).value
    aperture = CircularAperture((x, y), r=radius_pix)
    aperture_area_sr = np.pi * (radius.to(u.rad))**2

    # === CRITICAL UNIT HANDLING ===
    cube = cube.with_spectral_unit(u.m)  # Ensure wavelength in meters
    
    # Convert flux units properly
    # Step 1: MJy/sr → W/m²/Hz/sr
    cube = cube.to(u.W/(u.m**2 * u.Hz * u.sr))  
    
    # Step 2: Multiply by pixel area to get W/m²/Hz/pixel
    pix_area_sr = header['PIXAR_SR'] * u.sr
    cube = cube * pix_area_sr
    
    # Step 3: Perform aperture sum (now in W/m²/Hz)
    flux_density_spectrum = []
    nan_detected = 0
    for i in range(len(cube.spectral_axis)):
        image_slice = cube[i].value  # Now in W/m²/Hz
        if replace_negatives is not False:
            if replace_negatives == 0:
                image_slice[image_slice < 0] = 0
            else:
                min_positive = np.nanmin(image_slice[image_slice > 0])
                image_slice[image_slice < 0] = replace_negatives * min_positive
        if ~np.isnan(image_slice).sum() == 0:
            print('An entire wavelength slice in the cube is NaN')
        phot = aperture_photometry(image_slice, aperture)
        if (np.isnan(phot['aperture_sum'][0]).sum() !=0):                
            print(f"{np.isnan(phot['aperture_sum'][0]).sum()} nan values detected for wl[{i}]: {cube.spectral_axis[i]}, this makes {nan_detected}", end="\r")
            nan_detected += 1

        phot = aperture_photometry(np.nan_to_num(image_slice), aperture)
        flux_density_spectrum.append(phot['aperture_sum'][0])  #TJ No extra multiplication! already in correct units
    if nan_detected != 0:
        print(f'A total of {nan_detected} were detected within {radius} in {IFU_filepath.split("/")[-1]} over {len(cube.spectral_axis)} WLs')
    wavelengths = cube.spectral_axis.to(u.m).value
    flux_density_spectrum = np.array(flux_density_spectrum)


    if replace_negatives is not False:
        min_positive = min(flux_density_spectrum[flux_density_spectrum > 0])
        flux_density_spectrum[flux_density_spectrum < 0] = replace_negatives*min_positive  #TJ replace negative numbers with a very small positive value


    dtype = [('wavelength', 'f8'), ('intensity', 'f8')]
    spectrum = np.zeros(len(cube.spectral_axis), dtype=dtype)
    spectrum['wavelength'] = cube.spectral_axis.to(u.m).value
    spectrum['intensity'] = np.array(flux_density_spectrum)

    return spectrum

def find_point_spectrum(IFU_filepath, loc):
    '''extract the pixel spectrum with bilinear interpolation for a location in ra,dec
    -------------
    
    Parameters
    -------------
    IFU_filepath : type = str - string to location of IFU fits file
    loc : type = list - ra, dec in degrees or SkyCoord object
    
    Returns
    -------------
    SpectralCube slice corresponding to the interpolated spectrum
    '''   
    hdul = fits.open(IFU_filepath)
    header = hdul['SCI'].header
    wcs = WCS(header)
    cube = SpectralCube.read(IFU_filepath, hdu='SCI')

    # === CONVERT RA/DEC TO PIXEL COORDINATES ===
    # Create SkyCoord object for spatial coordinates
    if type(loc) == list:
        spatial_coords = SkyCoord(ra=loc[0]*u.deg, dec=loc[1]*u.deg)
    elif type(loc) == SkyCoord:
        spatial_coords = loc
    else:
        print('loc is not a list of ra, dec and it is not a SkyCoord object.')
        return None
    
    # Convert spatial coordinates to pixels
    x, y = wcs.celestial.all_world2pix(spatial_coords.ra.deg, 
                                      spatial_coords.dec.deg, 0)
    
    # Get integer and fractional parts
    x_int, y_int = int(np.floor(x)), int(np.floor(y))
    x_frac, y_frac = x - x_int, y - y_int
    
    # Ensure we don't go out of bounds
    x_max = cube.shape[2] - 1
    y_max = cube.shape[1] - 1
    
    x0 = max(0, min(x_int, x_max))
    x1 = max(0, min(x_int + 1, x_max))
    y0 = max(0, min(y_int, y_max))
    y1 = max(0, min(y_int + 1, y_max))
    
    # Get the four surrounding spectra
    spec00 = cube[:, y0, x0]
    spec01 = cube[:, y0, x1]
    spec10 = cube[:, y1, x0]
    spec11 = cube[:, y1, x1]
    
    # Perform bilinear interpolation
    interpolated_spectrum = (spec00 * (1 - x_frac) * (1 - y_frac) +
                            spec01 * x_frac * (1 - y_frac) +
                            spec10 * (1 - x_frac) * y_frac +
                            spec11 * x_frac * y_frac)
    
    return interpolated_spectrum