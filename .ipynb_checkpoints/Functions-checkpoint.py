import numpy as np
from tabulate import tabulate
from pathlib import Path
import pickle
import os
import glob
import re
import sys
import pandas as pd
import requests
import shutil

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec

from photutils.aperture import CircularAperture, aperture_photometry
from spectral_cube import SpectralCube

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from astropy.wcs import WCS
from astropy.constants import c
from astropy.io import fits
from astropy.visualization import simple_norm, imshow_norm
from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch 
from astropy.nddata import Cutout2D
from astropy.convolution import convolve_fft
from astropy.modeling.models import Gaussian2D

import warnings


warnings.filterwarnings("ignore", message="WCS1 is missing card .*")              

#TJ define functions needed to generate files
os.chdir('/d/ret1/Taylor/jupyter_notebooks/Research') #TJ change working directory to be the parent directory

locations = [[202.5062429, 47.2143358], [202.4335225, 47.1729608], [202.4340450, 47.1732517], [202.4823742, 47.1958589]]
print('Some functions use pre-defined data files, a warning will print when this is the case.')
filter_directory = '/d/crow1/tools/cigale/database_builder/filters/jwst/'
image_directory = 'Data_files/Image_files/v0p3p2/'
print(f'Current default filter data directory {filter_directory}')
print(f'Current default image directory {image_directory}')
print("Wavelength-sorted lists of files saved to variables 'filter_files' and 'image_files'")
print("Regenerate sorted lists using 'image_files, filter_files = generate_list_of_files(filter_directory, image_directory)'")

with open("Data_files/misc_data/jwst_filter_means.pkl", "rb") as file:
    jwst_means = pickle.load(file)
print('JWST filter mean wavelengths stored as dictionary, called using jwst_means["F115W"]')

with open("Data_files/misc_data/jwst_pivots.pkl", "rb") as file:
    jwst_pivots = pickle.load(file)
print('JWST filter pivot wavelengths stored as dictionary, called using jwst_pivots["F115W"]')
def extract_filter_name(filepath):
    """
    Extracts JWST-style filter names from a file path.
    Matches F + digits + {W, M, N}.
    """
    match = re.search(r'[Ff](\d+)([WMNwmn])', filepath)
    if match:
        return f"F{match.group(1).upper()}{match.group(2).upper()}"
    return None


def get_filter_number(filter_name):
    '''extracts numbers from filter name (drops F, N, W, etc from the ends)
    -------------
    
    Parameters
    -------------
    filter_files : type = list - list of filter names
    
    Returns
    -------------
    filter name as string
    '''   
    match = re.search(r'[A-Za-z](\d+)[A-Za-z]', filter_name)  # Numbers between ANY letters
    return int(match.group(1)) if match else 0

def generate_list_of_files(filter_directory, image_directory):
    '''cross-matches files in filter_directory to images in image_directory, sorts by filter number.
    Filters will be duplicated if multiple image files for the same filter are supplied.
    -------------
    
    Parameters
    -------------
    filter_directory - type = string : Must be followed by a folder /miri and /nircam to differentiate detectors
    image_directory - type = string : All files with .fits extention will be grabbed.
    
    Returns
    -------------
    list of arrays, first entry is the image file array, second is the filter file array, both sorted by filter numer (in name)
    '''   
    #filter_directory = '/d/crow1/tools/cigale/database_builder/filters/jwst/'
    path = ['nircam', 'miri']
    filter_files = np.concatenate([glob.glob(os.path.join(filter_directory + file_path, "*.dat")) for file_path in path])
    #image_directory = 'Data_files/Image_files'
    image_files = glob.glob(os.path.join(image_directory, "*.fits"))
    # Initialize aligned lists
    image_file_array = []
    filter_file_array = []
    
    # Loop through .fits files and find matching .dat files
    for fits_file in image_files:
        fits_filter = extract_filter_name(fits_file)

        if not fits_filter:
            continue  # Skip if no filter name found
        
        # Search for matching .dat file
        for dat_file in filter_files:
            dat_filter = extract_filter_name(dat_file)
            if dat_filter == fits_filter:
                image_file_array.append(fits_file)
                filter_file_array.append(dat_file)
                break  # Stop searching after first match
    filter_name_array = [f.split("/")[-1] for f in filter_file_array] #TJ generate array of just the filter names
    filter_numbers = np.array([get_filter_number(file) for file in filter_name_array]) #TJ generate array of just filter numbers
    sort_indices = np.argsort(filter_numbers) #TJ sort by these numbers
    sorted_filter_names = np.array(filter_file_array)[sort_indices]
    sorted_image_files = np.array(image_file_array)[sort_indices]
    return sorted_image_files, sorted_filter_names

image_files, filter_files = generate_list_of_files(filter_directory, image_directory)




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

    try:
        units = flux_array.unit
    except:
        units = None
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


def fit_voigt_to(wavelength_of_feature, tolerance, wavelength_array, flux_array, type = True, show_plot = False, feature_idx_width = 6):
    '''Fits voigt profile to feature nearest to given wavelength.


    need to add backup trial

    -------------

    Parameters
    -------------
    wavelength_of_feature : type = length quantity - wavelength closest to feature center
    tolerance : type = integer - wavelength indices the center of the feature can be away and still achieve a tag of 2
    wavelength_array :  type = float - array of wavelengths including features value of meters (no units though)
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
    
    center_idx = np.argmin(np.abs(wavelength_array - wavelength_of_feature.to(u.m).value)) #TJ assign the center index as the closest wavelength to the expected wavelength
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
    
    bounds = ([min(amp_bounds), wavelength_array[center_idx-tolerance], 0, 0],
              [max(amp_bounds), wavelength_array[center_idx+tolerance], np.inf, np.inf])
    
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

def load_and_sort_convolved_Karin_spectrum(file_path):
    '''import data and sort by wavelength from very particularly structured file. Karin's raw file has some wavelengths
    that are out of order and screw up plotting.
    -------------
    
    Parameters
    -------------
    file_path : type = str - path to file with data
    
    Returns
    -------------
    structured array ('wavelength', 'F_nu', 'uncertainty') where F_nu and uncertainty are in W/m2/Hz
    '''    
    with open(file_path, 'r') as file:
        header = file.readline().strip().split()
        #TJ check first line structure for compliance
        if ((len(header) == 3) & (type(try_float(header[0])) == type(0.1)) & (type(try_float(header[1])) == type(0.1)) & (type(try_float(header[2])) == type(0.1))):
            data_list = []
            data_list.append((try_float(header[0])*1e-6, try_float(header[1])*1e-20, try_float(header[2])*1e-20))
            aperture_area_sr = (np.pi * (((0.75*u.arcsec).to(u.rad))**2)).value
            for line in file:
                parts = line.strip().split(maxsplit=3)
                
                #TJ Convert numeric columns to floats
                wavelength = float(parts[0])*1e-6 #TJ float required for sorting
                intensity = try_float(parts[1])*1e-20 * aperture_area_sr
                uncertainty = try_float(parts[2])*1e-20 * aperture_area_sr
                
                data_list.append((wavelength, intensity, uncertainty))
        
            #TJ Define dtype with notes as string
            dtype = [
                ('wavelength', float),
                ('F_nu', float),
                ('uncertainty', float),
            ]
            
            data = np.array(data_list, dtype=dtype)
            sorted_data = np.sort(data, order=['wavelength'])  #TJ Sort by wavelength
            
            return sorted_data
        else:
            print('''File format is not as expected. Should be 3 columns no header, if not, see "import_data_and_sort_by_wavelength"
            function from Flux_calibration notebook''')
            return None


def is_loc_in_IFU(loc, IFU_file):
    """
    Check if a given coordinate lies within the spatial footprint of an IFU cube.

    Parameters
    ----------
    coord : list [ra, dec] or SkyCoord
        Sky position to check. RA/Dec in degrees if list.
    fits_file : str
        Path to the IFU FITS file.

    Returns
    -------
    bool
        True if the coordinate lies within the cube's spatial coverage.
    """
    # Convert to SkyCoord if needed
    if not isinstance(loc, SkyCoord):
        loc = SkyCoord(loc[0], loc[1], unit='deg')

    # Load WCS and shape from the SCI extension
    with fits.open(IFU_file) as hdul:
        sci_header = hdul['SCI'].header
        sci_data = hdul['SCI'].data
        wcs = WCS(sci_header)
        shape = sci_data.shape  # (nz, ny, nx)

    # Convert sky coordinate to pixel coordinates (ignore spectral axis)
    x, y = skycoord_to_pixel(loc, wcs, origin=0)

    # Get spatial shape
    _, ny, nx = shape
    return (0 <= x < nx) and (0 <= y < ny)

def which_fits(filter_file, list_of_fits):
    '''open the filter file, determine the range of wavelengths needed to compute synthetic flux through it, return which fits files
    are needed for this particular filter. This is to save time not convolving cubes we dont need.
    -------------
    
    Parameters
    -------------
    filter_file : type = str - string to location of filter file that we are interested in.
    list_of_fits : type = list - list of strings to the IFU fits files that you want to check
    
    Returns
    -------------
    needed_fits : type = list - list of strings to the fits files that are actually needed
    '''   
    filter_data = []
    with open(filter_file, 'r') as f:
            header = f.readline().strip().split()
            for line in f:
                data_line = line.strip().split()
                filter_data.append(data_line)
            
    header, filter_T = filter_data[:2], np.array(filter_data[2:])

    wl = [try_float(filter_T[i,0])*1e-10 for i in range(len(filter_T))]
    T = [try_float(filter_T[i,1]) for i in range(len(filter_T))]
    
    min_wl, max_wl = min(wl), max(wl)
    needed_fits = []
    entirely_in = []
    for file in list_of_fits:
        cube = SpectralCube.read(file, hdu='SCI')
        wavelength = cube.spectral_axis
        if (wavelength[0].value*1e-6 < max_wl) and (wavelength[-1].value*1e-6 > min_wl):
            needed_fits.append(file)
            if ((wavelength[0].value*1e-6 < min_wl) and (wavelength[-1].value*1e-6 > max_wl)):
                entirely_in.append(True)
            else:
                entirely_in.append(False)
    needed_fits = np.array(needed_fits)

    if (sum(entirely_in) == 1):
        return needed_fits[entirely_in]
    elif ((len(needed_fits) > 1) & (sum(entirely_in) == 0)):
        print(f'More than one IFU file is needed for filter {extract_filter_name(filter_file)}')
        return needed_fits
    elif ((len(needed_fits) > 1) & (sum(entirely_in) > 1)):
        print(f'More than one IFU file could be used for filter {extract_filter_name(filter_file)}')
        return needed_fits[0]

def full_coverage(filter_name, IFU_file):
    '''Checks if the IFU has full filter converage or not'''
    filter_coverage = get_filter_wl_range(filter_name)
    cube = SpectralCube.read(IFU_file, hdu = 'SCI')
    cube_range = [cube.spectral_axis[0], cube.spectral_axis[-1]]
    if filter_coverage[0] < cube_range[0]:
        return 'missing shorter'
    if filter_coverage[1] > cube_range[1]:
        return 'missing longer'
    else:
        return 'good'

def is_aperture_fully_covered(IFU_file, image_file, loc, radius):
    '''
    Check if a circular aperture is fully within imaged regions for both IFU and image files.
    
    Parameters:
    -----------
    IFU_file : str
        Path to the IFU FITS file (must have WCS and SCI extension).
    image_file : str
        Path to the image FITS file (must have WCS and valid data).
    loc : tuple (ra, dec)
        Sky coordinates of the aperture center in degrees.
    radius : float
        Aperture radius in arcseconds.
        
    Returns:
    --------
    tuple (bool, bool)
        (IFU_fully_covered, image_fully_covered)
        True if aperture is fully within imaged regions for each file.
    '''
    def _check_coverage(file, ext):
        # Open file and get data + WCS
        with fits.open(file) as hdul:
            data = hdul[ext].data
            header = hdul[ext].header
            wcs = WCS(header)
            
            # Handle 3D IFU cubes (use first wavelength slice)
            if data.ndim == 3:
                data = data[0]
            
            # Create coverage mask (1=imaged, 0=NaN/unimaged)
            coverage_mask = np.where(np.isnan(data) | (data == 0), 0, 1)
            
            # Convert sky coordinates to pixel coordinates
            loc_sky = SkyCoord(ra=loc[0] * u.deg, dec=loc[1] * u.deg, frame='icrs')
            x_center, y_center = wcs.celestial.all_world2pix(loc_sky.ra.deg, loc_sky.dec.deg, 0)
            
            # Calculate pixel scale (arcsec/pixel)
            try:
                pixel_scale = np.abs(header['CDELT1']) * 3600  # deg -> arcsec
            except KeyError:
                pixel_scale = np.sqrt(header['PIXAR_A2'])  # Fallback for JWST files
            
            radius_pix = radius.value / pixel_scale
            # Measure coverage
            aperture = CircularAperture((x_center, y_center), r=radius_pix)
            phot_table = aperture_photometry(coverage_mask, aperture)
            measured_area = phot_table['aperture_sum'][0]
            expected_area = np.pi * (radius_pix ** 2)
            
            # Allow 1-pixel tolerance for edge effects
            return np.isclose(measured_area, expected_area, atol=1.0)

    # Check IFU file (SCI extension)
    ifu_covered = _check_coverage(IFU_file, ext='SCI')
    
    # Check image file (primary HDU or SCI)
    try:
        image_covered = _check_coverage(image_file, ext=0)  # Try primary HDU
    except (KeyError, AttributeError):
        image_covered = _check_coverage(image_file, ext='SCI')  # Fallback to SCI
    
    return (ifu_covered, image_covered)

def find_max_radius(IFU_file, image_file, loc, min_radius=0.1*u.arcsec, max_radius=10.0*u.arcsec, tolerance=0.01*u.arcsec):
    """
    Find the maximum aperture radius (arcsec) fully covered in both IFU and image files.
    Uses binary search between min_radius and max_radius.
    
    Parameters:
    -----------
    IFU_file : str
        Path to IFU FITS file.
    image_file : str
        Path to image FITS file.
    loc : tuple (ra, dec)
        Sky coordinates in degrees.
    min_radius : float
        Minimum aperture radius to test (arcsec).
    max_radius : float
        Maximum aperture radius to test (arcsec).
    tolerance : float
        Precision threshold for convergence (arcsec).
        
    Returns:
    --------
    float
        Maximum fully covered radius (arcsec), or 0 if no valid radius found.
    """
    def _is_covered(radius):
        ifu_ok, image_ok = is_aperture_fully_covered(IFU_file, image_file, loc, radius)
        return ifu_ok and image_ok
    
    # Binary search
    best_radius = 0.0
    while max_radius - min_radius > tolerance:
        mid_radius = (min_radius + max_radius) / 2
        if _is_covered(mid_radius):
            best_radius = mid_radius
            min_radius = mid_radius  # Try larger radii
        else:
            max_radius = mid_radius  # Try smaller radii
    IFU_pix_scale = (fits.open(IFU_file)['SCI'].header['CDELT2']*u.deg).to(u.arcsec)
    image_pix_scale = (fits.open(image_files[0])['SCI'].header['CDELT2']*u.deg).to(u.arcsec)
    return (best_radius if best_radius > 0 else 0.0), best_radius/IFU_pix_scale,  best_radius/image_pix_scale


def get_filter_data(filter_name, filter_file_list = filter_files):
    filter_file = [filter_filepath for filter_filepath in filter_file_list if extract_filter_name(filter_filepath) == filter_name][0]

    filter_data = [] #TJ initialize filter data
    with open(filter_file, 'r') as f: #TJ extract filter data (still has headers)
        header = f.readline().strip().split()
        for line in f:
            data_line = line.strip().split()
            filter_data.append(data_line)
    if len(filter_data) < 2:
        print(f"Filter file {filter_file} seems empty or malformed.")
        return None
    
    header, filter_T = filter_data[:2], np.array(filter_data[2:]) #TJ separate filter header from data
    filter_wl = np.array([try_float(row[0]) * 1e-10 for row in filter_T])*u.m #TJ separate filter wavelengths from transmission
    filter_trans = np.array([try_float(row[1]) for row in filter_T])
    return filter_wl, filter_trans



def get_IFU_spectrum(IFU_filepath, loc, radius, replace_negatives = False):
    '''extract spectrum from IFU file with aperture of radius, centered at ra,dec = loc
    -------------
    
    Parameters
    -------------
    IFU_filepath : type = str - string to location of IFU fits file
    loc : type = list - ra, dec in degrees or SkyCoord object
    radius : type = float - radius of aperture, must have units attached (like u.deg or u.arcsecond)
    replace_negatives (optional, defaults to nothing) : type = float : replace negative fluxes with this float times the smallest positive flux value, specify             as None to leave as negative values
    Returns
    -------------
    structured array with entries for "wavelength" (m), "F_nu" (W/m2/Hz), "frequency" (Hz), and "F_lambda" (W/m2/m)
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
        print(f'A total of {nan_detected} NaN were detected within {radius} in {IFU_filepath.split("/")[-1]} over {len(cube.spectral_axis)} WLs')
    wavelength = cube.spectral_axis.to(u.m)
    flux_density_spectrum = np.array(flux_density_spectrum)


    if replace_negatives is not False:
        min_positive = min(flux_density_spectrum[flux_density_spectrum > 0])
        flux_density_spectrum[flux_density_spectrum < 0] = replace_negatives*min_positive  #TJ replace negative numbers with a very small positive value
    frequency = (c / wavelength).to(u.Hz)

    F_lambda = (flux_density_spectrum*u.W/(u.m**2 * u.Hz) * c / wavelength ** 2).to(u.W / (u.m ** 2 * u.m))

    spectrum = {
        'wavelength': wavelength,  # m
        'frequency': frequency,  # Hz
        'F_nu': flux_density_spectrum * u.W/(u.m**2 * u.Hz),  # W / m^2 / Hz
        'F_lambda': F_lambda  # W / m^2 / m
    }

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

def get_Fnu_transmission(Fnu_array, wl_array, transmission_array, trans_wl_array, warnings = True):
    '''get expected flux through filter. Assumes Fnu array is in W/m2/Hz and wl array is in meters. Otherwise, units will be weird.
    -------------
    
    Parameters
    -------------
    Fnu_array : type = array - array of flux density values with units
    wl_array : type = array - array of wavelength values for the corresponding Fnu_array values (should be in meters)
    transmission_array : type = array - array of unitless transmission coefficient
    trans_wl_array : type = array - array of wavelength values for the corresponding transmission values (should be in meters)

    
    Returns
    -------------
    total_flux : type = float - Ideally in units of W/m2
    '''
    try:
        Fnu_units = Fnu_array.unit
    except:
        print('first argument of get_Fnu_transmission() must have units')
        return None
    if ((trans_wl_array[0] < wl_array[0]) or (trans_wl_array[-1] > wl_array[-1])): #TJ Check if wavelengths are compatible with filter
        if warnings:
            print(f'filter goes from {trans_wl_array[0]} to {trans_wl_array[-1]}, but provided Fnu array goes from {wl_array[0]} to {wl_array[-1]}')
        idx_start = np.searchsorted(trans_wl_array, wl_array[0], side='left')
        idx_end = np.searchsorted(trans_wl_array, wl_array[-1], side='right')
        
        # Expand by one index if possible
        idx_start = max(0, idx_start - 1)  # Include one lower index
        idx_end = min(len(trans_wl_array), idx_end + 1)  # Include one higher index
        
        # Slice transmission data
        trans_wl_array = trans_wl_array[idx_start:idx_end]
        transmission_array = transmission_array[idx_start:idx_end]
    if len(trans_wl_array) == 0:
        raise ValueError("No overlap between flux wavelengths and filter transmission curve")
    #TJ convert all arrays to numpy arrays for better indexing and convert to MKS units

    try:
        Fnu_array = Fnu_array.to(u.W/(u.m**2 * u.Hz * u.sr))
    except:
        Fnu_array = Fnu_array.to(u.W / (u.m ** 2 * u.Hz))
    wl_array = wl_array.to(u.m)
    Fnu_array = np.array(Fnu_array)
    wl_array = np.array(wl_array)
    transmission_array = np.array(transmission_array)
    trans_wl_array = np.array(trans_wl_array)

    
    #TJ Convert wavelength to frequency, reverse so freq increases left to right
    spec_freq_array = c / wl_array[::-1]
    Fnu_array = Fnu_array[::-1]
    trans_freq_array = c / trans_wl_array[::-1]
    transmission_array = transmission_array[::-1]

    #TJ Interpolate Fnu onto the transmission frequency grid
    #TJ this is because jwst transmission arrays are averages over BW widths which are much coarser than Fnu is.
    interp_Fnu = np.interp(trans_freq_array, spec_freq_array, Fnu_array)
    
    
    weight = transmission_array / trans_freq_array #TJ weight the numerator and denominator by T *d_nu over nu for integration
    numerator = np.trapz(interp_Fnu * weight, trans_freq_array)#TJ perform integration
    denominator = np.trapz(weight, trans_freq_array)
    ab_mean_flux = numerator / denominator
    # Numerator: Fν * Transmission / nu integrated over frequency

    return ab_mean_flux*Fnu_units

def get_image_flux(image_file, loc, radius, replace_negatives = False):
    '''extract flux from image file with aperature of radius, centered at ra,dec = loc
    -------------
    
    Parameters
    -------------
    image_file : type = str - string to location of image fits file
    loc : type = list - ra, dec in degrees or SkyCoord object
    radius : type = float - radius of aperture, must have units attached (like u.deg or u.arcsecond)
    
    Returns
    -------------
    flux_density observed through filter
    '''
    #TJ assign location as SkyCoord object
    if type(loc) == list:
        spatial_coords = SkyCoord(ra=loc[0]*u.deg, dec=loc[1]*u.deg)
    elif type(loc) == SkyCoord:
        spatial_coords = loc
    else:
        print('loc is not a list of ra, dec and it is not a SkyCoord object.')
        return None
    hdul = fits.open(image_file) #TJ load file
    header = hdul['SCI'].header
    units = header['BUNIT']
    if units == 'MJy/sr':
        original_units = u.MJy/u.sr
        data = ((hdul['SCI'].data*original_units).to(u.W/(u.m**2 *u.Hz* u.sr))) #TJ convert flux density to mks units
        units = u.W/(u.m**2 * u.Hz * u.sr)
    elif units == "erg / (s cm2)":
        original_units = u.erg/(u.s * u.cm**2)
        if header['CDELT1'] != header['CDELT2']:
            print('Pixels are not square!!!!!')
            return None
        pixel_size = header['PIXAR_SR']*u.sr
        data = (((hdul['SCI'].data*original_units)/(pixel_size)).to(u.W/(u.m**2 * u.sr)))
        units = u.W/(u.m**2 * u.sr)
    else:
        print(f'need to add {units} unit support for function "get_image_flux"')
        return None
    
    if replace_negatives is not False:
        min_positive = min(data[data>0])
        data[data<0] = replace_negatives*min_positive

    pix_area = header["PIXAR_SR"]*u.sr #TJ define the angular size of a pixel in staradians
    wcs = WCS(header) #TJ read in the world coordinate system
    radius_pixels = (radius).to_value(u.deg) / abs(header['CDELT2']) #TJ get the radius of the aperture in number of pixels
    #TJ Convert RA/Dec to pixel coordinates
    x, y = wcs.all_world2pix(spatial_coords.ra.deg, spatial_coords.dec.deg, 0)
    aperture = CircularAperture((x, y), r = radius_pixels)

    #TJ Perform aperture photometry
    phot_result = aperture_photometry(data, aperture)
    total_flux = phot_result['aperture_sum'][0]*pix_area  #TJ the result is in pixel units, multiply by steradians per pixel to get units right
    
    return total_flux


def stitch_spectra(fits_files, loc, radius, anchor_idx=0, replace_negatives=False):
    """
    Stitch a list of IFU spectra keeping the spectrum at `anchor_idx` fixed.
    Each other spectrum is shifted (additive offset) to match the anchor/combined
    spectrum in the overlapping wavelength region before being prepended/appended.

    Parameters
    ----------
    fits_files : list of str (can also be two dictionaries with keys for "wavelength" and "F_nu"
    loc, radius : passed to get_IFU_spectrum(...)
    anchor_idx : int
        Index in fits_files of the spectrum to keep fixed (anchor).
    replace_negatives : bool
        Passed through to get_IFU_spectrum.

    Returns
    -------
    combined : dict with keys 'wavelength' (1D np.array) and 'F_nu' (1D np.array)
    """
    try:
        anchor = fits_files[anchor_idx]
        test = anchor['wavelength'][0]
    except:
        anchor = get_IFU_spectrum(fits_files[anchor_idx], loc, radius, replace_negatives=replace_negatives) #TJ extract anchor spectrum

    base_wl = np.asarray(anchor['wavelength'].value).astype(float) #TJ convert to np array for better indexing
    base_flux = np.asarray(anchor['F_nu'].value).astype(float)
    base = {'wavelength': base_wl, 'F_nu': base_flux} #TJ save as dictionary of np arrays

    for i in range(anchor_idx - 1, -1, -1): #TJ start from anchor index and go backward to zero
        print(f"Stitching LEFT: file {i} → anchor")
        try:
            cur = fits_files[i]
            test = cur['wavelength']
        except:
            cur = get_IFU_spectrum(fits_files[i], loc, radius, replace_negatives=replace_negatives) #TJ extract new spectrum to be stitched
        cur_wl = np.asarray(cur['wavelength'].value).astype(float)
        cur_flux = np.asarray(cur['F_nu'].value).astype(float)
        cur = {'wavelength': cur_wl, 'F_nu': cur_flux}

        base = _stitch_base_with_new(base, cur, side='left') #TJ stitch the two sections together

    for i in range(anchor_idx + 1, len(fits_files)): #TJ start from anchor index and go forwards to max index
        print(f"Stitching RIGHT: anchor → file {i}")
        try:
            cur = fits_files[i]
            test = cur['wavelength']
        except:
            cur = get_IFU_spectrum(fits_files[i], loc, radius, replace_negatives=replace_negatives) #TJ load the new spectrum to be stitched
        cur_wl = np.asarray(cur['wavelength'].value).astype(float)
        cur_flux = np.asarray(cur['F_nu'].value).astype(float)
        cur = {'wavelength': cur_wl, 'F_nu': cur_flux}

        base = _stitch_base_with_new(base, cur, side='right') #TJ stitch onto existing spectrum

    print(f'Combined spectrum: {base["wavelength"][0]} -- {base["wavelength"][-1]}')
    return base


def _stitch_base_with_new(base, new, side='right'):
    """
    Shift `new` to match `base` in overlap, then concatenate.
    - side='right': new lies at (or mostly) longer wavelengths than base -> append
    - side='left' : new lies at (or mostly) shorter wavelengths than base -> prepend

    The function always shifts `new` (not `base`) so that `base` remains fixed.
    """
    bw, bi = base['wavelength'], base['F_nu'] #TJ extract wavelength and intensity arrays from dictionaries
    nw, ni = new['wavelength'], new['F_nu']

    if bw.size == 0: #TJ check that arrays are not empty
        return new.copy()
    if nw.size == 0:
        return base.copy()

    overlap_min = max(bw[0], nw[0]) #TJ calculate edge values for overlapping region
    overlap_max = min(bw[-1], nw[-1])

    mask_b = (bw >= overlap_min) & (bw <= overlap_max) & np.isfinite(bi) #TJ create mask in base array to select overlapping region. Ignore nans in flux array
    mask_n = (nw >= overlap_min) & (nw <= overlap_max) & np.isfinite(ni)
    
    offset = 0.0 #TJ initialize offset to be zero

    if np.count_nonzero(mask_b) > 0 and np.count_nonzero(mask_n) >= 2: #TJ need overlapping region to be at least two points to be able to interpolate
        try: #TJ try to interpolate new spectrum's intensity into base spectrums wavelengths
            interp_ni_on_b = np.interp(bw[mask_b], nw[mask_n], ni[mask_n])
            diffs = bi[mask_b] - interp_ni_on_b #TJ create array of differences between interpolated new and base intensities
            offset = np.nanmedian(diffs) #TJ use the median value of that differences array as the offset
        except Exception:
            offset = 0.0 #TJ if that fails, go back to zero as the offset
    else:
        #TJ this should never trigger
        print('This print statement should not trigger, check the overlap between regions')
        nmatch = min(10, bw.size, nw.size) #TJ use the edge 10 values to estimate offset if there is no overlap
        if nmatch >= 1:
            if side == 'right':
                #TJ match base's right edge to new's left edge
                base_edge_med = np.nanmedian(bi[-nmatch:]) #TJ grab the median of the base array's rightmost 10 values
                new_edge_med = np.nanmedian(ni[:nmatch]) #TJ grab the median of the new array's leftmost 10 values
                offset = base_edge_med - new_edge_med #TJ offset is the difference in the medians (note the difference between median of differences)
            else:  #TJ repeat with opposite sides for the leftward stitch
                base_edge_med = np.nanmedian(bi[:nmatch])
                new_edge_med = np.nanmedian(ni[-nmatch:])
                offset = base_edge_med - new_edge_med
        else:
            offset = 0.0

    ni_corr = ni + offset #TJ new intensity is corrected by the offset

    #TJ concatenate while keeping base intact and only adding the non-overlapping portion of new
    if side == 'right':
        #TJ find indices to keep in the new array
        keep_idx = nw > bw[-1]
        if np.any(keep_idx):
            new_wl = np.concatenate([bw, nw[keep_idx]])
            new_flux = np.concatenate([bi, ni_corr[keep_idx]])
        else:
            #TJ if new array is entirely contained within the previous array, do nothing, just copy
            new_wl, new_flux = bw.copy(), bi.copy()
    else:  #TJ repeat with other side
        # Prepend new wavelengths strictly before base start
        keep_idx = nw < bw[0]
        if np.any(keep_idx):
            new_wl = np.concatenate([nw[keep_idx], bw])
            new_flux = np.concatenate([ni_corr[keep_idx], bi])
        else:
            new_wl, new_flux = bw.copy(), bi.copy()

    # Ensure final arrays are sorted ascending in wavelength (should already be, but safe)
    if not np.all(np.diff(new_wl) > 0):
        print('Something weird happened and now the wavelengths are not monotonically ascending!')

    return {'wavelength': new_wl, 'F_nu': new_flux}
jwst_pivot_wavelengths = {
    'F090W': 0.902e-6,
    'F115W': 1.154e-6,
    'F140M': 1.405e-6,
    'F150W': 1.501e-6,
    'F162M': 1.627e-6,
    'F164N': 1.645e-6,
    'F182M': 1.845e-6,
    'F187N': 1.874e-6,
    'F200W': 1.988e-6,
    'F210M': 2.096e-6,
    'F212N': 2.121e-6,
    'F250M': 2.503e-6,
    'F277W': 2.776e-6,
    'F300M': 2.996e-6,
    'F322W2': 3.247e-6,
    'F323N': 3.237e-6, 
    'F335M': 3.362e-6,  
    'F356W': 3.565e-6, 
    'F360M': 3.623e-6, 
    'F405N': 4.053e-6,
    'F410M': 4.083e-6,
    'F430M': 4.281e-6,
    'F444W': 4.402e-6,
    'F460M': 4.630e-6,
    'F466N': 4.654e-6,
    'F470N': 4.708e-6,
    'F480M': 4.817e-6,
    
    # MIRI Filters (from first table)
    'F560W': 5.635e-6,
    'F770W': 7.639e-6,
    'F1000W': 9.953e-6,
    'F1130W': 11.309e-6,
    'F1280W': 12.810e-6,
    'F1500W': 15.064e-6,
    'F1800W': 17.984e-6,
    'F2100W': 20.795e-6,
    'F2550W': 25.365e-6
}

def get_filter_wl_range(filter):
    '''Use the filter files to determine what wavelength range we need for each filter
    -------------
    
    Parameters
    -------------
    filter : type = str - string describing the filter name (case sensitive), for example "F335M"

    Returns
    -------------
    Path to newly convolved file as a string
    '''   
    filter_files = ['/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F115W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F140M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F150W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F164N.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F182M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F187N.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F200W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F210M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F212N.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F250M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F300M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F335M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F360M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F405N.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F430M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F444W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F560W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F770W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F1000W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F1130W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F1280W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F1500W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F1800W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F2100W.dat']
    filter_file = [filer_filepath for filer_filepath in filter_files if extract_filter_name(filer_filepath).upper() == filter][0]
    filter_data = []
    with open(filter_file, 'r') as f:
        header = f.readline().strip().split()
        for line in f:
            data_line = line.strip().split()
            filter_data.append(data_line)

    header, filter_T = filter_data[:2], np.array(filter_data[2:])
    filter_wl = [try_float(filter_T[i,0])*1e-10 for i in range(len(filter_T))]
    return filter_wl[0]*u.m, filter_wl[-1]*u.m

####################################################################



def show_images(image_files, loc, radius, ncols=3, cmap='viridis'):
    """
    Create a collage of cutout images with an aperture overlay.
    
    Parameters
    ----------
    list_of_image_fits_files : list of str
        List of FITS image file paths (must contain SCI extension).
    loc : list, tuple, or SkyCoord
        Location of aperture center, either [RA, Dec] in degrees or a SkyCoord object.
    radius : Quantity
        Aperture radius (must have angular units, e.g. arcsec).
    ncols : int, optional
        Number of columns in the collage (default = 3).
    cmap : str, optional
        Colormap for displaying images (default = 'viridis').
    """
    
    # Make sure loc is SkyCoord
    if not isinstance(loc, SkyCoord):
        loc_sky = SkyCoord(ra=loc[0]*u.deg, dec=loc[1]*u.deg, frame='icrs')
    else:
        loc_sky = loc

    n_images = len(image_files)
    nrows = int(np.ceil(n_images / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), 
                             subplot_kw={'projection': None})
    axes = np.atleast_1d(axes).ravel()  # Flatten in case of 1 row/col
    
    for ax, image_file in zip(axes, image_files):
        # Load FITS
        hdu = fits.open(image_file)['SCI']
        image = hdu.data
        header = hdu.header
        wcs = WCS(header, naxis=2)
        pixel_scale = np.abs(wcs.wcs.cdelt[0]) * 3600  # arcsec/pixel

        # Make cutout
        cutout = Cutout2D(image, position=loc_sky, size=(radius*3, radius*3), wcs=wcs)

        # Convert SkyCoord -> pixel coords
        x_img, y_img = cutout.wcs.world_to_pixel(loc_sky)

        # Plot
        im = ax.imshow(cutout.data, origin='lower', cmap=cmap,
                  norm=ImageNormalize(cutout.data, stretch=AsinhStretch(), 
                                      vmin=0, vmax=np.percentile(cutout.data, 99)))
        ax.add_patch(Circle((x_img, y_img), 
                            (radius.to(u.arcsec).value)/pixel_scale, 
                            ec='red', fc='none', lw=2, alpha=0.7))
        cbar = plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )
        cbar.set_label("Flux (native units)", fontsize=10)
        ax.set_title(image_file.split("/")[-1], fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty panels if n_images doesn’t fill full grid
    for ax in axes[n_images:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_image_and_synth(filter, ifu_fileset, loc, radius, image_files=image_files, color_min_max = [1, 99.5]):
    """
    Show real image and synthetic IFU-derived image side by side.
    Works with either one or two IFU cubes needed for the filter.
    """

    # ------------------------------------------------------------
    # Location handling
    # ------------------------------------------------------------
    if not isinstance(loc, SkyCoord):
        loc_sky = SkyCoord(ra=loc[0] * u.deg, dec=loc[1] * u.deg, frame="icrs")
    else:
        loc_sky = loc

    def nearest_spaxel_map(cube_src, cube_target):
        ny, nx = cube_target.shape[1:]
        y_t, x_t = np.mgrid[:ny, :nx]

        world = cube_target.wcs.celestial.pixel_to_world(x_t, y_t)
        x_s, y_s = cube_src.wcs.celestial.world_to_pixel(world)

        x_s = np.clip(np.round(x_s).astype(int), 0, cube_src.shape[2] - 1)
        y_s = np.clip(np.round(y_s).astype(int), 0, cube_src.shape[1] - 1)

        return y_s, x_s

    # ------------------------------------------------------------
    # Locate real image
    # ------------------------------------------------------------
    real_image_file = [x for x in image_files if extract_filter_name(x) == filter][0]
    
    short_wl, long_wl = [x.value for x in get_filter_wl_range(filter)]

    needed_ifus = []
    for file in ifu_fileset:
        wl = SpectralCube.read(file, hdu="SCI").spectral_axis.to(u.m).value
        if (wl[0] < short_wl) and (wl[-1] > long_wl):
            needed_ifus = [file]
            break
        if (long_wl > wl[0]) and (short_wl < wl[-1]):
            needed_ifus.append(file)
        if len(needed_ifus) > 1:
            break

    cube1 = SpectralCube.read(needed_ifus[0], hdu="SCI")
    cube2 = SpectralCube.read(needed_ifus[1], hdu="SCI") if len(needed_ifus) > 1 else None
    if cube1.unit == 'MJy / sr' and ((cube2 is None) or (cube2.unit == 'MJy / sr')):
        units = u.MJy / u.sr
    else:
        print(f'need to add {cube1.unit} unit support in "show_image_and_synth" function')

    # ------------------------------------------------------------
    # Base cube quantities
    # ------------------------------------------------------------
    wl1 = cube1.spectral_axis.to(u.m).value
    d1 = (cube1.unmasked_data[:].to(u.W/(u.m**2 * u.Hz * u.sr))).value

    ny, nx = cube1.shape[1:]
    n_pix = ny * nx
    d1 = d1.reshape(len(wl1), n_pix).T

    # ------------------------------------------------------------
    # Stitch spectra if needed
    # ------------------------------------------------------------
    if cube2 is not None:
        wl2 = cube2.spectral_axis.to(u.m).value
        d2 = (cube2.unmasked_data[:].to(u.W/(u.m**2 * u.Hz * u.sr))).value

        y2, x2 = nearest_spaxel_map(cube2, cube1)
        d2 = d2[:, y2, x2].reshape(len(wl2), n_pix).T

        wl_all = np.concatenate([wl1, wl2])
        sort_idx = np.argsort(wl_all)
        wl_all = wl_all[sort_idx]

        spec_all = np.concatenate([d1, d2], axis=1)[:, sort_idx]

        wl_min = max(wl1.min(), wl2.min())
        wl_max = min(wl1.max(), wl2.max())
        overlap = (wl_all >= wl_min) & (wl_all <= wl_max)
        both = np.isin(wl_all, wl1) & np.isin(wl_all, wl2) & overlap
        spec_all[:, both] *= 0.5
    else:
        wl_all = wl1
        spec_all = d1

    # ------------------------------------------------------------
    # Synthetic photometry
    # ------------------------------------------------------------
    spec_all = spec_all*u.W/(u.m**2 * u.Hz * u.sr)

    filter_wl, filter_trans = get_filter_data(filter)
    wl_all = wl_all*u.m
    image = np.zeros(n_pix) * spec_all.unit
    for i in range(n_pix):
        image[i] = get_Fnu_transmission(
            spec_all[i], wl_all, filter_trans, filter_wl, warnings=True)
    synth_image = image.reshape(ny, nx)

    # ------------------------------------------------------------
    # Attach WCS to synthetic image
    # ------------------------------------------------------------
    synth_hdu = fits.PrimaryHDU(
        synth_image,
        header=cube1.wcs.celestial.to_header()
    )

    # ------------------------------------------------------------
    # Plot real vs synthetic
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -------------------------
    # REAL IMAGE
    # -------------------------
    hdu = fits.open(real_image_file)["SCI"]
    header = hdu.header
    if header['BUNIT'] == 'MJy/sr':
        units = u.MJy / u.sr
    else:
        print(f'need to add {header["BUNIT"]} unit support in "show_image_and_synth" function')

    real_pix_size = hdu.header['PIXAR_A2']**0.5
    aperture_radius = radius.to(u.arcsec).value / real_pix_size
    cutout_real = Cutout2D(
        (hdu.data*units).to(u.W/(u.m**2 *u.Hz * u.sr)).value,
        position=loc_sky,
        size=(radius * 3, radius * 3),
        wcs=WCS(hdu.header)
    )

    # -------------------------
    # SYNTHETIC IMAGE (WCS CUTOUT)
    # -------------------------

    cutout_synth = Cutout2D(
        (synth_hdu.data*u.W/(u.m**2 *u.Hz * u.sr)).value,
        position=loc_sky,
        size=(radius * 3, radius * 3),
        wcs=WCS(synth_hdu.header)
    )

    # ------------------------------------------------------------
    # Shared normalization (1–99%)
    # ------------------------------------------------------------
    combined = np.concatenate([
        cutout_real.data[np.isfinite(cutout_real.data)],
        cutout_synth.data[np.isfinite(cutout_synth.data)]
    ])

    vmin = np.percentile(combined, color_min_max[0])
    vmax = np.percentile(combined, color_min_max[1])

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("black")
    cmap.set_over("white")
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    pix_scale = np.abs(cutout_synth.wcs.wcs.cdelt[0]) * 3600
    r_ap_pix = radius.to(u.arcsec).value / pix_scale

    # -------------------------
    # PLOT REAL
    # -------------------------
    im0 = axes[0].imshow(
        cutout_real.data,
        origin="lower",
        cmap=cmap,
        norm=norm
    )
    axes[0].set_title(f"{filter} – Real")

    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Flux", fontsize=14)
    cbar0.ax.tick_params(labelsize=12)

    x_r, y_r = cutout_real.wcs.world_to_pixel(loc_sky)
    axes[0].add_patch(
        Circle((x_r, y_r), aperture_radius, edgecolor="red", facecolor="none", linewidth=2)
    )

    # -------------------------
    # PLOT SYNTHETIC
    # -------------------------
    im1 = axes[1].imshow(
        cutout_synth.data,
        origin="lower",
        cmap=cmap,
        norm=norm
    )
    axes[1].set_title(f"{filter} – Synthetic (IFU)")

    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Flux", fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    x_s, y_s = cutout_synth.wcs.world_to_pixel(loc_sky)
    axes[1].add_patch(
        Circle((x_s, y_s), r_ap_pix, edgecolor="red", facecolor="none", linewidth=2)
    )

    # -------------------------
    # Cleanup
    # -------------------------
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def is_filter_relevent(filter, ifu_file):
    '''If a filter's mean wavelength is inside the ifu, returns True
    -------------
    
    Parameters
    -------------
    filter : type = string - name of filter ("F115W")
    ifu_file : type = string - string to location of ifu file
    
    Returns
    -------------
    True if filter's mean wavelength is inside the ifu_file, False if it is not.
    '''
    wls = SpectralCube.read(ifu_file, hdu = 'SCI').spectral_axis.to(u.m)
    short, long = wls[0], wls[-1]
    return (jwst_means[filter] > short) & (jwst_means[filter] < long)
    
def adjust_spectrum(original_ifu, filter_name, image_files, location, radius, adjustment_operation = 'add'):
    '''Takes an ifu file and adjusts the flux through an aperture centered at a location with specified radius.
    -------------
    
    Parameters
    -------------
    original_ifu : type = string (or, see retry=True)- string to location of ifu file
    filter_name : type = string - filter name like "F115W"
    location : type = either SkyCoord or list of [ra, dec] values in degrees - location of center of aperture
    radius : type = angular size - radius of aperture, must have units attached.
    adjustment_operation (optional, defaults to 'add'): type = string - either 'add' or 'multiply' to specify what kind of correction to use
    
    Returns
    -------------
    Structured Numpy array with 'F_nu' and 'wavelength' keys
    '''
    if filter_name is None:
        return get_IFU_spectrum(original_ifu, location, radius, replace_negatives = False), (0 if adjustment_operation == 'add' else 1)
    else:
        image_file = [x for x in image_files if extract_filter_name(x)==filter_name][0]
        raw_data = get_IFU_spectrum(original_ifu, location, radius, replace_negatives = False)
        filter_wl, filter_trans = get_filter_data(filter_name) #TJ this is the transmission vs wavelength function for this filter
        image_flux = get_image_flux(image_file, location, radius, replace_negatives = False) #TJ this is the flux we SHOULD get
        initial_synth_flux = get_Fnu_transmission(raw_data['F_nu'], raw_data['wavelength'], filter_trans, filter_wl, warnings = True) #TJ this is the current synthetic flux we get
        
        if adjustment_operation == 'add':
            correction = image_flux - initial_synth_flux
            raw_data['F_nu'] = raw_data['F_nu'] + correction
            return raw_data, correction #TJ now corrected to match photometry
        elif adjustment_operation == 'multiply':
            correction = image_flux/initial_synth_flux
            raw_data['F_nu'] = raw_data['F_nu']*correction
            return raw_data, correction #TJ now corrected
        else:
            print('adjustment operation not recognized, only "add" or "multiply" are currently implemented')
            return None



def get_largest_filter_within(ifu_file):
    '''Takes an ifu file and selects the filter with the largest bandpass that is entirely within it.
    -------------
    
    Parameters
    -------------
    ifu_file : type = string - string to location of ifu file
    
    Returns
    -------------
    Filter name (ex. "F115W") corresponding to the largest filter entirely contained within the IFU file 
    '''
    filters = [extract_filter_name(x) for x in filter_files if full_coverage(extract_filter_name(x),ifu_file)=="good"]
    if len(filters)<1:
        print(f'No filters entirely within {ifu_file}')
        return None
    else:
        best_filter = filters[np.argmax([(get_filter_wl_range(fil)[1].value - get_filter_wl_range(fil)[0].value) for fil in filters])]
        return best_filter

anchor_filters = ['F150W', 'F200W', 'F444W', 'F1000W', 'F1500W']

def needed_datasets(filter_name, datasets):
    '''returns which ifu_files should be considered when calculating the synthetic flux. If an ifu even slightly overlaps into
    the filter's range it is included.
    -------------
    
    Parameters
    -------------
    filter : type = string - name of filter ("F115W")
    datasets : type = structured array - array with keys for 'wavelength' and 'F_nu'
    
    Returns
    -------------
    Filter name (ex. "F115W") corresponding to the largest filter entirely contained within the IFU file 
    '''
    needed = []
    filter_wl, _ = get_filter_data(filter_name)
    for data in datasets:
        if (filter_wl[0] < data['wavelength'][-1]) & (filter_wl[-1] > data['wavelength'][0]):
            needed.append(data)
    return needed

def merge_datasets(ds1, ds2):
    """
    Merge two structured arrays with 'wavelength' and 'F_nu' keys.
    Handles overlapping regions by averaging intensities, and automatically
    determines which dataset has higher wavelength resolution.
    """
    # Sort by wavelength, just to be safe
    idx1 = np.argsort(ds1['wavelength'])
    idx2 = np.argsort(ds2['wavelength'])

    # Apply to all fields
    for key in ds1:
        ds1[key] = ds1[key][idx1]

    for key in ds2:
        ds2[key] = ds2[key][idx2]

    # Determine wavelength resolutions
    d1_res = np.mean(np.diff(ds1['wavelength']))
    d2_res = np.mean(np.diff(ds2['wavelength']))

    # Assign high- and low-resolution datasets
    if d1_res < d2_res:
        highres, lowres = ds1, ds2
    else:
        highres, lowres = ds2, ds1

    # Determine overlap region
    overlap_start = max(highres['wavelength'][0], lowres['wavelength'][0])
    overlap_end   = min(highres['wavelength'][-1], lowres['wavelength'][-1])

    # Interpolate the lowres data onto highres wavelengths (only inside overlap)
    overlap_mask = (highres['wavelength'] >= overlap_start) & (highres['wavelength'] <= overlap_end)
    interp_flux = np.interp(highres['wavelength'][overlap_mask],
                            lowres['wavelength'], lowres['F_nu'])

    # Combine in overlap by averaging
    merged_overlap_wl = highres['wavelength'][overlap_mask]
    F_nu = 0.5 * (highres['F_nu'][overlap_mask] + interp_flux)

    # Keep the unique non-overlapping parts from both sides
    mask1 = ds1['wavelength'] < overlap_start
    mask2 = ds2['wavelength'] > overlap_end

    full_low_side = {
        'wavelength': ds1['wavelength'][mask1],
        'frequency': ds1['frequency'][mask1],
        'F_nu': ds1['F_nu'][mask1],
        'F_lambda': ds1['F_lambda'][mask1],
    }

    full_high_side = {
        'wavelength': ds2['wavelength'][mask2],
        'frequency': ds2['frequency'][mask2],
        'F_nu': ds2['F_nu'][mask2],
        'F_lambda': ds2['F_lambda'][mask2],
    }
    # Concatenate all pieces and sort
    overlap_block = {
        'wavelength': merged_overlap_wl,
        'F_nu': F_nu,
    }

    overlap_block['frequency'] = (c / overlap_block['wavelength']).to(u.Hz)
    overlap_block['F_lambda'] = (overlap_block['F_nu'] * overlap_block['frequency'] / overlap_block['wavelength']).to(
        u.W / u.m ** 3)

    # --- Concatenate field-by-field ---
    merged = {}

    for key in ['wavelength', 'frequency', 'F_nu', 'F_lambda']:
        merged[key] = np.concatenate([
            full_low_side[key],
            overlap_block[key],
            full_high_side[key],
        ])

    # --- Sort by wavelength ---
    order = np.argsort(merged['wavelength'])

    for key in merged:
        merged[key] = merged[key][order]

    return merged, overlap_block


def get_all_fluxes(filter_files, spec_datasets, image_files, location, radius):
    '''Creates synthetic fluxes for all filters in the files that have wavelengths that span the entire filter.
    For filters that straddle multiple wavelengths, any wavelength inside a filter that has intensity values
    from multiple datasets uses the average intensity from each dataset.
    -------------
    
    Parameters
    -------------
    filter_files : type = list of strings - name of filter ["F115W", "F2100W"]
    datasets : type = list of structured arrays - arrays with keys for 'wavelength' and 'F_nu'
    image_files : type = list of strings - strings to image files
    location : type = either SkyCoord or list of [ra, dec] values in degrees - location of center of aperture
    radius : type = angular size - radius of aperture, must have units attached.
    
    Returns
    -------------
    A dictionary with keys for 'filter_name', 'mean_wl', 'synth_flux', and 'photo_flux'
    '''
    results = {}

    results['filter_name'] = []
    results['mean_wl'] = []
    results['synth_flux'] = []
    results['photo_flux'] = []
    results['wavelength'] = []
    results['F_nu'] = []

    for i, data in enumerate(spec_datasets[1:]):
        if i == 0:
            prior_data = spec_datasets[0]
        prior_data = merge_datasets(prior_data, data)[0]
        
    results['wavelength'].append(prior_data['wavelength'])
    results['F_nu'].append(prior_data['F_nu'])
    
    
    for filter_file in filter_files:
        filter_name = extract_filter_name(filter_file)
        image_file = [x for x in image_files if extract_filter_name(x)==filter_name][0]
        photo_flux = get_image_flux(image_file, location, radius, replace_negatives = False).value
        results['photo_flux'].append(photo_flux)
        filter_wl, filter_trans = get_filter_data(filter_name)
        results['filter_name'].append(filter_name)
        results['mean_wl'].append(jwst_means[filter_name].value)
        needed_data = needed_datasets(filter_name, spec_datasets)
        if len(needed_data) == 0:
            print('no spectral data was found for ', filter_name)
        if len(needed_data)<2:
            synth_flux = get_Fnu_transmission(needed_data[0]['F_nu'], needed_data[0]['wavelength'], filter_trans, filter_wl, warnings = True)
            results['synth_flux'].append(synth_flux.value)
            
        else:
            full_data = merge_datasets(needed_data[0], needed_data[1])[0]
            synth_flux = get_Fnu_transmission(full_data['F_nu'], full_data['wavelength'], filter_trans, filter_wl, warnings = True)
            results['synth_flux'].append(synth_flux.value)

    print('wavelengths: ', results['wavelength'])
    print('Fnu array: ', results['F_nu'])
    print('filter names: ', results['filter_name'])
    print('mean wls: ', results['mean_wl'])
    print('synthetic flux: ', results['synth_flux'])
    print('photoflux: ', results['photo_flux'])

    results['wavelength'] = np.array(results['wavelength'][0])
    results['F_nu'] = np.array(results['F_nu'][0])
    results['filter_name'] = np.array(results['filter_name'])
    results['mean_wl'] = np.array(results['mean_wl'])*u.m
    results['synth_flux'] = np.array(results['synth_flux'])*u.W/(u.m**2 *u.Hz)
    results['photo_flux'] = np.array(results['photo_flux'])*u.W/(u.m**2 *u.Hz)
    print(results)
    return results


def get_overlap_region(ds1, ds2):
    """
    Return only the overlapping wavelength region between two structured arrays
    with 'wavelength' and 'F_nu'. The returned region contains:
        - wavelength grid from the higher-resolution dataset (within overlap)
        - F_nu = average(intensity_highres, interpolated_intensity_lowres)
    """

    # Sort to ensure order
    ds1 = np.sort(ds1, order='wavelength')
    ds2 = np.sort(ds2, order='wavelength')

    # Compute wavelength resolutions
    d1_res = np.mean(np.diff(ds1['wavelength']))
    d2_res = np.mean(np.diff(ds2['wavelength']))

    # Identify high- and low-resolution datasets
    if d1_res < d2_res:
        highres, lowres = ds1, ds2
    else:
        highres, lowres = ds2, ds1

    # Determine numerical overlap bounds
    overlap_start = max(highres['wavelength'][0], lowres['wavelength'][0])
    overlap_end   = min(highres['wavelength'][-1], lowres['wavelength'][-1])

    # If no overlap, return empty structured array
    if overlap_start >= overlap_end:
        return np.recarray(0, dtype=[('wavelength', float), ('F_nu', float)])

    # Mask for high-res wavelengths inside the overlap
    mask = (highres['wavelength'] >= overlap_start) & (highres['wavelength'] <= overlap_end)

    high_wl = highres['wavelength'][mask]
    high_flux = highres['F_nu'][mask]

    # Interpolate lowres intensities onto the highres wavelength grid
    interp_flux = np.interp(high_wl,
                            lowres['wavelength'],
                            lowres['F_nu'])

    # Average intensities
    avg_flux = 0.5 * (high_flux + interp_flux)

    # Return structured array
    overlap = np.rec.fromarrays(
        [high_wl, avg_flux],
        names=('wavelength', 'F_nu')
    )

    return overlap




def compare_photometry(ifu_files, image_files, filter_files, loc, radius, anchor_filters = anchor_filters, correct = True):
    '''docstring in progress, see function for line comments
    


    '''
    temp_filepath = 'Data_files/misc_data/temp_data'
    if os.path.exists(temp_filepath):
        shutil.rmtree(temp_filepath)
    os.makedirs(temp_filepath)
    results = {}
    if loc == [202.5062429, 47.2143358]:
        loc_index = 0
    elif loc == [202.4335225, 47.1729608]:
        loc_index = 1
    elif loc == [202.4340450, 47.1732517]:
        loc_index = 2
    elif loc == [202.4823742, 47.1958589]:
        loc_index = 3
    else:
        loc_index = "?"
    # Create SkyCoord object for spatial coordinates
    if type(loc) == list:
        spatial_coords = SkyCoord(ra=loc[0]*u.deg, dec=loc[1]*u.deg)
    elif type(loc) == SkyCoord:
        spatial_coords = loc
    else:
        print('loc is not a list of ra, dec and it is not a SkyCoord object.')
        return None
    
    results['add_datasets'] = []
    results['mult_datasets'] = []

    results['ifu_files'] = ifu_files
    results['image_files'] = image_files
    
    results['location'] = spatial_coords
    results['loc_idx'] = loc_index
    results['radius'] = radius
    if correct:
        print('adjusting spectra using additive and multiplicative corrections')
    results['add_correction_values'] = []
    results['mult_correction_values'] = []
    
    for i, ifu_file in enumerate(ifu_files):
        if correct:
            mult_data, mult_correction = adjust_spectrum(ifu_file, get_largest_filter_within(ifu_file), image_files, loc, radius, adjustment_operation = 'multiply')
            results['mult_correction_values'].append(mult_correction)
            add_data, add_correction = adjust_spectrum(ifu_file, get_largest_filter_within(ifu_file), image_files, loc, radius, adjustment_operation = 'add')
            results['add_correction_values'].append(add_correction)

            fname = os.path.join(temp_filepath, f"add_grism_{i+1}_of_{len(ifu_files)}.npz")
            np.savez(
                fname,
                wavelength=add_data['wavelength'].value,
                frequency=add_data['frequency'].value,
                F_nu=add_data['F_nu'].value,
                F_lambda=add_data['F_lambda'].value
            )
            fname = os.path.join(temp_filepath, f"mult_grism_{i+1}_of_{len(ifu_files)}.npz")
            np.savez(
                fname,
                wavelength=mult_data['wavelength'].value,
                frequency=mult_data['frequency'].value,
                F_nu=mult_data['F_nu'].value,
                F_lambda=mult_data['F_lambda'].value
            )
            print(f'adjusted {i+1} of {len(ifu_files)}')
        else:
            mult_data, mult_correction = adjust_spectrum(ifu_file, None, image_files, loc, radius, adjustment_operation = 'multiply')
            fname = os.path.join(temp_filepath, f"add_grism_{i + 1}_of_{len(ifu_files)}.npz")
            add_data, add_correction = adjust_spectrum(ifu_file, get_largest_filter_within(ifu_file), image_files, loc,
                                                       radius, adjustment_operation='add')
            results['add_correction_values'].append(add_correction)

            np.savez(
                fname,
                wavelength=add_data['wavelength'].value,
                frequency=add_data['frequency'].value,
                F_nu=add_data['F_nu'].value,
                F_lambda=add_data['F_lambda'].value
            )
            fname = os.path.join(temp_filepath, f"mult_grism_{i + 1}_of_{len(ifu_files)}.npz")
            np.savez(
                fname,
                wavelength=mult_data['wavelength'].value,
                frequency=mult_data['frequency'].value,
                F_nu=mult_data['F_nu'].value,
                F_lambda=mult_data['F_lambda'].value
            )
    add_datasets = []
    mult_datasets = []
    add_files = glob.glob(f'Data_files/misc_data/temp_data/add_grism*')
    mult_files = glob.glob(f'Data_files/misc_data/temp_data/mult_grism*')
    for file in add_files:
        d = np.load(file)
        data = {
            'wavelength': d['wavelength'] * u.m,
            'frequency': d['frequency']*u.Hz,
            'F_nu': d['F_nu']*u.W/(u.m**2 * u.Hz),
            'F_lambda': d['F_lambda']*u.W/(u.m**3)
        }
        results['add_datasets'].append(data)
        add_datasets.append(data)
    for file in mult_files:
        d = np.load(file)
        data = {
            'wavelength': d['wavelength'] * u.m,
            'frequency': d['frequency'] * u.Hz,
            'F_nu': d['F_nu'] * u.W / (u.m ** 2 * u.Hz),
            'F_lambda': d['F_lambda'] * u.W / (u.m ** 3)
        }
        results['mult_datasets'].append(data)
        mult_datasets.append(data)

    print('calculating additive corrected synthetic photometry...')
    add_results = get_all_fluxes(filter_files, add_datasets, image_files, loc, radius)
    print('calculating multiplicative corrected synthetic photometry...')
    mult_results = get_all_fluxes(filter_files, mult_datasets, image_files, loc, radius)
    
    
    print('Compiling results and cleaning up...')
    
    results['filter_names'] = add_results['filter_name']
    results['filter_wavelengths'] = add_results['mean_wl']
    results['add_synthetic_fluxes'] = add_results['synth_flux']
    results['mult_synthetic_fluxes'] = mult_results['synth_flux']
    if np.mean(add_results['photo_flux']) != np.mean(mult_results['photo_flux']):
        print('!!!!!!!!Photo fluxes were not the same in the two datasets! Something has gone wrong')
    results['photo_fluxes'] = add_results['photo_flux']
    if np.mean(add_results['wavelength']) != np.mean(mult_results['wavelength']):
        print('!!!!!!!!!Wavelength arrays were not the same in the two datasets! Something has gone wrong')
    results['wavelength'] = add_results['wavelength']
    results['add_Fnu'] = add_results['F_nu']
    results['mult_Fnu'] = mult_results['F_nu']
    
    shutil.rmtree(temp_filepath)

    return results


def plot_results(results, correction='mult', show_images=[], color_min_max=[1, 99.5]):
    '''


    '''
    # TJ ensure that location is a skycoord
    if not isinstance(results['location'], SkyCoord):
        loc_sky = SkyCoord(ra=results['location'][0] * u.deg, dec=results['location'][1] * u.deg, frame="icrs")
    else:
        loc_sky = results['location']

    # TJ define helper function to match up pixels between multiple IFU files
    def nearest_spaxel_map(cube_src, cube_target):
        ny, nx = cube_target.shape[1:]
        y_t, x_t = np.mgrid[:ny, :nx]

        world = cube_target.wcs.celestial.pixel_to_world(x_t, y_t)
        x_s, y_s = cube_src.wcs.celestial.world_to_pixel(world)

        x_s = np.clip(np.round(x_s).astype(int), 0, cube_src.shape[2] - 1)
        y_s = np.clip(np.round(y_s).astype(int), 0, cube_src.shape[1] - 1)

        return y_s, x_s

    # TJ write title phrase for plot display
    if correction == 'mult':
        method = 'multiplicative correction'
    elif correction == 'add':
        method = 'additive correction'
    else:
        method = 'unrecognized method'

    # TJ create figure axes, fontsizes, marker sizes, colors, initial axis limits, etc
    fig = plt.figure(figsize=(45, 30))
    ax_spec = fig.add_axes((0.05, 0.4, 1, 0.6))
    ax_scat = fig.add_axes((0.05, 0.05, 1, 0.35))
    fontsize_sm = 35
    fontsize_lg = 45
    marker_size = 250
    cube_colors = ['purple', 'blue', 'cyan', 'green', 'orange', 'red', 'pink']
    spec_y_min = 1  # TJ Flux should always be around 10^-20 so setting limits of 0-1 should never be too strict
    spec_y_max = 0
    short_bounds = []
    long_bounds = []

    # TJ load the datasets one by one, plotting in different colors to show separate cubes. Plot averaged overlap regions in black
    for i, dataset in enumerate(results[correction + '_datasets']):

        short_bounds.append(dataset['wavelength'][0])
        long_bounds.append(dataset['wavelength'][-1])
        ax_spec.plot(dataset['wavelength'], dataset['F_nu'], alpha=0.5, color=cube_colors[i], linewidth=5)
        spec_y_min = min(spec_y_min, np.percentile(dataset['F_nu'].value, 1) * 0.5)
        spec_y_max = max(spec_y_max, np.percentile(dataset['F_nu'].value, 98) * 1.5)
        if i > 0:
            overlap_data = merge_datasets(results[correction + '_datasets'][i - 1], dataset)[1]
            ax_spec.plot(overlap_data['wavelength'], overlap_data['F_nu'], alpha=0.5, color='black')

    # TJ plot the entire spectrum on the scatter plot below in white to force scale to be the same as above.
    ax_scat.plot(results['wavelength'], [1] * len(results[correction + '_Fnu']), color='white', alpha=0)
    ax_spec.scatter(results['filter_wavelengths'], results[correction + '_synthetic_fluxes'], marker='*', s=marker_size,
                    color='black')
    ax_spec.scatter([], [], marker='*', s=marker_size, color='black', label='Synth')
    ax_spec.scatter(results['filter_wavelengths'], results['photo_fluxes'], marker="o", s=marker_size, color='black')
    ax_spec.scatter([], [], marker="o", s=marker_size, color='black', label='Photo')

    # TJ plot the horizontal filter coverage in black
    for i, filter in enumerate(results['filter_names']):
        filter_short_wl, filter_long_wl = [x.value for x in get_filter_wl_range(filter)]
        ax_spec.hlines(y=results['photo_fluxes'][i].value, xmin=filter_short_wl, xmax=filter_long_wl, color='black',
                       alpha=0.7, linewidth=3)

    # TJ plot the data on the bottom graph
    ax_scat.scatter(results['filter_wavelengths'], results[correction + '_synthetic_fluxes'] / results['photo_fluxes'],
                    s=marker_size, color='black')

    # TJ generate tick labels and sizes
    ax_scat.tick_params(axis='x', which='minor', width=2, length=10, right=True, top=True, direction='in',
                        labelsize=fontsize_sm)
    ax_scat.tick_params(axis='x', which='major', width=3, length=15, right=True, top=True, direction='in',
                        labelsize=fontsize_sm)
    ax_scat.tick_params(axis='y', which='both', width=3, length=15, right=True, top=True, direction='in',
                        labelsize=fontsize_sm)
    ax_scat.set_xlabel('wavelength (m)', fontsize=40)
    ax_scat.set_ylabel('synthetic/photometric flux', fontsize=40)
    ax_spec.tick_params(axis='x', which='minor', width=2, length=10, right=True, top=True, direction='in',
                        labelsize=fontsize_sm)
    ax_spec.tick_params(axis='x', which='major', width=3, length=15, right=True, top=True, direction='in',
                        labelsize=fontsize_sm)
    ax_spec.tick_params(axis='y', which='both', width=3, length=15, right=True, top=True, direction='in',
                        labelsize=fontsize_sm)
    units = results[correction+ '_datasets'][0]['F_nu'].unit
    ax_spec.set_ylabel(f'F_nu {units}', fontsize=40)

    ax_spec.set_title(f"{results['radius']}-radius aperture at location {results['loc_idx']}\nUsing {method}",
                      fontsize=50)

    # TJ set scale to logorithmic on both horizontal axes and the vertical axis
    ax_scat.set_xscale('log')
    ax_spec.set_xscale('log')
    ax_spec.set_yscale('log')

    # TJ initialize filter name label locations
    label_positions = []

    for x, y, name in zip(results['filter_wavelengths'].value,
                          results[correction + '_synthetic_fluxes'] / results['photo_fluxes'],
                          results['filter_names']):
        # TJ plot the horizontal filter coverage on the lower plot.
        filter_short_wl, filter_long_wl = [x.value for x in get_filter_wl_range(name)]
        ax_scat.hlines(y=y, xmin=filter_short_wl, xmax=filter_long_wl, color='black', alpha=0.7, linewidth=3)

        # TJ default offset is 0.05 lower than the scatter point
        y_offset = -0.05

        # TJ label the filters used to calibrate the cubes in red to distinguish them
        if name in anchor_filters:
            color = 'red'
        else:
            color = 'black'
        # TJ transform the entire figure axes to a coordinate system to calculate distances between labels
        x_disp, y_disp = ax_spec.transData.transform((x, y))

        # TJ check if labels are too close to another label
        too_close = False
        for (xx, yy) in label_positions:
            if abs(x_disp - xx) < 20 and abs(y_disp + y_offset - yy) < 5:
                # TJ use 20px horizontal and 5px vertical proximity as threshhold for "too close"
                too_close = True
                break

        # TJ if overlapping, nudge upward instead of downward
        if too_close:
            y_offset = +0.2
            # TJ if the location of the scatter point is less than 0.8, always default to labeling it up instead of down
        if y < 0.8:
            y_offset = +0.4
        # TJ save adjusted label position
        label_positions.append((x_disp, y_disp + y_offset))

        # TJ this filter is extremely close to another filter and should always be above instead of below
        if name == "F182M":
            y_offset = +0.25
        # TJ same with this one
        if name == 'F212N':
            y_offset = +0.25

        # TJ now plot text in chosen coordinates
        ax_scat.text(
            x, y + y_offset,
            name,
            ha="center", va="top",
            fontsize=fontsize_sm, rotation=90, color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5)
        )

    # TJ find the image files that were asked to be displayed within the figure
    show_image_files = [x for x in results['image_files'] if extract_filter_name(x) in show_images]
    # TJ set the locations of the images to display
    image_locations = [(0.05, 0.75, 0.2, 0.2), (0.3, 0.75, 0.2, 0.2), (0.6, 0.41, 0.2, 0.2), (0.8, 0.41, 0.2, 0.2),
                       (0.85, 0.65, 0.18, 0.18)]

    # TJ if show images is a list of filters, then put them into the figure
    for i, (img, title) in enumerate(zip(show_image_files, show_images)):
        # TJ add new axis in the next available location (real image takes one, synth image takes other)
        ax_real = fig.add_axes(image_locations[2 * i])
        ax_synth = fig.add_axes(image_locations[(2 * i) + 1])

        # TJ get real image file for this filter
        real_image_file = [x for x in image_files if extract_filter_name(x) == title][0]
        # TJ get wavelength coverage, for calculating which ifu cubes are needed
        short_wl, long_wl = [x.value for x in get_filter_wl_range(title)]

        # TJ initialize ifu cubes that are needed for this wavelength coverage
        needed_ifus = []
        for file in results['ifu_files']:
            wl = SpectralCube.read(file, hdu="SCI").spectral_axis.to(u.m).value
            # TJ if filter is even partially within a cube, add it to the list of needed cubes.
            if (long_wl > wl[0]) and (short_wl < wl[-1]):
                needed_ifus.append(file)
            # TJ no filter should ever need more than 2 cubes, so stop checking.
            if len(needed_ifus) > 1:
                break
        # TJ read in the cubes that are needed.
        cube1 = SpectralCube.read(needed_ifus[0], hdu="SCI")
        d1_units = cube1.header["BUNIT"]
        if d1_units == "MJy/sr" or d1_units == "MJy sr-1":
            d1_units = u.MJy/u.sr
        else:
            print(f'Need to add {d1_units} unit support for "plot_results"')
        cube2 = SpectralCube.read(needed_ifus[1], hdu="SCI") if len(needed_ifus) > 1 else None
        d2_units = cube2.header["BUNIT"]
        if d2_units == "MJy/sr" or d1_units == "MJy sr-1":
            d2_units = u.MJy / u.sr
        else:
            print(f'Need to add {d2_units} unit support for "plot_results"')
        # TJ extract wavelength array and remove masked data
        wl1 = cube1.spectral_axis.to(u.m).value
        d1 = cube1.unmasked_data[:].value

        # TJ extract the shape of the cubes (minus the wavelength axis)
        ny, nx = cube1.shape[1:]

        # TJ get number of pixels in first cube
        n_pix = ny * nx
        d1 = d1.reshape(len(wl1), n_pix).T

        # TJ if we need two cubes, we need to stitch them together and get averages for the overlapping region
        if cube2 is not None:

            wl2 = cube2.spectral_axis.to(u.m).value
            d2 = cube2.unmasked_data[:].value

            y2, x2 = nearest_spaxel_map(cube2, cube1)
            d2 = d2[:, y2, x2].reshape(len(wl2), n_pix).T

            # --- decide reference (higher spectral resolution) ---
            dlam1 = np.nanmedian(np.diff(wl1))
            dlam2 = np.nanmedian(np.diff(wl2))

            if dlam1 <= dlam2:
                wl_ref, spec_ref = wl1, d1
                wl_other, spec_other = wl2, d2
            else:
                wl_ref, spec_ref = wl2, d2
                wl_other, spec_other = wl1, d1

            # --- interpolate OTHER cube onto wl_ref ---
            n_ref = len(wl_ref)
            spec_other_interp = np.full((n_pix, n_ref), np.nan)

            for i in range(n_pix):
                f = interp1d(
                    wl_other,
                    spec_other[i],
                    bounds_error=False,
                    fill_value=np.nan
                )
                spec_other_interp[i] = f(wl_ref)

            # --- determine overlap ---
            wl_min = max(wl_ref.min(), wl_other.min())
            wl_max = min(wl_ref.max(), wl_other.max())

            ref_overlap = (wl_ref >= wl_min) & (wl_ref <= wl_max)

            # --- average in overlap ---
            spec_ref_avg = spec_ref.copy()
            both = ref_overlap & np.isfinite(spec_other_interp)

            spec_ref_avg[both] = 0.5 * (
                    spec_ref[both] + spec_other_interp[both]
            )

            # --- non-overlapping wavelengths from other cube ---
            left_mask = wl_other < wl_min
            right_mask = wl_other > wl_max

            wl_left = wl_other[left_mask]
            wl_right = wl_other[right_mask]

            spec_left = spec_other[:, left_mask]
            spec_right = spec_other[:, right_mask]

            # --- build full wavelength grid ---
            wl_all = np.concatenate([wl_left, wl_ref, wl_right])

            # --- build full spectrum ---
            spec_all = np.full((n_pix, len(wl_all)), np.nan)

            i0 = len(wl_left)
            i1 = i0 + len(wl_ref)

            spec_all[:, i0:i1] = spec_ref_avg

            if wl_left.size > 0:
                spec_all[:, :i0] = spec_left

            if wl_right.size > 0:
                spec_all[:, i1:] = spec_right

        else:
            wl_all = wl1
            spec_all = d1

        # TJ extract filter information
        filter_wl, filter_trans = get_filter_data(title)

        print(d1_units)
        spec_all*=d1_units
        wl_all *= u.m
        # TJ initialize image
        image = np.zeros(n_pix) * spec_all[0].unit
        # TJ loop through the pixels applying the filter to each one.
        for i in range(n_pix):
            image[i] = get_Fnu_transmission(spec_all[i], wl_all, filter_trans, filter_wl, warnings=True)

        # TJ reshape the image into a 2d array
        synth_image = image.reshape(ny, nx)

        # TJ assign a WCS coordinate system to the synthetic image
        synth_hdu = fits.PrimaryHDU(synth_image, header=cube1.wcs.celestial.to_header())
        
        # -------------------------
        # TJ FOR THE REAL IMAGE-----
        # -------------------------
        # TJ load real image hdu
        hdu = fits.open(real_image_file)["SCI"]
        # TJ calculate pixel size. If they are not square, print warning.
        if hdu.header['CDELT1'] != hdu.header['CDELT2']:
            print('!!!!!!!Pixels are not square!!!!!!')
            print(f"pixels in one direction are {hdu.header['CDELT1'] * 3600} arcseconds")
            print(f"pixels in other direction are {hdu.header['CDELT2'] * 3600} arcseconds")

        real_pix_size = hdu.header['PIXAR_A2'] ** 0.5
        # TJ calculate how many pixels the aperture radius should be
        aperture_radius = results['radius'].to(u.arcsec).value / real_pix_size
        print('real image_units', hdu.header['BUNIT'])
        if hdu.header['BUNIT'] == "MJy/sr":
            units = u.MJy/u.sr
        else:
            print(f'need to add unit support for {hdu.header["BUNIT"]} in "plot_results"')
        cutout_real = Cutout2D(
            (hdu.data*units).to(u.W/(u.m**2 * u.Hz *u.sr)).value,
            position=loc_sky,
            size=(results['radius'] * 3, results['radius'] * 3),
            wcs=WCS(hdu.header)
        )

        # ----------------------------
        # TJ FOR THE SYNTHETIC IMAGE---
        # ----------------------------

        cutout_synth = Cutout2D(
            (synth_hdu.data).to(u.W/(u.m**2 * u.Hz *u.sr)).value,
            position=loc_sky,
            size=(results['radius'] * 3, results['radius'] * 3),
            wcs=WCS(synth_hdu.header)
        )

        # TJ get pixel scale for the synthetic image and calculate aperture radius in pixels
        pix_scale = np.abs(cutout_synth.wcs.wcs.cdelt[0]) * 3600
        r_ap_pix = results['radius'].to(u.arcsec).value / pix_scale

        # TJ generate min and max flux values for color bar (to avoid a single bright pixel making everything else basically zero color)
        # TJ first, we need to isolate the aperture, we dont care if the bright pixel is outside the aperture
        x0_r, y0_r = cutout_real.wcs.world_to_pixel(loc_sky)
        x0_s, y0_s = cutout_synth.wcs.world_to_pixel(loc_sky)

        # Pixel grids
        yy_r, xx_r = np.indices(cutout_real.data.shape)
        yy_s, xx_s = np.indices(cutout_synth.data.shape)

        # Aperture masks
        ap_mask_real = ((xx_r - x0_r) ** 2 + (yy_r - y0_r) ** 2) <= aperture_radius ** 2
        ap_mask_synth = ((xx_s - x0_s) ** 2 + (yy_s - y0_s) ** 2) <= r_ap_pix ** 2

        # Collect aperture pixels only
        ap_pixels = np.concatenate([
            cutout_real.data[ap_mask_real & np.isfinite(cutout_real.data)],
            cutout_synth.data[ap_mask_synth & np.isfinite(cutout_synth.data)]
        ])

        # Robust statistics
        median = np.median(ap_pixels)
        sigma = np.std(ap_pixels)

        # Sigma limits (tweak if needed)
        nsig_low = 3.0
        nsig_high = 20.0

        vmin = median - nsig_low * sigma
        vmax = median + nsig_high * sigma

        # Safety fallback
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin = np.nanmin(ap_pixels)
            vmax = np.nanmax(ap_pixels)

        # Colormap behavior
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_under("black")
        cmap.set_over("white")

        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

        # -------------------------
        # TJ PLOT THE REAL IMAGE
        # -------------------------
        im0 = ax_real.imshow(
            cutout_real.data,
            origin="lower",
            cmap=cmap,
            norm=norm
        )
        # TJ add title
        ax_real.set_title(f"{title} – Real", fontsize=fontsize_lg)
        # TJ add color bar
        cbar0 = plt.colorbar(im0, ax=ax_real, fraction=0.046, pad=0.04)
        cbar0.set_label("MJy/sr", fontsize=fontsize_lg)
        cbar0.ax.tick_params(labelsize=fontsize_sm)

        x_r, y_r = cutout_real.wcs.world_to_pixel(loc_sky)
        ax_real.add_patch(
            Circle((x_r, y_r), aperture_radius, edgecolor="red", facecolor="none", linewidth=2)
        )

        # -----------------------------
        # TJ PLOT THE SYNTHETIC IMAGE---
        # -----------------------------
        im1 = ax_synth.imshow(
            cutout_synth.data,
            origin="lower",
            cmap=cmap,
            norm=norm
        )
        # TJ add title
        ax_synth.set_title(f"{title} – Synthetic", fontsize=fontsize_lg)
        # TJ add color bar
        cbar1 = plt.colorbar(im1, ax=ax_synth, fraction=0.046, pad=0.04)
        cbar1.set_label("MJy/sr", fontsize=fontsize_lg)
        cbar1.ax.tick_params(labelsize=fontsize_sm)

        x_s, y_s = cutout_synth.wcs.world_to_pixel(loc_sky)
        ax_synth.add_patch(
            Circle((x_s, y_s), r_ap_pix, edgecolor="red", facecolor="none", linewidth=2)
        )

        # TJ add tick marks
        ax_real.set_xticks([])
        ax_real.set_yticks([])
        ax_synth.set_xticks([])
        ax_synth.set_yticks([])

    ymin, ymax = ax_scat.get_ylim()
    text_y_pos = ymin * 1.1
    ax_scat.axhline(y=1, color='gray', linestyle='--', linewidth=4, alpha=0.5)
    ax_scat.axvline(x=5e-6, color='gray', linestyle='--', linewidth=4, alpha=0.7)
    ax_scat.text(4.9e-6, text_y_pos, "← NIRCam",
                 ha='right', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
                 fontsize=fontsize_lg)

    # Add MIRI label to the right
    ax_scat.text(5.1e-6, text_y_pos, "MIRI →",
                 ha='left', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
                 fontsize=fontsize_lg)
    ax_spec.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=fontsize_lg)
    print('mean ratio : ', np.mean(results[correction + '_synthetic_fluxes'] / results['photo_fluxes']))
    print('ratio std : ', np.std(results[correction + '_synthetic_fluxes'] / results['photo_fluxes']))
    correction_factors = np.array(results[correction + '_correction_values'])
    print('corrections per solid angle : ', correction_factors / (np.pi * (results['radius'].value) ** 2))
    print(f'Correction values : {correction_factors}')

    plt.show()
    return None


def generate_psf(
    psf_type,
    fwhm_pix=None,
    size=51,
    psf_array=None
):
    """
    Generate a 2D PSF kernel.

    Parameters
    ----------
    psf_type : str
        Type of PSF. Options:
        - 'gaussian'
        - 'array' (user-supplied PSF)
    fwhm_pix : float, optional
        FWHM of Gaussian PSF in pixels (required if psf_type='gaussian')
    size : int
        Size of PSF array (must be an odd integer)
    psf_array : 2D ndarray, optional
        User-provided PSF array (required if psf_type='array')

    Returns
    -------
    psf : 2D ndarray
        Normalized PSF kernel
    """
    #TJ check that size of the psf array is an odd integer, otherwise there is no true "center pixel"
    if size % 2 != 1:
        raise ValueError("PSF size must be odd.")
    #TJ check that required parameters are given for gaussian psf generation
    if psf_type == "gaussian": 
        if fwhm_pix is None:
            raise ValueError("fwhm_pix must be provided for Gaussian PSF.")
        #TJ convert fwhm value into a sigma value for gassian mathematics so we know how many sigma each pixel lies away from the center
        sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
        #TJ generate grid of pixels
        y, x = np.mgrid[:size, :size]
        cy = cx = size // 2
        
        #TJ calculate diagonal distance in pixels and compute the gaussian height at that distance from center
        psf = np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * sigma**2))

    #TJ check that required parameters are given for a custom array psf generation
    elif psf_type == "array":
        if psf_array is None:
            raise ValueError("psf_array must be provided for psf_type='array'")
        psf = np.array(psf_array, dtype=float)

    else:
        raise ValueError(f"Unknown psf_type: {psf_type}")

    #TJ Normalize the psf to conserve flux
    psf /= np.sum(psf)

    return psf


def generate_gaussian_psf(
    fits_path,
    fwhm_arcsec,
    size_factor=6,
    ext='SCI'
):
    """
    Generate a Gaussian PSF kernel whose FWHM is specified in ARCSECONDS,
    automatically converted to pixels using FITS WCS.

    Parameters
    ----------
    fits_path : str
        Path to FITS file (to read pixel scale)
    fwhm_arcsec : float
        Desired FWHM in arcseconds
    size_factor : float
        Kernel size = size_factor * FWHM
    ext : int or str
        FITS extension

    Returns
    -------
    psf : 2D ndarray
        Normalized PSF kernel
    fwhm_pix : float
        FWHM in pixels
    """

    #TJ read in header info
    with fits.open(fits_path) as hdul:
        hdr = hdul[ext].header

    #TJ calculate pixel scale from header info
    if "CDELT1" in hdr:
        pixscale = abs(hdr["CDELT1"]) * 3600.0
    else:
        raise ValueError("Cannot determine pixel scale from FITS header, CDELT1 key missing")

    #TJ determine FWHM in pixel units
    fwhm_pix = fwhm_arcsec / pixscale

    #TJ if the FWHM is less than a pixel, this will actually INCREASE flux at that pixel rather than smearing it
    if fwhm_pix < 0.95:
        print(f"⚠️ WARNING: PSF FWHM = {fwhm_pix:.2f} pixels → undersampled!")

    #TJ calculate the gaussian scale(sigma) in pixel units
    sigma_pix = fwhm_pix / 2.3548

    #TJ compute the entire kernel size (how many standard deviations we go out to is given as an argument)
    half_size = int(np.ceil(size_factor * sigma_pix))
    size = 2 * half_size + 1

    #TJ build grid of pixels with smear weights on them
    y, x = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
    r2 = x**2 + y**2

    #TJ define the PSF and normalize to conserve flux
    psf = np.exp(-0.5 * r2 / sigma_pix**2)
    psf /= psf.sum()

    return psf, fwhm_pix, pixscale


def convolve_ifu_cube(
    ifu_fits_path,
    psf,
    output_path,
    ext='SCI'
):
    """
    Convolve an IFU cube spatially with a PSF and save the result.

    Parameters
    ----------
    ifu_fits_path : str
        Path to input IFU FITS file
    psf : 2D ndarray
        PSF kernel (must be normalized)
    output_path : str
        Path to save convolved FITS file
    ext : int
        FITS extension containing the data cube

    Returns
    -------
    None
    """
    #TJ extract data and header from original ifu cube
    with fits.open(ifu_fits_path) as hdul:
        data = hdul[ext].data
        header = hdul[ext].header
    #TJ check shape of cube
    if data.ndim != 3:
        raise ValueError("Expected IFU cube with shape (n_lambda, ny, nx)")

    #TJ define sizes of each dimension
    n_lambda, ny, nx = data.shape
    convolved = np.full_like(data, np.nan)

    #TJ iterate through each wavelength slice
    for k in range(n_lambda):
        slice_k = data[k]

        #TJ if slice is completely empty, skip
        if not np.any(np.isfinite(slice_k)):
            print(f"Slice {k} is fully NaN, skipping.")
            continue

        #TJ replace NaNs by zero but remember where they were so we can replace them with nans when the convolution is done
        mask = np.isfinite(slice_k)
        slice_filled = np.zeros_like(slice_k)
        slice_filled[mask] = slice_k[mask]

        #TJ use fft convolve to convolve image to psf
        conv = fftconvolve(slice_filled, psf, mode="same")

        #TJ convolve mask to get normalization of only real contributed data
        weight = fftconvolve(mask.astype(float), psf, mode="same")

        #TJ renormalize the array to conserve total flux
        good = weight > 0
        conv[good] /= weight[good]
        #TJ replace the indices that were nans back with nans
        conv[~good] = np.nan

        #TJ this is now the convolved wavelength slice, add it back to the total cube
        convolved[k] = conv

    #TJ save output and create path to new file
    hdu = fits.PrimaryHDU(convolved, header=header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hdu.writeto(output_path, overwrite=True)

    return output_path


