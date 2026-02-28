import os
os.environ["STPSF_PATH"] = "/d/ret1/Taylor/stpsf-data/" 
import webbpsf
os.environ["STPSF_PATH"] = "/d/ret1/Taylor/stpsf-data/" #TJ for some reason this only works if you do this line twice... no idea why

print(os.path.exists(os.environ["STPSF_PATH"]))
print(os.environ["STPSF_PATH"]) #TJ check that this kernel has access to the filter files
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import glob
import re
import sys
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm, imshow_norm
from astropy.nddata.utils import extract_array, add_array

from ipywidgets import interact, Dropdown
from astropy.wcs import WCS
from astropy.constants import c
from photutils.aperture import CircularAperture, aperture_photometry
import astropy.units as u
from astropy.table import Table
from tabulate import tabulate
from pathlib import Path
from tqdm.notebook import tqdm
from scipy.ndimage import zoom
from scipy.signal import fftconvolve

from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
from astropy.nddata import Cutout2D

from scipy.fft import fft2, ifft2, fftshift

from scipy.ndimage import gaussian_filter


#parent_dir = Path().resolve().parent #TJ current notebook's parent directory

os.chdir('/d/ret1/Taylor/jupyter_notebooks/Research') #TJ change working directory to be the parent directory
from Py_files.Basic_analysis import * #import basic functions from custom package
from Py_files.Image_vs_spectra import * 
#TJ import flux calibration functions (mainly compare_IFU_to_image(IFU_filepath, image_filepath, filter_filepath, loc, radius))

#TJ Define jwst pivot wavelengths for all the filters

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



def convolve_filter(IFU_fits_file, filter, output_file = None):
    '''Convolve an IFU cube to the PSF of the provided filter.
    -------------
    
    Parameters
    -------------
    IFU_fits_file : type = str - string to location of IFU fits that you want to convolve.
    filter : type = str - string describing the filter name (case sensitive), for example "F335M"
    output_file (optional, defaults to use the IFU_file with _convolved_to_{filter}.fits) : type = str - name of the convolved file
    
    Returns
    -------------
    Path to newly convolved file as a string
    '''   

    IFU_hdul = fits.open(IFU_fits_file) #TJ open fits file and extract header
    header = IFU_hdul["SCI"].header 
    ifu_pixel_scale = ((header['CDELT1']*u.deg).to(u.arcsec)).value #TJ This is the scale of a single pixel in the IFU
    #TJ This will be used to scale the PSF spread from the native psf pixel scale to the actual image scale. Very important.

    wl1, wl2 = get_filter_wl_range(filter) #TJ Find out what range of wavelengths are needed for this filter
    cube = SpectralCube.read(IFU_fits_file, hdu='SCI') #TJ extract the cube information 
    cube_wl = cube.spectral_axis.to(u.m) #TJ convert spectral range to meters for consistency
    idx = np.where((cube_wl >= wl1) & (cube_wl <= wl2))[0] #TJ extract range of indices from the cube that are within the filter range
    if len(idx) > 0:
        start_idx = max(0, idx[0] - 1)
        end_idx = min(len(cube_wl) - 1, idx[-1] + 1)
        used_range = (cube_wl[start_idx], cube_wl[end_idx]) #TJ trim cube and print what wavelengths were kept (may not be the same as filter).
        #TJ if cube does not have wavelengths for full filter, additonal print statements should inform you what you are missing.
        print(f'cube was cut to be {used_range[0]} to {used_range[1]}. Filter is from {wl1} to {wl2}')
    coverage_status = "" #TJ initialize additional missing wavelength warnings and update with whatever is missing
    if wl1 < cube_wl[0]:
        coverage_status += f"\n- Missing short wavelengths: {wl1}-{cube_wl[0]} μm"
    if wl2 > cube_wl[-1]:
        coverage_status += f"\n- Missing long wavelengths: {cube_wl[-1]}-{wl2} μm"
    
    if coverage_status:
        print(f"⚠️ Partial filter coverage in {IFU_fits_file}")
        print(f"  Filter {filter} requires: {wl1.value}-{wl2.value} meters")
        print(f"  Cube provides: {cube_wl[0].value}-{cube_wl[-1].value} meters")
        print(f"  Missing ranges:{coverage_status}")
        print(f"  Using available range: {used_range[0].value}-{used_range[1].value} meters")

    print(f'prior to cut : cube goes from {cube.spectral_axis[0]} to {cube.spectral_axis[-1]}')
    cube = cube.spectral_slab(used_range[0], used_range[1]) #TJ cut the cube.
    print(f'after cut : cube goes from {cube.spectral_axis[0]} to {cube.spectral_axis[-1]}')
    
    spectral_axis = cube.spectral_axis  #TJ in micrometers!
    # === Load webbpsf instrument === #
    instrument = 'NIRCam' if get_filter_number(filter) < 450 else "MIRI" #TJ determine whether this is a MIRI or NIRCam filter and assign
    if instrument == 'NIRCam':
        inst = webbpsf.NIRCam()
    elif instrument == "MIRI":
        inst = webbpsf.MIRI()
    inst.filter = filter

    # === Prepare output cube ===
    convolved_data = np.zeros_like(cube.unmasked_data[:].value) #TJ initialize cube for faster assignment
    #TJ initialize progress tracker
    tqdm_kwargs = {
        'dynamic_ncols': True,  # Auto-adjusts width
        'mininterval': 0.5,     # Update every 0.5 seconds (optional)
        'position': 0,          # Fix position (set to 0 for notebooks)
        'leave': True           # Leaves progress bar after completion
    }
    # === Loop through wavelengths and convolve === #
    for i, wavelength in enumerate(tqdm(spectral_axis, desc=f"Convolving to {filter}")):
        psf = inst.calc_psf(monochromatic=wavelength.to(u.m).value) #TJ calculate PSF for each wavelength
        psf_pixel_scale = psf[0].header['PIXELSCL'] #TJ determine the pixel scale of the PSF. Very important.
        factor =  psf_pixel_scale / ifu_pixel_scale #TJ this is the conversion factor to get from PSF scale to IFU scale and back

        psf_data = psf[0].data #TJ this is the actual PSF, it is a complicated function with lots of parameters
        psf_data_resampled = zoom(psf_data, zoom=factor, order=3) #TJ convert to IFU scale by zooming out by a factor determined by the pixel scale ratio
        psf_data_resampled /= psf_data_resampled.sum() #TJ normalize the PSF so the integral is 1.0
        image_slice = cube.filled_data[...].value[i]  #TJ this is a 2D image at this wavelength, convolve it to the PSF above
        convolved_slice = convolve_fft(image_slice, psf_data_resampled, normalize_kernel=True, boundary='fill', fill_value=0)
        convolved_data[i] = convolved_slice #TJ assign convolved slice to the cube of zeros

    new_wl_axis = cube.spectral_axis  #TJ his is the new spectral axis after spectral_slab cut

    #TJ Update the header's spectral WCS, this is important for reading the new IFU cubes from the new fits files
    wcs = WCS(header)
    if wcs.has_spectral:
        wcs.wcs.crpix[2] = 1  # Reset reference pixel
        wcs.wcs.crval[2] = new_wl_axis[0].to(u.m).value  # Start wavelength in meters
        wcs.wcs.cdelt[2] = np.mean(np.diff(new_wl_axis.to(u.m).value))  # Delta wavelength
        
        #TJ Apply changes to header
        header.update(wcs.to_header())

    # === Save the convolved cube === #
    out_hdu = fits.PrimaryHDU(convolved_data, header=header)
    if output_file:
        out_hdu.writeto(f"Data_files/IFU_files/{output_file}", overwrite=True)
        print(f"✅ PSF convolution complete and saved as {output_file}")
        return f"Data_files/IFU_files/{output_file}"
    else:
        out_hdu.writeto(f"Data_files/IFU_files/{IFU_fits_file.split('.f')[0]}_convolved_to_{filter}.fits", overwrite=True)
        print(f"✅ PSF convolution complete and saved as {IFU_fits_file.split('.f')[0]}_convolved_to_{filter}.fits")
        return f"Data_files/IFU_files/{IFU_fits_file}_convolved_to_{filter}.fits"


def convolve_full(IFU_fits_file, filter, output_file = None):
    '''Convolve an IFU cube to the PSF of the provided filter, this does not cut the cube. This does not work if the cube has wavelengths
    That are outside the compatible range of the PSF. If you need to circumvent this, see convolve_using_reference() function.
    -------------
    
    Parameters
    -------------
    IFU_fits_file : type = str - string to location of IFU fits that you want to convolve.
    filter : type = str - string describing the filter name (case sensitive), for example "F335M"
    output_file (optional, defaults to use the IFU_file with _convolved_to_{filter}.fits) : type = str - name of the convolved file
    
    Returns
    -------------
    Path to newly convolved file as a string
    '''   

    IFU_hdul = fits.open(IFU_fits_file)
    header = IFU_hdul["SCI"].header
    ifu_pixel_scale = ((header['CDELT1']*u.deg).to(u.arcsec)).value

    cube = SpectralCube.read(IFU_fits_file, hdu='SCI')
    
    spectral_axis = cube.spectral_axis  #TJ in meters
    # === Load webbpsf instrument ===
    instrument = 'NIRCam' if get_filter_number(filter) < 450 else "MIRI"
    if instrument == 'NIRCam':
        inst = webbpsf.NIRCam()
    elif instrument == "MIRI":
        inst = webbpsf.MIRI()
    inst.filter = filter

    # === Prepare output cube ===
    convolved_data = np.zeros_like(cube.unmasked_data[:].value)
    tqdm_kwargs = {
        'dynamic_ncols': True,  # Auto-adjusts width
        'mininterval': 3,     # Update every 3 seconds (optional)
        'position': 0,          # Fix position (set to 0 for notebooks)
        'leave': True           # Leaves progress bar after completion
    }
    # === Loop through wavelengths and convolve ===
    for i, wavelength in enumerate(tqdm(spectral_axis, desc=f"Convolving to {filter}")):
        psf = inst.calc_psf(monochromatic=wavelength.to(u.m).value)    
        psf_pixel_scale = psf[0].header['PIXELSCL']
        factor =  psf_pixel_scale / ifu_pixel_scale

        psf_data = psf[0].data
        psf_data_resampled = zoom(psf_data, zoom=factor, order=3)
        psf_data_resampled /= psf_data_resampled.sum()
        image_slice = cube.filled_data[...].value[i]  # 2D image at this wavelength
        convolved_slice = convolve_fft(image_slice, psf_data_resampled, normalize_kernel=True, boundary='fill', fill_value=0)
        convolved_data[i] = convolved_slice

    # === Save the convolved cube ===
    out_hdu = fits.PrimaryHDU(convolved_data, header=header)
    if output_file:
        out_hdu.writeto(f"Data_files/IFU_files/{output_file}", overwrite=True)
        print(f"✅ PSF convolution complete and saved as {output_file}")
        return f"Data_files/IFU_files/{output_file}"
    else:
        out_hdu.writeto(f"Data_files/IFU_files/{IFU_fits_file.split('.f')[0]}_convolved_to_{filter}.fits", overwrite=True)
        print(f"✅ PSF convolution complete and saved as {IFU_fits_file.split('.f')[0]}_convolved_to_{filter}.fits")
        return f"Data_files/IFU_files/{IFU_fits_file}_convolved_to_{filter}.fits"



def convolve_using_reference(IFU_fits_file, filter, output_file=None):
    '''Convolve an entire IFU cube to a single PSF of the provided filter, this does not cut the cube. This is not a realistic convolution
    (which calculates the PSF for each wavelength independently), this convolves the entire cube to a single PSF
    -------------
    
    Parameters
    -------------
    IFU_fits_file : type = str - string to location of IFU fits that you want to convolve.
    filter : type = str - string describing the filter name (case sensitive), for example "F335M", or, if string doesnt start with "F", use wavelength in meters.
    output_file (optional, defaults to use the IFU_file with _convolved_to_{filter}.fits) : type = str - name of the convolved file
    
    Returns
    -------------
    Path to newly convolved file as a string
    '''   
    def find_closest_filter(target_wavelength, filter_dict):
        '''selects whichever filter from this dictionary is closest to the reference wavelength (for calculated filter effects on PSF)'''
        return min(filter_dict.items(), key=lambda item: abs(item[1] - target_wavelength))
        
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
        # Get reference wavelength (filter pivot wavelength)
    try:
        if filter.startswith('F'):
            ref_wavelength = jwst_pivot_wavelengths[filter]
    except:
        ref_wavelength = try_float(filter)
         
    print(f"Using PSF at {ref_wavelength} meters (filter pivot wavelength) for all slices")

    # Load data
    cube = SpectralCube.read(IFU_fits_file, hdu='SCI')
    header = fits.getheader(IFU_fits_file, 'SCI')
    ifu_pixel_scale = ((header['CDELT1']*u.deg).to(u.arcsec)).value

    # Initialize instrument
    instrument = 'NIRCam' if get_filter_number(find_closest_filter(ref_wavelength, jwst_pivot_wavelengths)[0]) < 450 else "MIRI"
    if instrument == 'NIRCam':
        inst = webbpsf.NIRCam()
    elif instrument == 'MIRI':
        inst = webbpsf.MIRI()
    else:
        raise ValueError("Instrument must be 'NIRCam' or 'MIRI'")
    inst.filter = find_closest_filter(ref_wavelength, jwst_pivot_wavelengths)[0]


    # Calculate PSF once at reference wavelength
    psf = inst.calc_psf(monochromatic=ref_wavelength)
    psf_data = psf[0].data
    psf_pixel_scale = psf[0].header['PIXELSCL']
    factor =  psf_pixel_scale / ifu_pixel_scale

    psf_data_resampled = zoom(psf_data, zoom=factor, order=3)
    psf_data_resampled /= psf_data_resampled.sum()
    # Convolve all slices with the same PSF
    convolved_data = np.zeros_like(cube.unmasked_data[:].value)
    for i in tqdm(range(cube.shape[0]), desc=f"Convolving to {filter} PSF"):
        convolved_data[i] = convolve_fft(
            cube.filled_data[i].value,
            psf_data_resampled,
            normalize_kernel=True,
            boundary='fill',
            fill_value=0
        )

    # Save output
    out_hdu = fits.PrimaryHDU(convolved_data, header=header)
    if output_file:
        out_hdu.writeto(f"Data_files/IFU_files/{output_file}", overwrite=True)
        print(f"✅ PSF convolution complete and saved as {output_file}")
        return f"Data_files/IFU_files/{output_file}"
    else:
        out_hdu.writeto(f"Data_files/IFU_files/{IFU_fits_file.split('.f')[0]}_convolved_to_{filter}.fits", overwrite=True)
        print(f"✅ PSF convolution complete and saved as {IFU_fits_file.split('.f')[0]}_convolved_to_{filter}.fits")
        return f"Data_files/IFU_files/{IFU_fits_file}_convolved_to_{filter}.fits"

def convolve_image_to_psf(image_file, filter, location=None,size=20*u.arcsec, output_file=None):
    '''Convolve an image file to a different PSF of the provided filter or reference wavelength. 
    -------------
    
    Parameters
    -------------
    IFU_fits_file : type = str - string to location of IFU fits that you want to convolve.
    filter : type = str - string describing the filter name (case sensitive), for example "F335M", or, if string doesnt start with "F", use wavelength in meters.
    location (optional, defaults to None) : type = [ra, dec] or skycoord - location of center of cropped image
    size (optional, defaults to 20 arcseconds) : type = angular side length of cropped image centered around location
    output_file (optional, defaults to use the IFU_file with _convolved_to_{filter}.fits) : type = str - name of the convolved file
    
    Returns
    -------------
    Path to newly convolved file as a string
    '''   
    def find_closest_filter(target_wavelength, filter_dict):
        '''selects whichever filter from this dictionary is closest to the reference wavelength (for calculated filter effects on PSF)'''
        return min(filter_dict.items(), key=lambda item: abs(item[1] - target_wavelength))
        
        # Get reference wavelength (filter pivot wavelength)
    try:
        if filter.startswith('F'):
            ref_wavelength = jwst_pivot_wavelengths[filter]
    except:
        ref_wavelength = try_float(filter)
         
    print(f"Using PSF at {ref_wavelength} meters (filter pivot wavelength) for all slices")
    with fits.open(image_file) as hdul:
        sci_hdu = hdul['SCI']
        header = sci_hdu.header
        data = sci_hdu.data
        wcs = WCS(header)
        image_pixel_scale = ((header['CDELT1']*u.deg).to(u.arcsec)).value

    
    # Initialize instrument
    instrument = 'NIRCam' if ref_wavelength < 5.6e-6 else "MIRI"
    if instrument == 'NIRCam':
        inst = webbpsf.NIRCam()
    elif instrument == 'MIRI':
        inst = webbpsf.MIRI()
    else:
        raise ValueError("Instrument must be 'NIRCam' or 'MIRI'")
    inst.filter = find_closest_filter(ref_wavelength, jwst_pivot_wavelengths)[0]

    if location:
        if type(location) == list:
            loc = SkyCoord(ra=location[0]*u.deg, dec=location[1]*u.deg)
        elif type(location) == SkyCoord:
            loc = location
        else:
            print('loc is not a list of ra, dec and it is not a SkyCoord object.')
            return None
        cutout_size_pix = (size / image_pixel_scale)  # square cutout size
        cutout = Cutout2D(data, position=loc, size=(cutout_size_pix, cutout_size_pix), wcs=wcs)
        data = cutout.data
        header.update(cutout.wcs.to_header())
    # Calculate PSF once at reference wavelength
    psf = inst.calc_psf(monochromatic=ref_wavelength)
    psf_data = psf[0].data
    psf_pixel_scale = psf[0].header['PIXELSCL']
    factor =  psf_pixel_scale / image_pixel_scale

    psf_data_resampled = zoom(psf_data, zoom=factor, order=3)
    psf_data_resampled /= psf_data_resampled.sum()
    # Convolve all slices with the same PSF
    print("Image shape:", data.shape)
    print("PSF shape:", psf_data.shape)

    convolved_data = convolve_fft(
            data,
            psf_data_resampled,
            normalize_kernel=True,
            boundary='fill',
            fill_value=0,
            allow_huge=True
        )
    

    # Save output
    out_hdu = fits.PrimaryHDU(convolved_data, header=header)
    if output_file:
        out_hdu.writeto(f"Data_files/Image_files/Convolved_images/{output_file}", overwrite=True)
        print(f"✅ PSF convolution complete and saved as {output_file}")
        return f"Data_files/Image_files/Convolved_images/{output_file}"
    else:
        out_hdu.writeto(f"Data_files/Image_files/Convolved_images/{image_file.split('lv3_')[-1].split('_i2d')[0]}_convolved_to_{filter}.fits", overwrite=True)
        print(f"✅ PSF convolution complete and saved as {image_file.split('lv3_')[-1].split('_i2d')[0]}_convolved_to_{filter}.fits")
        return f"Data_files/Image_files/Convolved_images/{image_file.split('lv3_')[-1].split('_i2d')[0]}_convolved_to_{filter}.fits"

def preprocess_IFU():

    
    return None


def homogenize_to_target_psf(input_fits_file, target_wavelength_m, output_file=None,
                              location=None, size=20*u.arcsec, native_wl = None):
    """
    Convolve a JWST image or IFU cube to the PSF of a target wavelength.
    Applies a *differential kernel* to preserve flux.

    Parameters
    ----------
    input_fits_file : str
        Path to the 2D image or 3D IFU cube FITS file.
    target_wavelength_m : float
        Target wavelength in meters (e.g., 21 micron = 21e-6).
    output_filepath : str
        Where to save the convolved file.
    location : list or SkyCoord, optional
        If image is large, center [RA, Dec] in deg or a SkyCoord object for cropping.
    cutout_size : Quantity, optional
        Area around `location` to extract before convolving (default 20 arcsec).

    Returns
    -------
    output_filepath : str
        File path of the saved convolved FITS file.
    """
    def find_closest_filter(target_wavelength, filter_dict):
        '''selects whichever filter from this dictionary is closest to the reference wavelength (for calculated filter effects on PSF)'''
        return min(filter_dict.items(), key=lambda item: abs(item[1] - target_wavelength))
        
    def get_psf(wavelength_m, instrument):
        inst = webbpsf.NIRCam() if instrument == 'NIRCam' else webbpsf.MIRI()
        inst.filter = find_closest_filter(wavelength_m, jwst_pivot_wavelengths)[0]
        print(wavelength_m, inst.filter)
        psf = inst.calc_psf(monochromatic=wavelength_m)
        psf_data = psf[0].data
        psf_scale = psf[0].header['PIXELSCL']
        return psf_data, psf_scale

    def match_shape(arr, target_shape):
        """
        Crop or pad `arr` to match `target_shape`.
        Pads with zeros if needed, or crops symmetrically.
        """
        out = np.zeros(target_shape)
        
        min_shape = np.minimum(arr.shape, target_shape)
        
        # Calculate slices for input and output arrays
        in_slices = tuple(slice((a - m)//2, (a - m)//2 + m) for a, m in zip(arr.shape, min_shape))
        out_slices = tuple(slice((t - m)//2, (t - m)//2 + m) for t, m in zip(target_shape, min_shape))
    
        out[out_slices] = arr[in_slices]
        return out
    try:
        hdul = fits.open(input_fits_file)["SCI"]
    except:
        hdul = fits.open(input_fits_file)[0]

    hdr = hdul.header
    ndim = hdr['NAXIS']
    if ndim == 2:
        data = hdul.data
        header = hdul.header
        wcs = WCS(header)
        image_scale = (header['CDELT1'] * u.deg).to(u.arcsec).value

        # Optional cutout
        if location:
            loc = SkyCoord(*location, unit='deg') if isinstance(location, list) else location
            cut_size = (size / image_scale)
            cutout = Cutout2D(data, position=loc, size=cut_size, wcs=wcs)
            data = cutout.data
            header.update(cutout.wcs.to_header())

        # Determine instrument and get PSFs
        if native_wl is None:
            native_filter = extract_filter_name(input_fits_file).upper()
            native_wavelength = jwst_pivot_wavelengths[native_filter]

        else:
            native_wavelength = native_wl
        if native_wavelength is None:
            raise ValueError("Unknown or unsupported filter in image header.")

        native_instrument = 'MIRI' if native_wavelength > 5.6e-6 else 'NIRCam'
        native_psf, psf_scale = get_psf(native_wavelength, native_instrument)
        target_instrument = 'MIRI' if target_wavelength_m > 5.6e-6 else 'NIRCam'
        target_psf, target_psf_scale = get_psf(target_wavelength_m, target_instrument)

        print(psf_scale)
        print(target_psf_scale)

        # Resample
        factor = psf_scale / image_scale
        native_psf = zoom(native_psf, factor)
        target_psf = zoom(target_psf, factor)
        native_psf /= native_psf.sum()
        target_psf /= target_psf.sum()

        max_shape = tuple(np.maximum(native_psf.shape, target_psf.shape))

        native_psf_resampled = match_shape(native_psf, max_shape)
        target_psf_resampled = match_shape(target_psf, max_shape)
        # Differential kernel and convolution
        kernel_ft = fft2(target_psf_resampled) / (fft2(native_psf_resampled) + 1e-16) #TJ to prevent divide by zero error in fft
        diff_kernel = np.real(ifft2(kernel_ft))
        diff_kernel = fftshift(diff_kernel)
        diff_kernel /= diff_kernel.sum()

        convolved = convolve_fft(data, diff_kernel, normalize_kernel=True)
        out_hdu = fits.PrimaryHDU(convolved, header=hdr)
        if output_file:
            out_hdu.writeto(f"Data_files/Image_files/Convolved_images/{output_file}", overwrite=True)
            print(f"✅ PSF convolution complete and saved as {output_file}")
            return f"Data_files/Image_files/Convolved_images/{output_file}"
        else:
            out_hdu.writeto(f"Data_files/Image_files/Convolved_images/{image_file.split('lv3_')[-1].split('_i2d')[0]}_convolved_to_{target_wavelength_m}.fits", overwrite=True)
            print(f"✅ PSF convolution complete and saved as {image_file.split('lv3_')[-1].split('_i2d')[0]}_convolved_to_{target_wavelength_m}.fits")
            return f"Data_files/Image_files/Convolved_images/{image_file.split('lv3_')[-1].split('_i2d')[0]}_convolved_to_{target_wavelength_m}.fits"

    elif ndim == 3:
        cube = SpectralCube.read(input_fits_file, hdu='SCI')
        pix_scale = (cube.header['CDELT1'] * u.deg).to(u.arcsec).value
        target_instrument = 'MIRI' if target_wavelength_m > 5.6e-6 else 'NIRCam'
        target_psf, psf_scale = get_psf(target_wavelength_m, target_instrument)
        target_psf = zoom(target_psf, psf_scale / pix_scale)
        target_psf /= target_psf.sum()


        output_cube = np.zeros_like(cube.unmasked_data[:].value)
        for i in range(cube.shape[0]):
            wl = cube.spectral_axis[i].to(u.m).value
            native_instrument = 'MIRI' if wl > 5.6e-6 else 'NIRCam'
            native_psf, _ = get_psf(wl, native_instrument)
            native_psf = zoom(native_psf, psf_scale / pix_scale)
            native_psf /= native_psf.sum()


            max_shape = tuple(np.maximum(native_psf.shape, target_psf.shape))

            native_psf_resampled = match_shape(native_psf, max_shape)
            target_psf_resampled = match_shape(target_psf, max_shape)
            kernel_ft = fft2(target_psf_resampled) / (fft2(native_psf_resampled) + (1e-6 * np.max(fft2(native_psf_resampled))))
            diff_kernel = np.real(ifft2(kernel_ft))
            diff_kernel = fftshift(diff_kernel)
            diff_kernel /= diff_kernel.sum()

            slice_data = cube.filled_data[i]
            pad_size = 10 * np.array(diff_kernel.shape)  # Example: triple the kernel size
            padded_data = np.pad(slice_data, [(pad_size[0],), (pad_size[1],)], mode='constant', constant_values=0)
            
            # Convolve
            padded_result = convolve_fft(padded_data, diff_kernel, normalize_kernel=True, boundary='fill', fill_value=0)
            
            # Crop back to original size
            output_cube[i] = extract_array(
                padded_result, 
                cube.shape[1:], 
                position=(padded_result.shape[0]//2, padded_result.shape[1]//2)  # (y, x) center
            )

        out_hdu = fits.PrimaryHDU(output_cube, header=hdr)
        if output_file:
            out_hdu.writeto(f"Data_files/IFU_files/{output_file}", overwrite=True)
            print(f"✅ PSF convolution complete and saved as {output_file}")
            return f"Data_files/IFU_files/{output_file}"
        else:
            out_hdu.writeto(f"Data_files/IFU_files/{IFU_fits_file.split('.f')[0]}_convolved_to_{target_wavelength_m}.fits", overwrite=True)
            print(f"✅ PSF convolution complete and saved as {IFU_fits_file.split('.f')[0]}_convolved_to_{target_wavelength_m}.fits")
            return f"Data_files/IFU_files/{IFU_fits_file}_convolved_to_{target_wavelength_m}.fits"

    else:
        raise ValueError("Only 2D or 3D FITS files supported.")


import warnings

def apply_fourier_lowpass_filter(kernel_ft, sigma_fraction=0.05):
    """
    Apply a low-pass Gaussian filter to the Fourier transform of a kernel.

    Parameters:
        kernel_ft (np.ndarray): The Fourier transform of the differential kernel.
        sigma_fraction (float): Width of the Gaussian taper as a fraction of the Nyquist frequency (default 0.05).

    Returns:
        np.ndarray: The filtered Fourier transform.
    """
    shape = kernel_ft.shape
    ny, nx = shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)

    # Create a Gaussian filter in Fourier space
    sigma = sigma_fraction  # relative to Nyquist frequency
    gaussian_taper = np.exp(-(FX**2 + FY**2) / (2 * sigma**2))

    return kernel_ft * gaussian_taper

def test_convolve_image(file_path, target_wavelength_m, native_filter, output_filepath=None):
    """
    Convolve a JWST IFU cube or 2D image to match the PSF at the target wavelength.

    Parameters:
    - file_path : str
        Path to FITS file (either IFU cube or 2D image).
    - target_wavelength_m : float
        Target wavelength in meters for the desired PSF resolution.
    - output_filepath : str, optional
        Path to save the convolved FITS file. If None, will auto-generate one.
    - get_native_psf : function
        Callable: get_native_psf(wavelength_m) → PSF 2D numpy array.
    - get_target_psf : function
        Callable: get_target_psf(target_wavelength_m) → PSF 2D numpy array.
    """
    def get_psf(wavelength_m, filter, native_pixel_scale):
        instrument = 'MIRI' if get_filter_number(filter) > 550 else 'NIRCam'

        inst = webbpsf.NIRCam() if instrument == 'NIRCam' else webbpsf.MIRI()
        inst.filter = filter
        psf = inst.calc_psf(monochromatic=wavelength_m)
        psf_data = psf[0].data
        psf_scale = psf[0].header['PIXELSCL']
        factor = psf_scale / native_pixel_scale
        new_psf = zoom(psf_data, factor)
        new_psf /= new_psf.sum()
        return new_psf

    # Load FITS
    hdul = fits.open(file_path)
    header = hdul['SCI'].header
    native_pixel_scale = (header['CDELT1']*u.deg).to(u.arcsec).value
    data = hdul['SCI'].data
    data = np.nan_to_num(data, nan=0.0)
    is_cube = data.ndim == 3
    shape = data.shape
    
    # Load target PSF
    psf_target = get_psf(target_wavelength_m, 'F2100W', native_pixel_scale)

    # Pad PSFs to the same shape
    def pad_to_match(psf1, psf2):
        """Pad both PSFs to have the same shape (centered)."""
        shape1 = np.array(psf1.shape)
        shape2 = np.array(psf2.shape)
        target_shape = np.maximum(shape1, shape2)
    
        def pad_center(psf, target_shape):
            pad_total = target_shape - np.array(psf.shape)
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            padding = tuple((before, after) for before, after in zip(pad_before, pad_after))
            return np.pad(psf, padding, mode='constant', constant_values=0)
    
        return pad_center(psf1, target_shape), pad_center(psf2, target_shape)

    # Compute differential kernel in Fourier space
    def compute_diff_kernel(psf_native, psf_target):
        psf_native, psf_target = pad_to_match(psf_native, psf_target)
    
        fft_native = np.fft.fft2(np.fft.ifftshift(psf_native))
        fft_target = np.fft.fft2(np.fft.ifftshift(psf_target))
    
        kernel_ft = fft_target / (fft_native + 1e-8)  # prevent divide-by-zero
        differential_kernel_ft_filtered = apply_fourier_lowpass_filter(kernel_ft, sigma_fraction=0.05)
        
        kernel = np.fft.fftshift(np.fft.ifft2(differential_kernel_ft_filtered).real)
        return kernel

    # Pad image before convolution
    def convolve_with_kernel(img, kernel):
        """
        Convolve an image with a kernel using FFT, handling padding safely.
        Returns an image of the same shape as `img`.
        """
        # Pad to avoid edge artifacts
        pad_y, pad_x = kernel.shape[0]//2, kernel.shape[1]//2
        padded_img = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
    
        # Convolve in 'same' mode (keeps padded_img size)
        convolved = fftconvolve(padded_img, kernel, mode='same')
    
        # Crop back to original shape
        center_y, center_x = convolved.shape[0]//2, convolved.shape[1]//2
        half_y, half_x = img.shape[0]//2, img.shape[1]//2
        start_y = center_y - half_y
        start_x = center_x - half_x
        return convolved[start_y:start_y + img.shape[0], start_x:start_x + img.shape[1]]

    

    # Handle cube or image
    if is_cube:
        cube = SpectralCube.read(file_path, hdu='SCI')

        output = np.zeros_like(data)
        for i in range(shape[0]):
            wl = cube.spectral_axis[i].to(u.m).value
            psf_native = get_psf(wl, native_filter, native_pixel_scale)
            kernel = compute_diff_kernel(psf_native, psf_target)
            output[i] = convolve_with_kernel(data[i], kernel)

        out_path = f'Data_files/IFU_files/Convolved_to_21um/{native_filter}_ifu_convolved.fits'
    else:
        native_wl = jwst_pivot_wavelengths[native_filter]
        psf_native = get_psf(native_wl, native_filter, native_pixel_scale)
        kernel = compute_diff_kernel(psf_native, psf_target)
        output = convolve_with_kernel(data, kernel)
        out_path = f'Data_files/Image_files/Convolved_images/{native_filter}_img_convolved.fits'
    # Save to new FITS
    hdu = fits.PrimaryHDU(data=output, header=header)
    hdul_out = fits.HDUList([hdu])
    print("Output stats:", np.nanmin(output), np.nanmax(output), np.isnan(output).sum())
    print("Data stats:", np.nanmin(data), np.nanmax(data), np.isnan(data).sum())

    if output_filepath is None:
        output_filepath = out_path
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdul_out.writeto(output_filepath, overwrite=True)

    hdul.close()

    return output_filepath



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
        Size of PSF array (must be odd)
    psf_array : 2D ndarray, optional
        User-provided PSF array (required if psf_type='array')

    Returns
    -------
    psf : 2D ndarray
        Normalized PSF kernel
    """

    if size % 2 == 0:
        raise ValueError("PSF size must be odd.")

    if psf_type == "gaussian":
        if fwhm_pix is None:
            raise ValueError("fwhm_pix must be provided for Gaussian PSF.")

        sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        y, x = np.mgrid[:size, :size]
        cy = cx = size // 2

        psf = np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * sigma**2))

    elif psf_type == "array":
        if psf_array is None:
            raise ValueError("psf_array must be provided for psf_type='array'")
        psf = np.array(psf_array, dtype=float)

    else:
        raise ValueError(f"Unknown psf_type: {psf_type}")

    # Normalize to unit integral (flux-conserving)
    psf /= np.sum(psf)

    return psf


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

    with fits.open(ifu_fits_path) as hdul:
        data = hdul[ext].data
        header = hdul[ext].header

    # Expected shape: (n_lambda, ny, nx)
    if data.ndim != 3:
        raise ValueError("Expected IFU cube with shape (n_lambda, ny, nx)")

    n_lambda, ny, nx = data.shape
    convolved = np.full_like(data, np.nan)

    for k in range(n_lambda):
        slice_k = data[k]

        if np.all(~np.isfinite(slice_k)):
            continue

        # Replace NaNs with zero for convolution
        mask = np.isfinite(slice_k)
        slice_filled = np.zeros_like(slice_k)
        slice_filled[mask] = slice_k[mask]

        conv = fftconvolve(slice_filled, psf, mode="same")

        # Renormalize to avoid edge bias
        weight = fftconvolve(mask.astype(float), psf, mode="same")
        good = weight > 0
        conv[good] /= weight[good]
        conv[~good] = np.nan

        convolved[k] = conv

    # Save output
    hdu = fits.PrimaryHDU(convolved, header=header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hdu.writeto(output_path, overwrite=True)
    return output_path

