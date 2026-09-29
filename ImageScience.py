import os
import numpy as np
import glob
import time
from time import sleep

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

from Functions import *

from reproject import reproject_interp

from scipy.ndimage import fourier_shift
from scipy.ndimage import zoom, shift as ndi_shift

from skimage.registration import phase_cross_correlation
from reproject import reproject_interp as rpj

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_quadratic
from photutils.detection import DAOStarFinder

from astropy import constants as const
from astropy.table import Table
import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.wcs.utils import pixel_to_skycoord

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft

from astropy.visualization import ZScaleInterval, LogStretch, ImageNormalize, AsinhStretch

from scipy.optimize import curve_fit


def convert_to_fnu_sr(data, header, wcs):
    """
    Convert image to F_nu [W m^-2 Hz^-1 sr^-1].

    Supports:
        ELECTRONS/S
        COUNTS/S
        Jy/pixel
        MJy/sr
        Jy/sr

    Returns
    -------
    data, header
        Converted image and updated header.
    """

    bunit = str(header.get('BUNIT', '')).strip().upper()

    #TJ pixel area in steradians
    pixel_area_sr = proj_plane_pixel_area(wcs) * (np.pi/180.)**2

    # --------------------------------------------------
    #TJ ACS/HST calibrated count rate images
    # --------------------------------------------------
    if bunit == 'W M-2 HZ-1 SR-1':
        print('Data was already in desired units.')
        return data, header

    elif bunit == '1E-20 ERG/S/CM2/ARCSEC2':
        print('Cnverting data in image from 1e-20 ergs/s/cm2/arcsec2 to W/m2/Hz/sr')
        f_lam = ((data*1e-20) / header['PHOTBW'])* u.erg/u.s/u.cm**2/u.arcsec**2/u.AA
        if 'Angstroms' in header.comments['PHOTPLAM']:
            pivot = (header['PHOTPLAM']*1e-10)*u.m #TJ units of angstroms in 
        else:
            print('Could not find units for PHOTPLAM in header')
        f_nu = (f_lam.to(u.W/u.m**2/u.sr/u.m)) * pivot**2 / const.c
        data = f_nu.value
    elif bunit in ['ERG/S/CM2/PIXEL', 'ERG/S/CM2/PIX']:

        print('converting data in image from erg/s/cm2/pix to W/m2/Hz/sr')

        f_lambda_si = data * 1e-7 * 1e4  #TJ erg/s/cm^2 → W/m^2

        if 'PHOTPLAM' in header and 'PHOTFLAM' in header:

            photflam = header['PHOTFLAM']  # erg/s/cm2/Å/electron (or flux density scale)
            pivot_A = header['PHOTPLAM']

            # interpret as flux density calibration anchor
            lam = pivot_A * 1e-10  # m

            # convert F_lambda → F_nu
            # (we assume PHOTFLAM already embedded in data scaling)
            f_nu = f_lambda_si * lam**2 / const.c.value

        else:
            # fallback: assume image already represents band-integrated flux
            # approximate conversion using pivot wavelength if available
            if 'PHOTPLAM' in header:
                lam = header['PHOTPLAM'] * 1e-10
            else:
                raise ValueError("Cannot convert ERG/S/CM2/PIX: missing PHOTPLAM")

            f_nu = f_lambda_si * lam**2 / const.c.value

        data = f_nu / pixel_area_sr

    elif bunit in ['ELECTRONS/S', 'COUNTS/S', 'COUNTS', 'ELECTRONS']:
        print('converting data in image from electrons/s to W/m2/hz/sr')
        if 'PHOTFLAM' not in header:
            raise ValueError(
                f'{bunit} image missing PHOTFLAM keyword'
            )

        comment = header.comments['PHOTFLAM'].lower()

        expected = ['ergs/cm2/ang/electron', 'ergs/cm2/a/e-']
        if (expected[0] not in comment.replace(' ', '') and expected[1] not in comment.replace(' ', '')):
            raise ValueError(
                f'Unexpected PHOTFLAM definition:\n{comment}'
            )

        if 'PHOTPLAM' not in header:
            raise ValueError(
                'PHOTPLAM required for count-rate conversion'
            )

        photflam = header['PHOTFLAM']
        pivot_A = header['PHOTPLAM']

        # counts -> F_lambda
        f_lambda_cgs = data * photflam

        # cgs -> SI
        f_lambda_si = f_lambda_cgs * 1e7 / 1e4 / 1e-10

        lam = pivot_A * 1e-10

        f_nu = f_lambda_si * lam**2 / const.c.value

        data = f_nu / pixel_area_sr

    # --------------------------------------------------
    # MJy/sr
    # --------------------------------------------------
    elif bunit == 'MJY/SR':
        print('converting data in image from MJy/sr to W/m2/hz/sr')

        data = data * (u.MJy/u.sr).to(u.W/u.m**2/u.Hz/u.sr)

    # --------------------------------------------------
    # Jy/sr
    # --------------------------------------------------
    elif bunit == 'JY/SR':
        print('converting data in image from JY/sr to W/m2/hz/sr')

        data = data * (u.Jy/u.sr).to(u.W/u.m**2/u.Hz/u.sr)

    # --------------------------------------------------
    # Jy/pixel
    # --------------------------------------------------
    elif bunit in ['JY/PIXEL', 'JY/PIX', 'JY']:
        print('converting data in image from JY/pix to W/m2/hz/sr')

        data = data * 1e-26
        data /= pixel_area_sr

    else:

        raise ValueError(
            f'Unsupported BUNIT = {bunit}'
        )

    # Update header
    header['BUNIT'] = (
        'W m-2 Hz-1 sr-1',
        'Converted to SI F_nu surface brightness'
    )

    header['ORIGUNIT'] = (
        bunit,
        'Original image units'
    )

    return data, header


class ImageScience:

    """
    Tools for continuum subtraction from filters.
    """


    def __init__(self):
        #TJ initialize dictionary of class attributes
        self.images = {}
        self.headers = {}
        self.files = {}
        self.wcs = {}
    
    def load_image(self, name, filename, hdu=None, clear_sip=False):

        """
        Load FITS image and convert to

            F_nu [W m^-2 Hz^-1 sr^-1]
        """

        hdul = fits.open(filename)

        self.files[name] = filename
        if hdu is not None:
            data = hdul[hdu].data.astype(float)
            header = hdul[hdu].header.copy()
            wcs = WCS(header, hdul)
        else:
            try:

                data = hdul['SCI'].data.astype(float)
                header = hdul['SCI'].header.copy()
                wcs = WCS(header, hdul)

            except Exception:

                print('SCI extension not found, using primary')

                data = hdul[0].data.astype(float)
                header = hdul[0].header.copy()
                wcs = WCS(header)

        # Convert to common units
        data, header = convert_to_fnu_sr(
            data,
            header,
            wcs
        )

        self.images[name] = data
        self.headers[name] = header
        if clear_sip:
            wcs.sip=None
            print('Clear_SIP given in function, check that file name has drz in the title and ignore warning about inconsistent wcs info')
        self.wcs[name] = wcs

        print(
            f'Loaded: {name} '
            f'[{header["BUNIT"]}]'
        )
        
    def circular_mask(
        self,
        image_name,
        x_center,
        y_center,
        radius
    ):
        '''
        Used to mask out the galaxy center if it is too bright
        '''

        data = self.images[image_name].copy()

        yy, xx = np.indices(data.shape)

        r = np.sqrt(
            (xx - x_center)**2 +
            (yy - y_center)**2
        )

        mask = r <= radius

        data[mask] = np.nan

        self.images[image_name] = data

        print(f'Masked {image_name}')

    def check_alignment(
        self,
        image1,
        image2
    ):

        """
        Print basic alignment diagnostics.
        """

        data1 = self.images[image1]
        data2 = self.images[image2]

        print('----------------------------------')
        print('Alignment Diagnostics')
        print('----------------------------------')
        print(f'{image1} shape: {data1.shape}')
        print(f'{image2} shape: {data2.shape}')

        h1 = self.headers[image1]
        h2 = self.headers[image2]

        try:

            pix1 = abs(h1['CDELT1'])
            pix2 = abs(h2['CDELT1'])

            print(f'{image1} pixel scale: {pix1}')
            print(f'{image2} pixel scale: {pix2}')

        except:

            print('Could not determine CDELT1')

        print('----------------------------------')

    def align_images(
        self,
        reference_image,
        other_image,
        out_file=None,
        out_name=None
    ):
        '''
        Reproject other image into pixel grid of the reference image using WCS info
        '''

        if out_name is None:

            out_name = f'{other_image}_aligned'

        print(
            f'Reprojecting {other_image} '
            f'onto {reference_image}'
        )

        target_header = self.headers[reference_image]

        target_shape = self.images[
            reference_image
        ].shape

        #TJ use reproject to align images using their WCS info
    
        reproj, footprint = reproject_interp(
            (
                self.images[other_image],
                self.wcs[other_image]
            ),
            self.wcs[reference_image],
            shape_out=target_shape
        )
        if out_file is not None:
            hdu = fits.PrimaryHDU(
                data=reproj,
                header=target_header.copy()
            )

            hdu.writeto(
                out_file,
                overwrite=True
            )

            print(f'Saved aligned image:')
            print(out_file)
        self.images[out_name] = reproj
        self.headers[out_name] = target_header.copy()
        self.wcs[out_name] = self.wcs[reference_image].copy()

    def make_cutout(
        self,
        image_name,
        size,
        center=None,
        out_name=None,
        mode="trim",
        fill_value=np.nan,
    ):
        """
        Create a WCS-aware cutout of an image.

        Parameters
        ----------
        image_name : str
            Name of image stored in the ImageScience object.

        size : Quantity or tuple of Quantities
            Angular size of the cutout (e.g. 5*u.arcsec).

        center : None, SkyCoord, tuple, optional
            Center of the cutout.

            Supported formats
            -----------------
            None
                Uses the center of the image.

            SkyCoord
                Uses the supplied sky coordinate.

            (ra, dec)
                Quantities or floats (assumed degrees).

            (x, y)
                Pixel coordinates.

        out_name : str, optional
            If supplied, store the cutout in the ImageScience object.

        mode : {"trim", "partial", "strict"}
            Passed directly to Cutout2D.

        fill_value : float
            Used when mode="partial".

        Returns
        -------
        cutout : Cutout2D
        """

        image = self.images[image_name]
        wcs = self.wcs[image_name]
        header = self.headers[image_name]

        # ---------------------------------------------------------
        # Determine center
        # ---------------------------------------------------------

        if center is None:

            ny, nx = image.shape

            center = pixel_to_skycoord(
                (nx - 1) / 2,
                (ny - 1) / 2,
                wcs,
            )

        elif isinstance(center, SkyCoord):

            pass

        elif isinstance(center, (tuple, list)) and len(center) == 2:

            a, b = center

            # Sky coordinates
            if isinstance(a, u.Quantity) or isinstance(b, u.Quantity):

                center = SkyCoord(a, b)

            elif (
                np.issubdtype(type(a), np.floating)
                and
                np.issubdtype(type(b), np.floating)
            ):

                # Assume degrees
                center = SkyCoord(
                    a*u.deg,
                    b*u.deg,
                )

            else:

                # Pixel coordinates
                center = (a, b)

        else:

            raise ValueError(
                "center must be None, SkyCoord, "
                "(ra,dec), or (x,y)"
            )

        # ---------------------------------------------------------
        # Make cutout
        # ---------------------------------------------------------

        cutout = Cutout2D(
            image,
            position=center,
            size=size,
            wcs=wcs,
            mode=mode,
            fill_value=fill_value,
        )

        # ---------------------------------------------------------
        # Store if requested
        # ---------------------------------------------------------

        if out_name is not None:

            self.images[out_name] = cutout.data

            self.wcs[out_name] = cutout.wcs

            new_header = header.copy()
            new_header.update(cutout.wcs.to_header())

            self.headers[out_name] = new_header

        return cutout

    def get_pa(self, wcs_name):

        if self.wcs[wcs_name].wcs.has_cd():
            M = self.wcs[wcs_name].wcs.cd
        else:
            pc = self.wcs[wcs_name].wcs.get_pc()
            cdelt = self.wcs[wcs_name].wcs.cdelt
            M = pc @ np.diag(cdelt)

        return np.degrees(np.arctan2(M[0,0], M[1,0]))

    def sum_images(self, name1, name2, out_name=None, out_file=None, scales=[1,1]):
        '''
        Sum two images with the same shape.
        Usually to take a continuum subtracted line image and a continuum image to recreate full image
        '''

        im1 = self.images[name1]
        im2 = self.images[name2]
        try:
            image = (im1*scales[0])+(im2*scales[1])
        except ValueError:
            print('Images were not the same size')
        if out_file is None:
            out_file = (self.files[name1].replace('.fits', '_summed.fits'))
        hdu = fits.PrimaryHDU(
            data=image,
            header=self.headers[name1]
        )

        hdu.writeto(
            out_file,
            overwrite=True
        )
        self.images[out_name] = image
        self.headers[out_name] = self.headers[name1]
        self.files[out_name] = out_file
        self.wcs[out_name] = self.wcs[name1]

    def sub_images(self, name1, name2, out_name=None, out_file=None, scales=[1,1]):
        '''
        Subtract imagename2 from imagename1 with the same shape. For continuum subtraction usually.
        '''

        im1 = self.images[name1]
        im2 = self.images[name2]
        try:
            image = (im1*scales[0])-(im2*scales[1])
        except ValueError:
            print('Images were not the same size')
        if out_file is None:
            out_file = (self.files[name1].replace('.fits', '_diff.fits'))
        hdu = fits.PrimaryHDU(
            data=image,
            header=self.headers[name1]
        )

        hdu.writeto(
            out_file,
            overwrite=True
        )
        self.images[out_name] = image
        self.headers[out_name] = self.headers[name1]
        self.files[out_name] = out_file
        self.wcs[out_name] = self.wcs[name1]

    def make_ratio(self, name1, name2, out_name=None, out_file=None, scales=[1,1]):
        '''
        Make an image that is the ratio of two images.
        '''

        im1 = self.images[name1]
        im2 = self.images[name2]
        try:
            image = (im1*scales[0])/(im2*scales[1])
        except ValueError:
            print('Images were not the same size')
        if out_name is None:
            out_name = name1 + '_' + name2 + '_ratio'
        if out_file is not None:
            hdu = fits.PrimaryHDU(
                data=image,
                header=self.headers[name1]
            )

            hdu.writeto(
                out_file,
                overwrite=True
            )
            self.files[out_name] = out_file


        self.images[out_name] = image
        self.headers[out_name] = self.headers[name1]
        self.wcs[out_name] = self.wcs[name1]

    def get_pix_scale(self, wcs_name):
        return (proj_plane_pixel_scales(self.wcs[wcs_name])[0]*u.deg).to(u.arcsec)

    def get_pix_area(self, name):
        """
        Return pixel area in steradians.

        Priority:
        1) PIXAR_SR keyword
        2) CDELT1/CDELT2 + CUNIT1/CUNIT2
        3) CD1_1/CD2_2 + CUNIT1/CUNIT2

        Parameters
        ----------
        header : astropy.io.fits.Header

        Returns
        -------
        pix_area_sr : float
            Pixel area in steradians.
        """
        header = self.headers[name]
        # --------------------------------------------------
        # JWST-style pixel area keyword
        # --------------------------------------------------
        if 'PIXAR_SR' in header:
            return float(header['PIXAR_SR'])*u.sr

        # --------------------------------------------------
        # Determine coordinate units
        # --------------------------------------------------
        cunit1 = header.get('CUNIT1')
        cunit2 = header.get('CUNIT2')
        if cunit1 is None:
            return ((proj_plane_pixel_area(self.wcs[name]))*u.deg**2).to(u.sr)

        try:
            unit1 = u.Unit(cunit1)
            unit2 = u.Unit(cunit2)
        except Exception:
            raise ValueError(
                f"Could not interpret CUNIT1='{cunit1}' "
                f"or CUNIT2='{cunit2}'"
            )
        # --------------------------------------------------
        # First choice: CDELT keywords
        # --------------------------------------------------
        if 'CDELT1' in header and 'CDELT2' in header:

            pix_x = abs(header['CDELT1']) * unit1
            pix_y = abs(header['CDELT2']) * unit2

            return (pix_x * pix_y).to(u.sr)

        # --------------------------------------------------
        # Second choice: CD matrix diagonal elements
        # --------------------------------------------------
        if all(k in header for k in ['CD1_1','CD1_2','CD2_1','CD2_2']):
            cd = np.array([
                [header['CD1_1'], header['CD1_2']],
                [header['CD2_1'], header['CD2_2']]
            ])

            area = abs(np.linalg.det(cd)) * unit1 * unit2
            return area.to(u.sr)

        # --------------------------------------------------
        # Nothing usable found
        # --------------------------------------------------
        raise KeyError(
            "Could not determine pixel area. "
            "Need PIXAR_SR, or CDELT1/CDELT2, "
            "or CD1_1/CD2_2."
        )

    def make_kernel(self,
        image_name,
        fwhm=None,
        kernel_filepath=None,
        crop_size=None,
        normalize=True
    ):
        """
        Build or prepare a 2D kernel to convolve with `image_name`.

        Two modes:
        - kernel_filepath given: reproject an existing kernel FITS file
        onto this image's pixel grid/WCS.
        - fwhm given (arcsec): build a Gaussian2DKernel matched to this
        image's pixel scale.

        Parameters
        ----------
        image_name : str
            Key into self.images / self.headers used to get pixel scale/header.
        fwhm : float or None
            Target Gaussian FWHM in arcsec (used if kernel_filepath is None).
        kernel_filepath : str or None
            Path to a kernel FITS file to reproject onto image_name's grid.
        crop_size : int or None
            Passed through to reproject_kernel_to_image, if used.
        normalize : bool
            If True, normalize kernel to sum to 1.

        Returns
        -------
        kernel : 2D ndarray
        """
        if kernel_filepath is not None:
            kernel = reproject_kernel_to_image(
                kernel_filepath,
                self.headers[image_name],
                crop_size=crop_size,
                normalize=normalize
            )

        elif fwhm is not None:
            pixscale = self.get_pix_scale(image_name).to_value(u.arcsec)
            fwhm_pix = fwhm / pixscale
            sigma_pix = fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
            kernel = Gaussian2DKernel(sigma_pix).array

        else:
            raise ValueError(
                "make_kernel requires either kernel_filepath or fwhm to build a kernel."
            )

        #TJ fft convolve hates nans, replace them with zeros
        kernel = np.nan_to_num(kernel)

        if normalize:
            kernel /= np.sum(kernel)

        return kernel

    def fft_convolve(self,
        image_name,
        kernel,
        out_name=None,
        normalize_kernel=True,
        preserve_nan=True,
        boundary='fill',
        fill_value=0.0,
        return_time=False,
        out_file=None
    ):
        """
        Convolve image using FFT convolution.

        Parameters
        ----------
        image_name : str
            Key into self.images for the image to convolve.
        kernel : 2D ndarray
            Kernel to convolve with (e.g. from make_kernel()).

        Returns
        -------
        convolved : ndarray
        elapsed_time : float (seconds)
        """

        #TJ start timer to keep track of how long this method takes to convolve
        t0 = time.perf_counter()

        image = self.images[image_name]

        #TJ fft convolve hates nans, replace them with zeros
        kernel = np.nan_to_num(kernel)

        if normalize_kernel:
            kernel = kernel / np.sum(kernel)

        #TJ keep track of where the nans were
        if preserve_nan:
            nan_mask = ~np.isfinite(image)

        #TJ then remove the nans and convolve
        image_filled = np.nan_to_num(image)

        convolved = convolve_fft(
            image_filled,
            kernel,
            boundary=boundary,
            fill_value=fill_value,
            normalize_kernel=False,
            preserve_nan=False,
            allow_huge=True
        )

        if preserve_nan:
            convolved[nan_mask] = np.nan

        #TJ end timer when convolution ends
        elapsed = time.perf_counter() - t0

        #TJ copy header info from base file and save data as convolved version
        if out_name is None:
            out_name = f'{image_name}_fft_conv'

        self.images[out_name] = convolved
        self.headers[out_name] = self.headers[image_name].copy()
        self.wcs[out_name] = self.wcs[image_name].copy()
        if out_file is not None:
            out_file = (
                self.files[image_name]
                .replace('.fits', '_convolved.fits')
            )
            hdu = fits.PrimaryHDU(
                data=self.images[out_name],
                header=self.headers[out_name]
            )

            hdu.writeto(
                out_file,
                overwrite=True
            )
            print(f'File written to {out_file}')

        if return_time:
            print(f'fft convolution took {elapsed} seconds')
            return elapsed

    def continuum_subtract(
        self,
        feature_image,
        continuum_image,
        scale_factor,
        out_name='cont_subtracted'
    ):
        """
        Outdated function for subtracting continuum
        """

        subtracted = (
            self.images[feature_image] -
            scale_factor * self.images[continuum_image]
        )

        self.images[out_name] = subtracted
        self.headers[out_name] = self.headers[feature_image].copy()

        print(f'Created: {out_name}')

    def save_fits(
        self,
        image_name,
        output_file,
        scale=1
    ):
        '''
        Simple function to save image to a fits file
        '''

        hdu = fits.PrimaryHDU(
            data=self.images[image_name]*scale,
            header=self.headers[image_name]
        )

        hdu.writeto(
            output_file,
            overwrite=True
        )

        print(f'Saved: {output_file}')

    def inspect_continuum_subtraction(
        self,
        feature_name,
        continuum_name,
        initial_scale=1.072,
        zoom_size=1000,
        zoom_center=None,
        mask_x=None,
        mask_y=None,
        mask_radius=None,
        show_all=False
    ):
        """
        Displays an image that is the feature image minus some scaler times
        the continuum image, with an interactable slider for changing that scaler.
        Used to check if stellar light is fully subtracted
        """

        # -----------------------------------------------------
        # COPY DATA
        # -----------------------------------------------------

        feature = self.images[feature_name].copy()
        cont = self.images[continuum_name].copy()

        # -----------------------------------------------------
        # OPTIONAL MASK
        # -----------------------------------------------------

        if (
            mask_x is not None and
            mask_y is not None and
            mask_radius is not None
        ):

            yy, xx = np.indices(feature.shape)

            r = np.sqrt(
                (xx - mask_x)**2 +
                (yy - mask_y)**2
            )

            mask = r <= mask_radius

            feature[mask] = np.nan
            cont[mask] = np.nan

        # -----------------------------------------------------
        # CENTRAL CUTOUT
        # -----------------------------------------------------
        if zoom_size is not None:
            ny, nx = feature.shape
            if zoom_center is None:
                x_center = nx // 2
                y_center = ny // 2
            else:
                x_center = zoom_center[0]
                y_center = zoom_center[1]

            x1 = x_center - zoom_size // 2
            x2 = x_center + zoom_size // 2

            y1 = y_center - zoom_size // 2
            y2 = y_center + zoom_size // 2

            feature_cut = feature[y1:y2, x1:x2]
            cont_cut = cont[y1:y2, x1:x2]
        else:
            feature_cut = feature
            cont_cut = cont

        # -----------------------------------------------------
        # INITIAL MODEL
        # -----------------------------------------------------

        continuum = initial_scale * cont_cut

        subtracted = feature_cut - continuum

        # -----------------------------------------------------
        # NORMALIZATION
        # -----------------------------------------------------
        if show_all:
            combined = np.concatenate([
                feature_cut[np.isfinite(feature_cut)].ravel(),
                continuum[np.isfinite(continuum)].ravel(),
                cont_cut[np.isfinite(cont_cut)].ravel()
            ])

            vmin = np.percentile(combined, 1)
            vmax = np.percentile(combined, 99.7)
            fig, axes = plt.subplots(
            2,
            2,
            figsize=(8, 8)
            )

            axes = axes.ravel()

        else:
            vmax = np.percentile(feature_cut[np.isfinite(feature_cut)].ravel(), 99.7)
            vmin = np.percentile(feature_cut[np.isfinite(feature_cut)].ravel(), 1)

            fig, axes = plt.subplots(figsize=(6, 6))



        sub_v = np.nanpercentile(
            np.abs(subtracted),
            99
        )

        # -----------------------------------------------------
        # FIGURE
        # -----------------------------------------------------


        plt.subplots_adjust(bottom=0.15)

        # -----------------------------------------------------
        # F187N
        # -----------------------------------------------------
        if show_all:
            im0 = axes[0].imshow(
                feature_cut,
                origin='lower',
                cmap='gray',
                vmin=vmin,
                vmax=vmax
            )

            axes[0].set_title('feature')
            axes[0].axis('off')

            # -----------------------------------------------------
            # CONTINUUM
            # -----------------------------------------------------

            im1 = axes[1].imshow(
                continuum,
                origin='lower',
                cmap='gray',
                vmin=vmin,
                vmax=vmax
            )

            title1 = axes[1].set_title(
                f'Continuum = {initial_scale:.5f}'
            )

            axes[1].axis('off')

            # -----------------------------------------------------
            # SUBTRACTED
            # -----------------------------------------------------

            im2 = axes[2].imshow(
                subtracted,
                origin='lower',
                cmap='RdBu_r',
                vmin=-sub_v,
                vmax=sub_v
            )

            title2 = axes[2].set_title(
                'feature - Continuum'
            )

            axes[2].axis('off')

            # -----------------------------------------------------
            # F150W
            # -----------------------------------------------------

            im3 = axes[3].imshow(
                cont_cut,
                origin='lower',
                cmap='gray',
                vmin=vmin,
                vmax=vmax
            )

            axes[3].set_title('Continuum Image')
            axes[3].axis('off')

            # -----------------------------------------------------
            # COLORBARS
            # -----------------------------------------------------

            plt.colorbar(
                im0,
                ax=axes[0],
                fraction=0.046
            )

            plt.colorbar(
                im1,
                ax=axes[1],
                fraction=0.046
            )

            plt.colorbar(
                im2,
                ax=axes[2],
                fraction=0.046
            )

            plt.colorbar(
                im3,
                ax=axes[3],
                fraction=0.046
            )

        else:
            im2 = axes.imshow(
                subtracted,
                origin='lower',
                cmap='RdBu_r',
                vmin=-sub_v,
                vmax=sub_v
            )

            title2 = axes.set_title(
                'feature - Continuum'
            )
            plt.colorbar(
                im2,
                ax=axes,
                fraction=0.046
            )

            axes.axis('off')

        # -----------------------------------------------------
        # SLIDER
        # -----------------------------------------------------

        ax_slider = plt.axes(
            [0.2, 0.05, 0.6, 0.03]
        )

        scale_slider = Slider(
            ax=ax_slider,
            label='Scale Factor',
            valmin=0.01,
            valmax=2,
            valinit=initial_scale,
            valstep=0.001
        )

        # -----------------------------------------------------
        # UPDATE
        # -----------------------------------------------------
        ny, nx = feature_cut.shape
        cx0, cy0 = nx // 2, ny // 2

        aperture_patch = Circle(
            (cx0, cy0),
            radius=5,
            edgecolor='cyan',
            facecolor='none',
            lw=1.5
        )
        axes.add_patch(aperture_patch)
        def update(val):

            scale = scale_slider.val

            continuum_new = scale * cont_cut

            subtracted_new = (
                feature_cut -
                continuum_new
            )
            if show_all:
                im1.set_data(continuum_new)
                title1.set_text(f'Continuum = {scale:.5f}')
            im2.set_data(subtracted_new)

            sub_v_new = np.nanpercentile(
                np.abs(subtracted_new),
                99
            )

            im2.set_clim(
                -sub_v_new,
                sub_v_new
            )


            ny, nx = subtracted_new.shape
            y, x = np.indices((ny, nx))

            cx, cy = nx // 2, ny // 2
            r = np.sqrt((x - cx)**2 + (y - cy)**2)

            aperture = r <= 5

            total_flux = np.nansum(subtracted_new[aperture])
            aperture_patch.center = (cx, cy)

            title2.set_text(
                f'Total Flux (r=5 pix) = {total_flux:.5e}'
            )


            fig.canvas.draw_idle()

        scale_slider.on_changed(update)

        plt.show()

    def get_background_subtracted_flux(
        self,
        image_name,
        loc,
        radius,
        background_annulus_thickness,
        buffer=0*u.arcsec
    ):

        """
        Exact circular aperture photometry with
        annulus background subtraction.

        Uses:
            - fractional pixel overlap
            - median background estimate per pixel
            - may still overestimate background in crowded fields

        Parameters
        ----------
        image_name : str

        loc : SkyCoord or [ra, dec]

        radius : astropy Quantity
            Source aperture radius.

        background_annulus_thickness : astropy Quantity
            Thickness of background annulus.

        buffer : astropy Quantity
            Gap between source aperture and annulus.

        Returns
        -------
        results : dict

            Contains:
                source_flux
                background_flux
                net_flux
                background_per_pixel
                source_area_pixels
                annulus_area_pixels
        """
        #TJ load file and check arguments are correct types
        # =====================================================
        image = self.images[image_name]
        header = self.headers[image_name]
        wcs = self.wcs[image_name]
        if isinstance(loc, list):
            spatial_coords = SkyCoord(
                ra=loc[0] * u.deg,
                dec=loc[1] * u.deg
            )

        elif isinstance(loc, SkyCoord):
            spatial_coords = loc
        else:
            raise ValueError(
                'loc is not SkyCoord or [ra, dec]'
            )

        #TJ Check units
        # =====================================================
        try:
            units = header['BUNIT']
        except:
            print('Units not found in header with key BUNIT, aperture photometry failed')
            return None
        if units == 'W m-2 Hz-1 sr-1':
            original_units = u.W / (u.m**2 * u.Hz * u.sr)
            image_quantity = image*original_units
        
        elif units == 'MJy/sr':
            original_units = u.MJy / u.sr
            image_quantity = (
                image * original_units
            ).to(
                u.W / (u.m**2 * u.Hz * u.sr)
            )
        elif units == "erg / (s cm2)":
            original_units = (
                u.erg / (u.s * u.cm**2)
            )
            pixel_area = self.get_pix_area(image_name)

            image_quantity = (
                (image * original_units) /
                pixel_area
            ).to(
                u.W / (u.m**2 * u.sr)
            )
        else:
            raise ValueError(
                f'Unsupported BUNIT: {units}'
            )

        pix_area = self.get_pix_area(image_name)
        try:
            pixel_scale_deg = proj_plane_pixel_scales(wcs)[0]
        except:
            print('Pixel size not found in header, aperture photometry failed')
            return None
        #TJ convert to pixel units instead of angular
        source_radius_pixels = (radius.to_value(u.deg) / pixel_scale_deg)

        x, y = wcs.all_world2pix(
            spatial_coords.ra.deg,
            spatial_coords.dec.deg,
            0
        )

        #TJ create the apertures and calculate fluxes
        # =====================================================
        try:
            source_aperture = CircularAperture(
                (x, y),
                r=source_radius_pixels
            )
        except:
            print('source aperture not valid')
            return {
            'source_flux': np.nan*u.W / (u.m**2 * u.Hz),
            'background_flux': np.nan*u.W / (u.m**2 * u.Hz),
            'net_flux': np.nan*u.W / (u.m**2 * u.Hz),
            'background_surface_brightness': np.nan*u.W / (u.m**2 * u.Hz),
            'source_area_pixels': np.nan,
            'annulus_area_pixels': np.nan}

        source_flux = aperture_photometry(
            image_quantity,
            source_aperture,
            method='exact'
        )['aperture_sum'][0] * pix_area
        source_area_pixels = (
            source_aperture.area
        )

        if background_annulus_thickness > 0:
            bg_inner_pixels = ((radius + buffer).to_value(u.deg) / pixel_scale_deg)

            bg_outer_pixels = ((radius + buffer + background_annulus_thickness).to_value(u.deg) / pixel_scale_deg)

            bg_annulus = CircularAnnulus(
                (x, y),
                r_in=bg_inner_pixels,
                r_out=bg_outer_pixels
            )



            annulus_mask = bg_annulus.to_mask(method='exact')

            annulus_data = annulus_mask.multiply(image_quantity.value)

            annulus_weights = annulus_mask.data

            # VALID PIXELS
            # =====================================================

            valid = (
                np.isfinite(annulus_data) &
                (annulus_weights > 0.01)
            )

            annulus_values = annulus_data[valid]

            annulus_weights = annulus_weights[valid]

            #TJ calculate median pixel value in annulus
            # =====================================================

            # Recover intrinsic pixel values by dividing
            # weighted contributions by overlap fraction

            intrinsic_pixel_values = (
                annulus_values /
                annulus_weights
            )

            background_surface_brightness = (np.nanmedian(intrinsic_pixel_values) * image_quantity.unit)

            background_flux = (background_surface_brightness * pix_area * source_area_pixels)

            annulus_area_pixels = np.sum(annulus_weights)

        else:
            background_flux = 0*source_flux.unit
            background_surface_brightness = 0 * image_quantity.unit
            annulus_area_pixels = 0

        net_flux = (source_flux - background_flux)
        
        return {
            'source_flux': source_flux,
            'background_flux': background_flux,
            'net_flux': net_flux,
            'background_surface_brightness': background_surface_brightness,
            'source_area_pixels': source_area_pixels,
            'annulus_area_pixels': annulus_area_pixels
        }


    def select_aperture(
        self,
        image_name,
        loc=None,
        radius=1.0 * u.arcsec,
        zoom=8,
        buff=0.1 * u.arcsec,
        ann=0.1 * u.arcsec,
        cmap='viridis'
    ):
        """
        Interactively select an aperture for aperture photometry.

        Left-click and drag inside aperture:
            Move the aperture.

        Left-click and drag near aperture edge:
            Resize the aperture.

        Click 'Confirm Aperture' button:
            Accept the aperture. Then call get_result() in the next cell
            to retrieve the final center and radius.

        Parameters
        ----------
        image_name : str
            Name of the image in self.images.
        loc : SkyCoord, tuple, list, or None
            Initial aperture center. If None, the center of the image is used.
            If a tuple/list is supplied, it is interpreted as (RA, Dec) in degrees.
        radius : Quantity
            Initial aperture radius. Default is 1 arcsec.
        zoom : float
            Width of displayed cutout in units of aperture radii. Default is 8.
        buff : Quantity
            Width between the source aperture and background annulus.
        ann : Quantity
            Width of the background annulus.
        cmap : str
            Matplotlib colormap.

        Returns
        -------
        get_result : callable
            Call get_result() in a subsequent cell after confirming the aperture.
            Returns (loc, radius) as (SkyCoord, Quantity).
        """

        import ipywidgets as widgets
        from IPython.display import display as ipy_display

        # ================================================================
        # Get image information
        # ================================================================

        image = self.images[image_name]
        wcs   = self.wcs[image_name]

        # ================================================================
        # Convert center to SkyCoord
        # ================================================================

        if loc is None:
            x0  = (image.shape[1] - 1) / 2
            y0  = (image.shape[0] - 1) / 2
            loc = wcs.pixel_to_world(x0, y0)
        elif not isinstance(loc, SkyCoord):
            loc = SkyCoord(ra=loc[0]*u.deg, dec=loc[1]*u.deg, frame='icrs')

        # ================================================================
        # Pixel scale
        # ================================================================

        pixscale = self.get_pix_scale(image_name).to(u.arcsec)

        # ================================================================
        # Make initial cutout
        # ================================================================

        display_size = zoom * (radius + buff + ann)

        cutout = Cutout2D(
            image,
            position=loc,
            size=display_size,
            wcs=wcs,
            mode='trim'
        )

        cut_data = cutout.data
        cut_wcs  = cutout.wcs

        cx, cy = cut_wcs.world_to_pixel(loc)

        # ================================================================
        # Initial radii in pixels
        # ================================================================

        radius_pix = radius.to(u.arcsec).value / pixscale.value
        buff_pix   = buff.to(u.arcsec).value   / pixscale.value
        ann_pix    = ann.to(u.arcsec).value    / pixscale.value

        # ================================================================
        # Plot
        # ================================================================

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': cut_wcs})

        good = np.isfinite(cut_data)
        vmin = np.nanpercentile(cut_data[good], 5)
        vmax = np.nanpercentile(cut_data[good], 99.5)

        ax.imshow(
            cut_data,
            origin='lower',
            cmap=cmap,
            norm=ImageNormalize(cut_data, stretch=AsinhStretch(), vmin=vmin, vmax=vmax)
        )

        ax.coords.grid(color='white', linestyle='--', linewidth=0.8, alpha=0.5)

        # ================================================================
        # Aperture circles
        # ================================================================

        aperture_circle = Circle((cx, cy), radius_pix,
                                edgecolor='red', facecolor='none', linewidth=2)
        buffer_circle   = Circle((cx, cy), radius_pix + buff_pix,
                                edgecolor='cyan', facecolor='none', linewidth=1.5, linestyle='--')
        annulus_circle  = Circle((cx, cy), radius_pix + buff_pix + ann_pix,
                                edgecolor='cyan', facecolor='none', linewidth=1.5, linestyle='--')

        ax.add_patch(aperture_circle)
        ax.add_patch(buffer_circle)
        ax.add_patch(annulus_circle)

        ax.set_title(
            f'{image_name}\nDrag center to move  |  Drag edge to resize  |  Click Confirm when done',
            fontsize=11
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # ================================================================
        # State
        # ================================================================

        state = {
            'loc':          loc,
            'radius_pix':   radius_pix,
            'mode':         None,
            'final_loc':    None,
            'final_radius': None,
        }

        # ================================================================
        # Mouse callbacks
        # ================================================================

        def on_press(event):
            if event.inaxes != ax or event.xdata is None:
                return
            dx = event.xdata - aperture_circle.center[0]
            dy = event.ydata - aperture_circle.center[1]
            dist = np.sqrt(dx**2 + dy**2)
            edge_tol = max(5, 0.25 * aperture_circle.radius)
            if abs(dist - aperture_circle.radius) < edge_tol:
                state['mode'] = 'resize'
            elif dist <= aperture_circle.radius:
                state['mode'] = 'move'
            else:
                state['mode'] = None

        def on_release(event):
            state['mode'] = None

        def on_motion(event):
            if state['mode'] is None or event.inaxes != ax or event.xdata is None:
                return

            if state['mode'] == 'move':
                new_x, new_y = event.xdata, event.ydata
                for circle in [aperture_circle, buffer_circle, annulus_circle]:
                    circle.center = (new_x, new_y)
                # Convert cutout pixel -> full image pixel -> sky
                x_full = new_x + cutout.xmin_original
                y_full = new_y + cutout.ymin_original
                state['loc'] = wcs.pixel_to_world(x_full, y_full)

            elif state['mode'] == 'resize':
                dx = event.xdata - aperture_circle.center[0]
                dy = event.ydata - aperture_circle.center[1]
                new_r = max(np.sqrt(dx**2 + dy**2), 1.0)
                state['radius_pix'] = new_r
                aperture_circle.set_radius(new_r)
                buffer_circle.set_radius(new_r + buff_pix)
                annulus_circle.set_radius(new_r + buff_pix + ann_pix)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event',   on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event',  on_motion)

        # ================================================================
        # Confirm button
        # ================================================================

        confirm_button = widgets.Button(
            description='Confirm Aperture',
            button_style='success',
            icon='check',
            layout=widgets.Layout(width='200px', height='40px')
        )
        output = widgets.Output()

        def on_confirm(b):
            state['final_radius'] = state['radius_pix'] * pixscale
            state['final_loc']    = state['loc']
            confirm_button.disabled    = True
            confirm_button.description = '✓ Confirmed'
            with output:
                print('Aperture confirmed:')
                print(f'  RA     = {state["final_loc"].ra.deg:.8f} deg')
                print(f'  Dec    = {state["final_loc"].dec.deg:.8f} deg')
                print(f'  Radius = {state["final_radius"]:.4f}')
                print('Now call loc, radius in the next cell using the output.')

        confirm_button.on_click(on_confirm)

        plt.tight_layout()
        ipy_display(widgets.VBox([confirm_button, output]))
        plt.show(block=False)

        # ================================================================
        # get_result callable returned to the user
        # ================================================================

        def get_result():
            if state['final_loc'] is None:
                print('Aperture not yet confirmed — click "Confirm Aperture" first.')
                return None, None
            return state['final_loc'], state['final_radius']

        return get_result


    def get_equivalent_width(self,
        feature_image_name,
        continuum_image_name,
        location,
        radius,
        background_annulus_thickness,
        buffer=0*u.arcsec
    ):

        """
        Compute equivalent width using:
            - narrowband feature image
            - aligned/scaled continuum image

        Includes annular background subtraction.

        Parameters
        ----------
        feature_image_file : str (this is the full feature image, not the continuum subtracted one)

        continuum_filter_file : str (This is just the continuum image, subtracting this from the feature_image should give just the line)

        location : SkyCoord or [ra, dec] or (x, y)

        radius : float
            Aperture radius in pixels.

        background_annulus_thickness : astropy Quantity
            Thickness of annulus for background
        
        buffer : astropy Quantity
            Gap between source aperture and annulus.

        Returns
        -------
        EW : astropy Quantity

        line_flux : astropy Quantity

        continuum_flux_density : astropy Quantity

        feature_flux : astropy Quantity

        continuum_flux : astropy Quantity
        """

        #TJ do the background subtraction for the feature image
        feature_dict = self.get_background_subtracted_flux(
                feature_image_name,
                location,
                radius,
                background_annulus_thickness,
                buffer
            )
        feature_flux = feature_dict['net_flux']
        feature_bg = feature_dict['background_flux']
        
        continuum_dict = self.get_background_subtracted_flux(
                continuum_image_name,
                location,
                radius,
                background_annulus_thickness,
                buffer
            )
        continuum_flux = continuum_dict['net_flux']
        continuum_bg = continuum_dict['background_flux']

        #TJ check units are same in both images
        if feature_flux.unit != continuum_flux.unit:

            raise ValueError(
                'Feature and continuum images '
                'have different units.'
            )

        #TJ get filter name and specs
        feature_filter = extract_filter_name(self.files[feature_image_name])

        wl, T, _, pivot, _ = get_filter_data(feature_filter, aux_info=True)

        #TJ effective width calculation
        bandwidth = (
            np.trapezoid(T, wl) /
            np.max(T)
        )

        #TJ convert f_nu to f_lambda in both files
        flam_feature = (
            feature_flux * c / pivot**2
        ).to(
            u.W / u.m**2 / u.m
        )

        flam_continuum = (
            continuum_flux * c / pivot**2
        ).to(
            u.W / u.m**2 / u.m
        )

        #TJ multiply f_lambda by dlambda to get total flux
        feature_in_filter = (
            flam_feature * bandwidth
        )

        continuum_in_filter = (
            flam_continuum * bandwidth
        )

        #TJ subtract off continuum
        line_flux = (
            feature_in_filter -
            continuum_in_filter
        )

        #TJ Equivalent width is then just cont-subtracted flux divided by continuum
        EW = (
            line_flux /
            flam_continuum
        ).to(u.Angstrom)

        return EW, line_flux, flam_continuum, feature_flux, continuum_flux
        
    def make_ew_ratio_image(
        self,
        numerator_continuum_name,
        numerator_line_name,
        denominator_continuum_name,
        denominator_line_name,
        output_name='EW_Ha_over_PaA',
        min_continuum=0,
        min_line=0,
        replace_num_negs = False,
        replace_den_negs = False
    ):
        """
        Create an image of the ratio of equivalent widths, for example:

            EW(Hα) / EW(Paα)

        from continuum and continuum-subtracted images, assumed to already be reprojected.

        Parameters
        ----------
        numerator_continuum_name : str
            Name of numerator continuum image.

        numerator_line_name : str
            Name of continuum-subtracted numerator image.

        denominator_continuum_name : str
            Name of denominator continuum image.

        denominator_line_name : str
            Name of continuum-subtracted denominator image.

        output_name : str
            Name used to store the output image.

        min_continuum : float
            Continuum values <= this are masked.

        min_line : float
            Line values <= this are masked.

        Returns
        -------
        ratio : ndarray
            EW(Hα)/EW(Paα)
        """

        numerator_cont = self.images[numerator_continuum_name].astype(float)
        numerator_line = self.images[numerator_line_name].astype(float)
        
        denominator_cont = self.images[denominator_continuum_name].astype(float)
        denominator_line = self.images[denominator_line_name].astype(float)

        numerator_EW = np.full_like(numerator_cont, np.nan, dtype=float)
        denominator_EW = np.full_like(denominator_cont, np.nan, dtype=float)
        ratio = np.full_like(denominator_cont, np.nan, dtype=float)

        numerator_EW = (numerator_line / numerator_cont)
        denominator_EW = (denominator_line / denominator_cont)
        if replace_num_negs:
            print(f'{len(numerator_EW[numerator_EW < 0])} zeros in the numerator image replaced with nans')
            numerator_EW[numerator_EW < 0] = np.nan
        if replace_den_negs:
            print(f'{len(denominator_EW[denominator_EW < 0])} zeros in the denominator image replaced with nans')
            denominator_EW[denominator_EW < 0] = np.nan

        valid = (
            np.isfinite(numerator_cont) &
            np.isfinite(numerator_line) &
            np.isfinite(denominator_cont) &
            np.isfinite(denominator_line) &
            np.isfinite(numerator_EW) &
            np.isfinite(denominator_EW) &
            (numerator_cont > min_continuum) &
            (denominator_cont > min_continuum) &
            (numerator_line > min_line) &
            (denominator_line > min_line)
        )
        ratio[valid] = numerator_EW[valid] / denominator_EW[valid]

        self.images[output_name] = ratio
        self.headers[output_name] = self.headers[numerator_line_name].copy()
        self.wcs[output_name] = self.wcs[numerator_line_name]

        self.headers[output_name]['BUNIT'] = 'dimensionless'

        return ratio

    def find_sources(self,
        image_name,
        threshold,
        fwhm,
        sigma_clip=3.0,
        sharplo=0.2,
        sharphi=1.0,
        roundlo=-1.0,
        roundhi=1.0,
        exclude_border=True,
        return_coords=True,
        min_sep=1
    ):
        """
        Detect compact (point-like) sources in an image using DAOStarFinder.

        Parameters
        ----------
        image_name : str
            Key into self.images / self.wcs for the image to search.
        threshold : float
            Detection threshold in units of the (sigma-clipped) background
            standard deviation. E.g. threshold=5 means sources must have a
            peak pixel value > 5-sigma above the background.
        fwhm : Quantity
            Expected FWHM of the PSF, in angular units (e.g. arcsec). This is
            converted to pixels internally using this image's pixel scale.
        sigma_clip : float, optional
            Sigma for sigma-clipped background stats used to estimate the
            background level and noise (default = 3.0).
        sharplo, sharphi : float, optional
            Lower/upper bound on source "sharpness" passed to DAOStarFinder,
            used to reject extended sources or hot pixels (defaults = 0.2, 1.0).
        roundlo, roundhi : float, optional
            Lower/upper bound on source "roundness" passed to DAOStarFinder,
            used to reject elongated sources (defaults = -1.0, 1.0).
        exclude_border : bool, optional
            If True, drop sources whose footprint falls off the edge of the
            image (default = True).
        return_coords : bool, optional
            If True, add a 'sky_coord' column with SkyCoord objects for each
            source, computed from this image's WCS (default = True).
        min_sep : float, optional
            Minimum number of fwhm between sources. Defaults to 1

        Returns
        -------
        sources : astropy.table.Table or None
            Table of detected sources (positions, fluxes, sharpness,
            roundness, etc.), or None if no sources were found.
        """

        image = self.images[image_name]
        wcs = self.wcs[image_name]

        #TJ convert PSF FWHM from arcsec to pixels using this image's pixel scale
        pixscale = self.get_pix_scale(image_name).to_value(u.arcsec)
        fwhm_pix = fwhm.to_value(u.arcsec) / pixscale

        #TJ estimate background level/noise via sigma-clipped stats
        #TJ fill nans first since sigma_clipped_stats and DAOStarFinder don't like them
        image_filled = np.nan_to_num(image)
        mean, median, std = sigma_clipped_stats(image_filled, sigma=sigma_clip)

        daofind = DAOStarFinder(
            fwhm=fwhm_pix,
            threshold=threshold * std,
            sharplo=sharplo,
            sharphi=sharphi,
            roundlo=roundlo,
            roundhi=roundhi,
            exclude_border=exclude_border,
            min_separation=fwhm_pix * min_sep
        )

        sources = daofind(image_filled - median)

        if sources is None:
            print(f"No sources found in '{image_name}' at threshold={threshold}.")
            return None

        print(f"Found {len(sources)} sources in '{image_name}'.")

        if return_coords:
            sky_coords = wcs.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
            sources['sky_coord'] = sky_coords

        return sources

    def display(self, names, loc=None, radius=None, bg_ann=0*u.arcsec, buffer=0*u.arcsec, ncols=3, cmap='viridis', zoom=5, show_grid=False, vmin=None, vmax=None):
        """
        Create a collage of cutout images with an aperture overlay.

        Parameters
        ----------
        names : list of str
            List of image keys (must be present in self.images / self.wcs).
        loc : list, tuple, SkyCoord, or None
            Location of aperture center, either [RA, Dec] in degrees or a SkyCoord object.
            If None (along with radius=None), the full image is displayed instead of a cutout,
            using ZScale limits and log stretch, with no aperture overlay.
        radius : Quantity or None
            Aperture radius (must have angular units, e.g. arcsec).
            If None (along with loc=None), the full image is displayed instead of a cutout,
            using ZScale limits and log stretch, with no aperture overlay.
        ncols : int, optional
            Number of columns in the collage (default = 3).
        cmap : str, optional
            Colormap for displaying images (default = 'viridis').
        zoom : float, optional
            How many radii does image include (default = 5)
        """

        full_image_mode = (loc is None) or (radius is None)

        # Make sure loc is SkyCoord (only needed in cutout mode)
        if not full_image_mode:
            if not isinstance(loc, SkyCoord):
                loc_sky = SkyCoord(ra=loc[0]*u.deg, dec=loc[1]*u.deg, frame='icrs')
            else:
                loc_sky = loc

        n_images = len(names)
        nrows = int(np.ceil(n_images / ncols))

        fig = plt.figure(figsize=(5*ncols, 5*nrows))

        for i, name in enumerate(names):
            image = self.images[name]
            wcs = self.wcs[name]
            pixel_scale = self.get_pix_scale(name).value

            if full_image_mode:
                display_data = image
                display_wcs = wcs
            else:
                cutout = Cutout2D(image, position=loc_sky, size=((radius+buffer+bg_ann)*zoom, (radius+buffer+bg_ann)*zoom), wcs=wcs)
                display_data = cutout.data
                display_wcs = cutout.wcs

            # Each subplot gets its own WCS projection
            ax = fig.add_subplot(nrows, ncols, i+1, projection=display_wcs)
            interval = ZScaleInterval()

            if full_image_mode:
                if (vmin is None) and (vmax is None):
                    ax_vmin, ax_vmax = interval.get_limits(display_data)
                else:
                    ax_vmin, ax_vmax = vmin, vmax
                im = ax.imshow(display_data, origin='lower', cmap=cmap,
                        norm=ImageNormalize(display_data, stretch=LogStretch(),
                                            vmin=ax_vmin, vmax=ax_vmax))
            else:
                x_img, y_img = display_wcs.world_to_pixel(loc_sky)
                if (vmin is None) and (vmax is None):
                    ax_vmin, ax_vmax = interval.get_limits(display_data)
                else:
                    ax_vmin, ax_vmax = vmin, vmax
                im = ax.imshow(display_data, origin='lower', cmap=cmap,
                        norm=ImageNormalize(display_data, stretch=AsinhStretch(),
                                            vmin=ax_vmin, vmax=ax_vmax))

            if show_grid:
                ax.coords.grid(color='white', linestyle='--', linewidth=1, alpha=0.7)

            if not full_image_mode:
                ax.add_patch(Circle((x_img, y_img),
                                    (radius.to(u.arcsec).value) / pixel_scale,
                                    ec='red', fc='none', lw=2, alpha=0.7))
                # Background annulus
                if bg_ann > 0*u.arcsec:

                    inner_r = (
                        radius.to(u.arcsec).value +
                        buffer.to(u.arcsec).value
                    ) / pixel_scale

                    outer_r = (
                        radius.to(u.arcsec).value +
                        buffer.to(u.arcsec).value +
                        bg_ann.to(u.arcsec).value
                    ) / pixel_scale

                    ax.add_patch(
                        Circle(
                            (x_img, y_img),
                            inner_r,
                            ec='cyan',
                            fc='none',
                            lw=2,
                            ls='--',
                            alpha=0.8
                        )
                    )

                    ax.add_patch(
                        Circle(
                            (x_img, y_img),
                            outer_r,
                            ec='cyan',
                            fc='none',
                            lw=2,
                            ls='--',
                            alpha=0.8
                        )
                    )

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Flux (native units)", fontsize=10)
            ax.set_title(name, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        for i in range(n_images, nrows*ncols):
            fig.add_subplot(nrows, ncols, i+1).axis('off')

        plt.tight_layout()
        plt.show()

    def save_object(self, filename):
        """
        Save entire ImageScience object to disk.

        Parameters
        ----------
        filename : str
            Output filename, e.g. 'science.pkl'
        """

        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved ImageScience object to {filename}")

    @classmethod
    def load_object(cls, filename):
        """
        Load a saved ImageScience object.
        """

        with open(filename, "rb") as f:
            obj = pickle.load(f)

        print(f"Loaded ImageScience object from {filename}")
        return obj

    # ============================================================
    # QA / DIAGNOSTIC PLOTTING UTILITIES
    # ============================================================
    
    def qa_cutout(
        self,
        image_name,
        center,
        size=200,
        title=None,
        cmap='gray',
        vmin_percentile=1,
        vmax_percentile=99.7
    ):
    
        """
        Display zoomed cutout around a source.
    
        Parameters
        ----------
        image_name : str
    
        center : [ra, dec] OR (x, y)
    
        size : int
            Cutout size in pixels
        """
    
        image = self.images[image_name]
        header = self.headers[image_name]
        wcs = self.wcs[image_name]
    
        # --------------------------------------------------------
        # COORDS
        # --------------------------------------------------------
    
        if isinstance(center, SkyCoord):
    
            x, y = wcs.all_world2pix(
                center.ra.deg,
                center.dec.deg,
                0
            )
    
        elif isinstance(center, list):
    
            x, y = wcs.all_world2pix(
                center[0],
                center[1],
                0
            )
    
        else:
    
            x, y = center
    
        x = int(x)
        y = int(y)
    
        # --------------------------------------------------------
        # CUTOUT
        # --------------------------------------------------------
    
        half = size // 2
    
        cut = image[
            y-half:y+half,
            x-half:x+half
        ]
    
        # --------------------------------------------------------
        # DISPLAY
        # --------------------------------------------------------
    
        finite = np.isfinite(cut)
    
        vmin = np.nanpercentile(
            cut[finite],
            vmin_percentile
        )
    
        vmax = np.nanpercentile(
            cut[finite],
            vmax_percentile
        )
    
        plt.figure(figsize=(6,6))
    
        plt.imshow(
            cut,
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
    
        plt.colorbar()
    
        if title is None:
            title = image_name
    
        plt.title(title)
    
        plt.show()
    
    def qa_compare_images(
        self,
        image1,
        image2,
        center,
        size=200,
        titles=None
    ):
    
        """
        Side-by-side comparison of two aligned images.
        """
    
        if titles is None:
            titles = [image1, image2]
    
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12,6)
        )
    
        for ax, name, title in zip(
            axes,
            [image1, image2],
            titles
        ):
    
            image = self.images[name]
            wcs = self.wcs[name]
    
            # coords
            if isinstance(center, list):
    
                x, y = wcs.all_world2pix(
                    center[0],
                    center[1],
                    0
                )
    
            else:
    
                x, y = center
    
            x = int(x)
            y = int(y)
    
            half = size // 2
    
            cut = image[
                y-half:y+half,
                x-half:x+half
            ]
    
            vmin = np.nanpercentile(cut, 1)
            vmax = np.nanpercentile(cut, 99.7)
    
            ax.imshow(
                cut,
                origin='lower',
                cmap='gray',
                vmin=vmin,
                vmax=vmax
            )
    
            ax.set_title(title)
    
        plt.tight_layout()
        plt.show()
    
    def qa_rgb_overlay(
        self,
        image1,
        image2,
        center,
        size=200
    ):
    
        """
        RGB overlay for alignment QA.
    
        image1 -> red
        image2 -> cyan
    
        Perfect alignment -> white
        """
    
        im1 = self.images[image1]
        im2 = self.images[image2]
    
        wcs = self.wcs[image1]
    
        # coords
        if isinstance(center, list):
    
            x, y = wcs.all_world2pix(
                center[0],
                center[1],
                0
            )
    
        else:
    
            x, y = center
    
        x = int(x)
        y = int(y)
    
        half = size // 2
    
        cut1 = im1[
            y-half:y+half,
            x-half:x+half
        ]
    
        cut2 = im2[
            y-half:y+half,
            x-half:x+half
        ]
    
        # normalize
        cut1 = cut1 / np.nanpercentile(cut1, 99)
        cut2 = cut2 / np.nanpercentile(cut2, 99)
    
        cut1 = np.clip(cut1, 0, 1)
        cut2 = np.clip(cut2, 0, 1)
    
        rgb = np.zeros(
            (*cut1.shape, 3)
        )
    
        rgb[...,0] = cut1
        rgb[...,1] = cut2
        rgb[...,2] = cut2
    
        plt.figure(figsize=(7,7))
    
        plt.imshow(
            rgb,
            origin='lower'
        )
    
        plt.title(
            f'{image1}=red, {image2}=cyan'
        )
    
        plt.show()
    
    def qa_alignment_shift(
        self,
        image1,
        image2,
        center,
        size=300
    ):
    
        """
        Numerically estimate residual alignment offset.
        """
    
        im1 = self.images[image1]
        im2 = self.images[image2]
    
        wcs = self.wcs[image1]
    
        if isinstance(center, list):
    
            x, y = wcs.all_world2pix(
                center[0],
                center[1],
                0
            )
    
        else:
    
            x, y = center
    
        x = int(x)
        y = int(y)
    
        half = size // 2
    
        cut1 = im1[
            y-half:y+half,
            x-half:x+half
        ]
    
        cut2 = im2[
            y-half:y+half,
            x-half:x+half
        ]
    
        shift, error, phasediff = (
            phase_cross_correlation(
                np.nan_to_num(cut1),
                np.nan_to_num(cut2),
                upsample_factor=100
            )
        )
    
        print('--------------------------------')
        print('Alignment QA')
        print('--------------------------------')
        print(f'Shift (y,x): {shift}')
        print(f'Error: {error}')
        print('--------------------------------')
    
    def qa_convolution_residual(
        self,
        original_image,
        convolved_image,
        center,
        size=200
    ):
    
        """
        Show residuals after convolution.
    
        Useful for checking:
        - kernel centering
        - ringing
        - FFT failures
        """
    
        orig = self.images[original_image]
        conv = self.images[convolved_image]
    
        wcs = self.wcs[original_image]
    
        if isinstance(center, list):
    
            x, y = wcs.all_world2pix(
                center[0],
                center[1],
                0
            )
    
        else:
    
            x, y = center
    
        x = int(x)
        y = int(y)
    
        half = size // 2
    
        o = orig[
            y-half:y+half,
            x-half:x+half
        ]
    
        c = conv[
            y-half:y+half,
            x-half:x+half
        ]
    
        residual = o - c
    
        vmax = np.nanpercentile(
            np.abs(residual),
            99
        )
    
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15,5)
        )
    
        axes[0].imshow(
            o,
            origin='lower',
            cmap='gray'
        )
    
        axes[0].set_title('Original')
    
        axes[1].imshow(
            c,
            origin='lower',
            cmap='gray'
        )
    
        axes[1].set_title('Convolved')
    
        axes[2].imshow(
            residual,
            origin='lower',
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax
        )
    
        axes[2].set_title('Residual')
    
        plt.tight_layout()
        plt.show()
    
    def qa_apertures(
        self,
        image_name,
        location,
        radius,
        annulus_thickness,
        buffer=0*u.arcsec,
        size=200
    ):
    
        """
        Plot source aperture + background annulus.
        """
    
        image = self.images[image_name]
        header = self.headers[image_name]
        wcs = self.wcs[image_name]
    
        if isinstance(location, list):
    
            x, y = wcs.all_world2pix(
                location[0],
                location[1],
                0
            )
    
        else:
    
            x, y = location
    
        pixscale = abs(header['CDELT1']) * u.deg
    
        r_source = (
            radius / pixscale
        ).decompose().value
    
        r_in = (
            (radius + buffer) / pixscale
        ).decompose().value
    
        r_out = (
            (radius + buffer + annulus_thickness)
            / pixscale
        ).decompose().value
    
        x = int(x)
        y = int(y)
    
        half = size // 2
    
        cut = image[
            y-half:y+half,
            x-half:x+half
        ]
    
        vmin = np.nanpercentile(cut, 1)
        vmax = np.nanpercentile(cut, 99.7)
    
        fig, ax = plt.subplots(
            figsize=(7,7)
        )
    
        ax.imshow(
            cut,
            origin='lower',
            cmap='gray',
            vmin=vmin,
            vmax=vmax
        )
    
        # shift coords into cutout frame
        xc = half
        yc = half
    
        source = Circle(
            (xc, yc),
            r_source,
            edgecolor='lime',
            facecolor='none',
            linewidth=2
        )
    
        inner = Circle(
            (xc, yc),
            r_in,
            edgecolor='yellow',
            facecolor='none',
            linestyle='--'
        )
    
        outer = Circle(
            (xc, yc),
            r_out,
            edgecolor='red',
            facecolor='none'
        )
    
        ax.add_patch(source)
        ax.add_patch(inner)
        ax.add_patch(outer)
    
        ax.set_title(image_name)
    
        plt.show()
