import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
from matplotlib.widgets import Slider
from astropy.table import Table
import astropy.units as u
from astropy import constants as const

from reproject import reproject_interp
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
from Functions import *
from astropy.wcs import WCS
from reproject import reproject_interp as rpj
from astropy.convolution import convolve, convolve_fft
from scipy.ndimage import zoom, shift as ndi_shift
from photutils.centroids import centroid_quadratic
import time
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.wcs.utils import proj_plane_pixel_area

from astropy.visualization import ZScaleInterval
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

    # pixel area in steradians
    pixel_area_sr = proj_plane_pixel_area(wcs) * (np.pi/180.)**2

    # --------------------------------------------------
    # ACS/HST calibrated count rate images
    # --------------------------------------------------
    if bunit == 'W M-2 HZ-1 SR-1':
        print('Data was already in desired units.')
        return data, header

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

        data = data * 1e6 * 1e-26

    # --------------------------------------------------
    # Jy/sr
    # --------------------------------------------------
    elif bunit == 'JY/SR':
        print('converting data in image from JY/sr to W/m2/hz/sr')

        data = data * 1e-26

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
    
    def load_image(self, name, filename, hdu=None):

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
        out_file=None
    ):

        if out_file is None:

            out_file = (
                self.files[other_image]
                .replace('.fits', '_aligned.fits')
            )

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

        # Reload into object
        self.load_image(
            f'{other_image}_aligned',
            out_file
        )

    def sum_images(self, name1, name2, out_name=None, out_file=None, scales=[1,1]):
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
            print('No pixel units found in header under CUNIT1')
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

    def fft_convolve(self,
        image_name,
        kernel_filepath,
        out_name=None,
        normalize_kernel=True,
        preserve_nan=True,
        boundary='fill',
        fill_value=0.0,
        return_time=False
    ):
        """
        Convolve image using FFT convolution.

        Parameters
        ----------
        image : 2D ndarray
        kernel : 2D ndarray

        Returns
        -------
        convolved : ndarray
        elapsed_time : float (seconds)
        """
        
        #TJ start timer to keep track of how long this method takes to convolve
        t0 = time.perf_counter()
        kernel = reproject_kernel_to_image(kernel_filepath, self.headers[image_name], crop_size=None, normalize=True)

        image = self.images[image_name]

        #TJ fft convolve hates nans, replace them with zeros
        kernel = np.nan_to_num(kernel)

        if normalize_kernel:

            kernel /= np.sum(kernel)

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
            self.images[f'{image_name}_fft_conv'] = convolved
            self.headers[f'{image_name}_fft_conv'] = self.headers[image_name].copy()
            self.wcs[f'{image_name}_fft_conv'] = self.wcs[image_name].copy()
        else:
            self.images[out_name] = convolved
            self.headers[out_name] = self.headers[image_name].copy()
            self.wcs[out_name] = self.wcs[image_name].copy()
        
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
    
    def convolve(self,
        image_name,
        kernel_filepath,
        normalize_kernel=True,
        preserve_nan=True,
        boundary='fill',
        fill_value=0.0,
        return_time=False
    ):

        """
        Convolve image using direct linear convolution.
        THIS MAY TAKE A LOOONNG TIME...

        Parameters
        ----------
        image : 2D ndarray
        kernel : 2D ndarray

        Returns
        -------
        convolved : ndarray
        elapsed_time : float (seconds)
        """
        
        t0 = time.perf_counter()
        kernel = reproject_kernel_to_image(kernel_filepath, self.headers[image_name], crop_size=None, normalize=True)
    
        image = self.images[image_name]

        kernel = np.nan_to_num(kernel)

        if normalize_kernel:

            kernel /= np.sum(kernel)

        if preserve_nan:

            nan_mask = ~np.isfinite(image)

        image_filled = np.nan_to_num(image)

        convolved = convolve(
            image_filled,
            kernel,
            boundary=boundary,
            fill_value=fill_value,
            normalize_kernel=False,
            preserve_nan=False
        )

        if preserve_nan:

            convolved[nan_mask] = np.nan

        elapsed = time.perf_counter() - t0

        print('----------------------------------')
        print('Direct Convolution Complete')
        print('----------------------------------')
        print(f'Time elapsed : {elapsed:.3f} sec')
        print('----------------------------------')
        
        self.images['cont_conv'] = convolved
        self.headers['cont_conv'] = self.headers[image_name].copy()
        self.wcs['cont_conv'] = self.wcs[image_name].copy()
        if return_time:
            print(f'convolution took {elapsed} seconds')
            return elapsed

    def continuum_subtract(
        self,
        f187_name,
        continuum_name,
        scale_factor,
        out_name='cont_subtracted'
    ):

        subtracted = (
            self.images[f187_name] -
            scale_factor * self.images[continuum_name]
        )

        self.images[out_name] = subtracted
        self.headers[out_name] = self.headers[f187_name].copy()

        print(f'Created: {out_name}')

    def save_fits(
        self,
        image_name,
        output_file,
        scale=1
    ):

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
            pixel_area = get_pix_area(image_name)

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
        if 'CDELT1' in header:
            pixel_scale_deg = abs(header['CDELT1'])
        elif 'CD1_1' in header:
            pixel_scale_deg = abs(header['CD1_1'])
        else:
            print('Pixel size not found in header with key CDELT1 or CD1_1, aperture photometry failed')
            return
        #TJ convert to pixel units instead of angular
        source_radius_pixels = (radius.to_value(u.deg) / pixel_scale_deg)

        bg_inner_pixels = ((radius + buffer).to_value(u.deg) / pixel_scale_deg)

        bg_outer_pixels = ((radius + buffer + background_annulus_thickness).to_value(u.deg) / pixel_scale_deg)

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
            'background_per_pixel': np.nan*u.W / (u.m**2 * u.Hz),
            'source_area_pixels': np.nan*u.W / (u.m**2 * u.Hz),
            'annulus_area_pixels': np.nan*u.W / (u.m**2 * u.Hz)}

        bg_annulus = CircularAnnulus(
            (x, y),
            r_in=bg_inner_pixels,
            r_out=bg_outer_pixels
        )

        source_flux = aperture_photometry(
            image_quantity,
            source_aperture,
            method='exact'
        )['aperture_sum'][0] * pix_area
        source_area_pixels = (
            source_aperture.area
        )

        annulus_mask = bg_annulus.to_mask(method='exact')

        annulus_data = annulus_mask.multiply(image_quantity.value)

        annulus_weights = annulus_mask.data

        # VALID PIXELS
        # =====================================================

        valid = (
            np.isfinite(annulus_data) &
            (annulus_weights > 0)
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

        #TJ extract median background flux with proper units
        background_per_pixel = np.nanmedian(
            intrinsic_pixel_values
        ) * image_quantity.unit * pix_area

        annulus_area_pixels = np.sum(
            annulus_weights
        )
        
        #TJ now get background in source aperture by multiplying by source area
        # =====================================================

        background_flux = (
            background_per_pixel *
            source_area_pixels
        )

        net_flux = (
            source_flux -
            background_flux
        )
        
        return {
            'source_flux': source_flux,
            'background_flux': background_flux,
            'net_flux': net_flux,
            'background_per_pixel': background_per_pixel,
            'source_area_pixels': source_area_pixels,
            'annulus_area_pixels': annulus_area_pixels
        }

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
        feature_filter_file : str

        continuum_filter_file : str

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
    
    def display(self, names, loc, radius, ncols=3, cmap='viridis', zoom = 5):
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
        zoom : float, optional
            How many radii does image include (default = 5)
        """
        
        # Make sure loc is SkyCoord
        if not isinstance(loc, SkyCoord):
            loc_sky = SkyCoord(ra=loc[0]*u.deg, dec=loc[1]*u.deg, frame='icrs')
        else:
            loc_sky = loc

        n_images = len(names)
        nrows = int(np.ceil(n_images / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), 
                                subplot_kw={'projection': None})
        axes = np.atleast_1d(axes).ravel()  # Flatten in case of 1 row/col
        
        for ax, name in zip(axes, names):
            # Load FITS

            image = self.images[name]
            header = self.headers[name]
            wcs = self.wcs[name]


            try:
                pixel_scale = np.abs(wcs.wcs.cd[0][0]) *3600
            except:
                pixel_scale = np.abs(wcs.wcs.cdelt[0]) * 3600  # arcsec/pixel
            # Make cutout
            cutout = Cutout2D(image, position=loc_sky, size=(radius*zoom, radius*zoom), wcs=wcs)

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
            ax.set_title(name, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide empty panels if n_images doesn’t fill full grid
        for ax in axes[n_images:]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

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
