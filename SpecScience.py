from __future__ import annotations

import os

from ImageScience import ImageScience
from Functions import *

import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.wcs import WCS
from spectral_cube import SpectralCube
from astropy.constants import c
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import rotate as ndimage_rotate
from scipy.ndimage import zoom

from scipy.ndimage import median_filter

import warnings
from typing import Optional
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import FITSFixedWarning
from astropy.wcs.utils import fit_wcs_from_points
from photutils.detection import DAOStarFinder
from photutils.centroids import centroid_2dg
from scipy.spatial import cKDTree
from skimage.transform import estimate_transform  # 'similarity': rot+trans+scale



def _combine_spectra(base, new, side='right', method='add_shift'):
    """
    Combine two spectra dicts (keys: 'wavelength', 'F_nu') into one.

    Parameters
    ----------
    base : dict   — fixed reference spectrum
    new  : dict   — spectrum to combine with base
    side : str    — 'right' (new is redward) or 'left' (new is blueward).
                    Ignored for method='mean'.
    method : str
        'add_shift'  — shift new additively to match base in overlap, then concatenate
        'mult_shift' — shift new multiplicatively to match base in overlap, then concatenate
        'mean'       — average the overlap region on the finer wavelength grid, concatenate tails
    """
    bw = np.array(base['wavelength'], dtype=float)
    bi = np.array(base['F_nu'],       dtype=float)
    nw = np.array(new['wavelength'],  dtype=float)
    ni = np.array(new['F_nu'],        dtype=float)
    #TJ trim nans
    valid_b = np.isfinite(bi)
    bw = bw[valid_b]
    bi = bi[valid_b]

    valid_n = np.isfinite(ni)
    nw = nw[valid_n]
    ni = ni[valid_n]
    if bw.size == 0: return {'wavelength': nw.copy(), 'F_nu': ni.copy()}
    if nw.size == 0: return {'wavelength': bw.copy(), 'F_nu': bi.copy()}

    overlap_min = max(bw[0], nw[0])
    overlap_max = min(bw[-1], nw[-1])
    has_overlap = overlap_max > overlap_min

    # ------------------------------------------------------------------ #
    #  Shift methods: offset new to match base, keep base wavelength grid  #
    # ------------------------------------------------------------------ #
    if method in ('add_shift', 'mult_shift'):
        offset = 1.0 if method == 'mult_shift' else 0.0   # neutral values

        if has_overlap:
            mask_b = (bw >= overlap_min) & (bw <= overlap_max) & np.isfinite(bi)
            mask_n = (nw >= overlap_min) & (nw <= overlap_max) & np.isfinite(ni)
            if mask_b.any() and mask_n.sum() >= 2:
                try:
                    interp_ni = np.interp(bw[mask_b], nw[mask_n], ni[mask_n])
                    if method == 'add_shift':
                        offset = np.nanmedian(bi[mask_b] - interp_ni)
                    else:  # mult_shift
                        offset = np.nanmedian(bi[mask_b] / interp_ni)
                except Exception:
                    pass  # offset stays neutral
        else:
            # No overlap — estimate from edges
            print('Warning: no overlap detected between spectra, estimating offset from edges, this is not intended to be rigorous')
            nmatch = min(10, bw.size, nw.size)
            b_edge = np.nanmedian(bi[-nmatch:] if side == 'right' else bi[:nmatch])
            n_edge = np.nanmedian(ni[:nmatch]  if side == 'right' else ni[-nmatch:])
            offset = (b_edge - n_edge) if method == 'add_shift' else (b_edge / n_edge)

        ni_corr = ni + offset if method == 'add_shift' else ni * offset

        # Append only the non-overlapping tail of new
        if side == 'right':
            tail = nw > bw[-1]
            if tail.any():
                return {'wavelength': np.concatenate([bw,         nw[tail]]),
                        'F_nu':       np.concatenate([bi, ni_corr[tail]])}
        else:  # left
            tail = nw < bw[0]
            if tail.any():
                return {'wavelength': np.concatenate([nw[tail],         bw]),
                        'F_nu':       np.concatenate([ni_corr[tail],    bi])}
        # new entirely contained within base — nothing to add
        return {'wavelength': bw.copy(), 'F_nu': bi.copy()}

    # ------------------------------------------------------------------ #
    #  Mean method: average overlap on finer grid, concatenate tails       #
    # ------------------------------------------------------------------ #
    elif method == 'mean':
        if not has_overlap:
            # No overlap — just concatenate in wavelength order
            print('Warning: no overlap detected between spectra, concatenating without averaging')
            if bw[0] < nw[0]:
                return {'wavelength': np.concatenate([bw, nw]),
                        'F_nu':       np.concatenate([bi, ni])}
            else:
                return {'wavelength': np.concatenate([nw, bw]),
                        'F_nu':       np.concatenate([ni, bi])}

        # Pick finer grid for the overlap region
        b_res = np.mean(np.diff(bw[(bw >= overlap_min) & (bw <= overlap_max)]))
        n_res = np.mean(np.diff(nw[(nw >= overlap_min) & (nw <= overlap_max)]))
        if b_res <= n_res:
            fine_w, fine_f, coarse_w, coarse_f = bw, bi, nw, ni
        else:
            fine_w, fine_f, coarse_w, coarse_f = nw, ni, bw, bi

        ov_mask      = (fine_w >= overlap_min) & (fine_w <= overlap_max)
        interp_coarse = np.interp(fine_w[ov_mask], coarse_w, coarse_f)
        ov_wl        = fine_w[ov_mask]
        ov_fl        = 0.5 * (fine_f[ov_mask] + interp_coarse)

        # Tails: take from whichever dataset extends further on each side
        left_src  = (bw, bi) if bw[0]  < nw[0]  else (nw, ni)
        right_src = (bw, bi) if bw[-1] > nw[-1] else (nw, ni)
        left_wl,  left_fl  = left_src[0][left_src[0]   < overlap_min], left_src[1][left_src[0]   < overlap_min]
        right_wl, right_fl = right_src[0][right_src[0] > overlap_max], right_src[1][right_src[0] > overlap_max]

        wl = np.concatenate([left_wl, ov_wl, right_wl])
        fl = np.concatenate([left_fl, ov_fl, right_fl])
        order = np.argsort(wl)
        return {'wavelength': wl[order], 'F_nu': fl[order]}

    else:
        raise ValueError(f"method must be 'add_shift', 'mult_shift', or 'mean', got '{method}'")




class SpecScience:
    'Tools for IFU cube analysis'
    def __init__(self):
        #TJ initialize dictionary of class attributes
        self.cubes = {} #TJ full 4D datacube
        self.headers = {} #TJ header from the file (may be modified with functions)
        self.wavelengths = {}
        self.files = {} #TJ paths to files
        self.wcs = {} #TJ WCS information from header files
        self.spectra = {} #TJ dictionary with keys for wavelength, frequency, F_nu, and F_lambda
        self.data = {}
        self.images = {}
    
    def load_cube(self, name, filename, hdu=None):

        """
        Load FITS image and convert to

            F_nu [W m^-2 Hz^-1 sr^-1]
        """

        hdul = fits.open(filename)

        self.files[name] = filename
        if hdu is not None:
            cube = SpectralCube.read(filename, hdu=hdu)
            header = hdul[hdu].header.copy()
            wcs = WCS(header, hdul)
        else:
            try:

                cube = SpectralCube.read(filename, hdu='SCI')
                header = hdul['SCI'].header.copy()
                wcs = WCS(header, hdul)

            except Exception:

                print('SCI extension not found, using primary')

                cube = SpectralCube.read(filename)
                header = hdul[0].header.copy()
                wcs = WCS(header, hdul)
        if cube.unit == 'MJy / sr':
            cube = cube.to(u.W / (u.m**2 * u.Hz * u.sr))
        elif cube.unit == 'MJy':
            cube = cube.to(u.W / (u.m**2 * u.Hz))
        self.wavelengths[name] = cube.spectral_axis.to(u.m).copy()
        self.cubes[name] = cube
        self.headers[name] = header
        self.wcs[name] = wcs

        print(
            f'Loaded: {name}'
            f'[cube units: {cube.unit}]'
        )

    def align_to_image(self,
        image_object,
        image_name,
        cube_name,
        filter_name,
        out_name = None,
        out_file = None,
        fwhm = 2.3,
        detection_threshold = 5.0,
        max_offset_pixels = 10.0,
        min_matches = 3,
        show_diagnostic_plots = True,
    ):
        """Align an IFU cube's WCS to a broadband reference image.

        Parameters
        ----------
        image_object : ImageScience
            Container holding the reference broadband image.
        image_name : str
            Key into ``image_object.images`` for the reference image (2-D array).
        spec_object : SpecScience
            Container holding the IFU data cube.
        cube_name : str
            Key into ``spec_object.cubes`` for the cube to be aligned.
        filter_name : str
            Name of the filter to be passed to
            ``SpecScience.create_synthetic_image``.
        out_name : str, optional
            Key under which the aligned cube / WCS / header are stored on
            ``spec_object``.  Defaults to ``f'{cube_name}_aligned'``.
        out_file : str, optional
            If given, the aligned cube is also written to this FITS path and
            ``spec_object.files[out_name]`` is set.
        fwhm : float
            Expected PSF FWHM in pixels used by the source finder (default 3.0).
        detection_threshold : float
            Detection threshold in units of the image background RMS (default 5σ).
        max_offset_pixels : float
            Maximum pixel distance when matching centroids between the two images
            (default 10 px, generous to accommodate the ≤3 px offset stated in
            the problem plus some margin).
        min_matches : int
            Minimum number of matched centroid pairs required to solve the
            transform (default 3).
        show_diagnostic_plots : bool
            If True (default), display a two-panel diagnostic figure after
            alignment: the first row shows detected sources overlaid on the
            synthetic IFU image and the reference cutout with RA/Dec grids;
            the second row shows collapsed white-light images of the original
            and aligned cube side-by-side with RA/Dec grids for comparison.

        Returns
        -------
        aligned_cube : np.ndarray  (n_wave, ny, nx)
            The cube array with its spatial axes remapped to the new WCS grid.
            The same array is stored in ``spec_object.cubes[out_name]``.

        Raises
        ------
        RuntimeError
            If fewer than ``min_matches`` centroid pairs can be matched.
        """
            
        def _make_ifu_footprint_cutout(
            ref_image,
            ref_wcs,
            synth_image,
            synth_wcs,
        ):
            """Return a Cutout2D of *ref_image* covering the IFU spatial footprint.

            The cutout is sized to the IFU footprint plus a 20 % border so that
            nearby sources just outside the IFU edge can still be detected and used
            as tie-points.
            """
            ny_ifu, nx_ifu = synth_image.shape

            # Corner pixels of the synthetic image (0-indexed)
            corners_pix = np.array([
                [0,          0         ],
                [nx_ifu - 1, 0         ],
                [nx_ifu - 1, ny_ifu - 1],
                [0,          ny_ifu - 1],
            ])                                                          # (4, 2) [x, y]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                corners_sky = synth_wcs.all_pix2world(corners_pix, 0)  # (4, 2) [ra, dec]

            # Project IFU corners into the reference image frame
            corners_ref_pix = ref_wcs.all_world2pix(corners_sky, 0)    # (4, 2) [x, y]

            x_min, x_max = corners_ref_pix[:, 0].min(), corners_ref_pix[:, 0].max()
            y_min, y_max = corners_ref_pix[:, 1].min(), corners_ref_pix[:, 1].max()

            # Add a 20 % border
            dx = 0*(x_max - x_min) * 0.20
            dy = 0*(y_max - y_min) * 0.20
            x_min -= dx;  x_max += dx
            y_min -= dy;  y_max += dy

            # Centre and size of the cutout
            x_cen = (x_min + x_max) / 2.0
            y_cen = (y_min + y_max) / 2.0
            size  = (int(np.ceil(y_max - y_min)), int(np.ceil(x_max - x_min)))  # (ny, nx)

            # Cutout2D expects the centre as a (y, x) pixel tuple or SkyCoord
            position = (x_cen, y_cen)   # (x, y) — Cutout2D accepts both orderings
            # Use the safer SkyCoord path to avoid row/col confusion
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                sky_cen_arr = ref_wcs.all_pix2world([[x_cen, y_cen]], 0)[0]
            sky_cen = SkyCoord(ra=sky_cen_arr[0] * u.deg, dec=sky_cen_arr[1] * u.deg)

            cutout = Cutout2D(
                ref_image,
                position=sky_cen,
                size=size,
                wcs=ref_wcs,
                mode="partial",
                fill_value=0.0,
            )
            return cutout
    
        def _find_centroids(
            image,
            fwhm,
            threshold,
        ):
            """Detect point sources and return sub-pixel centroids (x, y).

            Uses DAOStarFinder for detection and 2-D Gaussian centroiding for
            sub-pixel refinement.  Returns an (N, 2) array of [x, y] positions.
            """
            from astropy.stats import mad_std

            # Robust background estimation
            bkg = np.nanmedian(image)
            bkg_rms = mad_std(image, ignore_nan=True)

            finder = DAOStarFinder(
                fwhm=fwhm,
                threshold=threshold * bkg_rms,
                brightest=50,     # cap to avoid confusion in crowded fields
                exclude_border=True,
            )
            sources = finder(image - bkg)

            if sources is None or len(sources) == 0:
                return np.empty((0, 2))

            box_half = max(int(np.ceil(fwhm * 2)), 5)
            centroids = []
            ny, nx = image.shape

            for row in sources:
                x0 = int(round(row["xcentroid"]))
                y0 = int(round(row["ycentroid"]))

                # Bounds check
                x_lo = max(x0 - box_half, 0);  x_hi = min(x0 + box_half + 1, nx)
                y_lo = max(y0 - box_half, 0);  y_hi = min(y0 + box_half + 1, ny)
                if (x_hi - x_lo) < 3 or (y_hi - y_lo) < 3:
                    centroids.append([row["xcentroid"], row["ycentroid"]])
                    continue

                stamp = image[y_lo:y_hi, x_lo:x_hi] - bkg
                stamp = np.where(np.isfinite(stamp), stamp, 0.0)

                try:
                    cx, cy = centroid_2dg(stamp)
                    centroids.append([cx + x_lo, cy + y_lo])
                except Exception:
                    centroids.append([row["xcentroid"], row["ycentroid"]])

            return np.array(centroids)   # (N, 2) [x, y]

        def _pixels_to_sky(
            pixel_xy,
            wcs,
        ):
            """Convert (N,2) [x,y] pixel array to a SkyCoord array."""
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                world = wcs.all_pix2world(pixel_xy, 0)   # (N, 2) [ra, dec]
            return SkyCoord(ra=world[:, 0] * u.deg, dec=world[:, 1] * u.deg)

        def _match_centroids(
            synth_sky,
            synth_in_ref_pix,   # synthetic sources projected into cut_wcs frame
            ref_sky,
            ref_pix,            # reference sources in cut_wcs frame
            max_dist_px,
        ):
            """Nearest-neighbour matching in the reference pixel frame.

            Returns
            -------
            matched_synth_sky : SkyCoord  (M,)
            matched_ref_pix   : np.ndarray (M, 2) [x, y]
            """
            if len(synth_in_ref_pix) == 0 or len(ref_pix) == 0:
                return SkyCoord(ra=[] * u.deg, dec=[] * u.deg), np.empty((0, 2))

            tree = cKDTree(ref_pix)
            dists, idx = tree.query(synth_in_ref_pix, k=1, workers=-1)

            mask = dists <= max_dist_px

            matched_synth_sky = synth_sky[mask]
            matched_ref_pix   = ref_pix[idx[mask]]

            return matched_synth_sky, matched_ref_pix

        def _fit_new_wcs(
            sky_coords,
            pixel_xy,
            ref_sky,
            ref_pix_in_ref_frame,
            ref_wcs,
            synth_wcs,
        ):
            """Derive a corrected 2-D WCS for the IFU spatial plane.

            The approach:
            - We have matched pairs (synth_pixel ↔ ref_pixel in cut_wcs).
            - Estimate the similarity transform (rotation + translation + isotropic
                scale) that maps synth pixels → ref pixels in the *sky* plane.
            - Back out the corrected CD matrix and CRPIX for the synth WCS.

            Uses skimage's estimate_transform('similarity') for robustness with
            as few as 2 matches (3+ preferred).
            """
            # Sky angular offsets (arcsec) relative to the synth CRVAL, used as a
            # common intermediate frame.
            crval_ra  = synth_wcs.wcs.crval[0]
            crval_dec = synth_wcs.wcs.crval[1]
            ref_coord = SkyCoord(ra=crval_ra * u.deg, dec=crval_dec * u.deg)

            def sky_to_tangent(coords: SkyCoord):
                """Gnomonic projection onto synth CRVAL tangent plane (arcsec)."""
                dra  = (coords.ra.deg  - crval_ra)  * np.cos(np.radians(crval_dec)) * 3600.0
                ddec = (coords.dec.deg - crval_dec) * 3600.0
                return np.column_stack([dra, ddec])

            src_tan = sky_to_tangent(sky_coords)   # synth pixel → sky tangent (arcsec)
            dst_tan = sky_to_tangent(ref_sky)      # ref   pixel → sky tangent (arcsec)

            # Map synth pixels to tangent plane via old WCS, and ref pixels to tangent
            # plane via ref WCS; then solve for the 2-D rigid transform between them.
            #
            # Both tangent-plane representations are in the same sky frame, so the
            # transform is the pure astrometric offset.
            #
            # We use the pixel coordinate pairs directly:
            #   src: pixel coords of sources *in the synthetic image*
            #   dst: pixel coords of the *same* sources in the reference cutout

            tform = estimate_transform(
                "similarity",
                src=pixel_xy,                   # (M, 2) synth pixels
                dst=ref_pix_in_ref_frame,       # (M, 2) ref cutout pixels
            )

            # Decompose the similarity transform into its components
            scale    = tform.scale
            rotation = tform.rotation            # radians
            tx, ty   = tform.translation         # pixels (in ref cutout frame)

            # Build the corrected WCS by updating the CD matrix of the synth WCS.
            # The CD matrix encodes scale + rotation; the corrected version
            # incorporates the additional rotation and scale found above.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                new_wcs = synth_wcs.deepcopy()

            # Retrieve or reconstruct the CD matrix
            if hasattr(new_wcs.wcs, "cd") and new_wcs.wcs.cd is not None:
                cd_old = np.array(new_wcs.wcs.cd)
            else:
                # Build from CDELT + PC matrix if CD is absent
                cdelt = new_wcs.wcs.cdelt[:2]
                pc    = new_wcs.wcs.get_pc()[:2, :2]
                cd_old = pc * cdelt[np.newaxis, :]

            # Rotation matrix corresponding to the additional angular offset
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            R = np.array([[cos_r, -sin_r],
                        [sin_r,  cos_r]])

            cd_new = scale * (R @ cd_old)
            new_wcs.wcs.cd = cd_new

            # Update CRPIX: the old CRPIX in synth frame maps to a new position after
            # the transform.  We keep the sky value at CRPIX (CRVAL) unchanged and
            # adjust CRPIX so the new WCS still passes through the same sky point.
            # Because CRVAL is unchanged, and the CD matrix has been updated, the
            # CRPIX must shift to compensate for the translational component.
            #
            # In the ref cutout frame the old CRPIX would map to:
            crpix_old = np.array(new_wcs.wcs.crpix)  # (x, y), 1-indexed FITS convention
            crpix_in_ref = tform(crpix_old - 1)       # 0-indexed transform input
            # Convert back: CRPIX should remain where it is sky-wise; instead we
            # embed the full transform by recomputing CRVAL at the transformed CRPIX
            # and then updating CRPIX to the inverse-transformed value.

            # Simpler & numerically stable: keep CRPIX pixel location fixed in the
            # new cube and update CRVAL to the sky coordinate that the *reference
            # WCS* says corresponds to the transformed centroid.
            crpix_0idx = crpix_old - 1               # to 0-indexed
            crpix_trans = tform(crpix_0idx.reshape(1, 2))[0]  # where CRPIX maps in ref frame

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                new_crval = ref_wcs.all_pix2world([crpix_trans], 0)[0]  # sky at new CRPIX

            new_wcs.wcs.crval = new_crval
            new_wcs.wcs.crpix = crpix_old   # CRPIX stays at the same synthetic pixel

            new_wcs.wcs.set()
            return new_wcs

        def _remap_cube(
            cube,
            old_header,
            old_spatial_wcs,
            new_spatial_wcs,
        ):
            """Remap every wavelength slice of *cube* onto the grid defined by
            *new_spatial_wcs* using scipy's affine map (preserves flux density).

            Returns the remapped cube and an updated FITS header.
            """
            from scipy.ndimage import affine_transform

            n_wave, ny, nx = cube.shape

            # Compute the pixel-to-pixel transform between old and new spatial WCS.
            # We work entirely in pixel space: for every pixel (x, y) in the new
            # grid, find the corresponding (x, y) in the old grid.

            # Sample the transform on the four corners + centre
            sample_pix = np.array([
                [0,        0       ],
                [nx - 1,   0       ],
                [nx - 1,   ny - 1  ],
                [0,        ny - 1  ],
                [nx / 2,   ny / 2  ],
            ])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                sky_samples = new_spatial_wcs.all_pix2world(sample_pix, 0)
                old_pix_samples = old_spatial_wcs.all_world2pix(sky_samples, 0)

            # Estimate the linear (affine) part of the transform from new→old
            # using the corner correspondences.
            tform = estimate_transform("affine", src=sample_pix, dst=old_pix_samples)

            # scipy affine_transform expects a (2, 2) matrix and (2,) offset
            # for a 2-D image; the transform maps OUTPUT coords → INPUT coords.
            matrix = tform.params[:2, :2]     # linear part
            offset = tform.params[:2,  2]     # translation

            aligned = np.empty_like(cube)
            for i in range(n_wave):
                slc = cube[i].astype(float)
                slc = np.where(np.isfinite(slc), slc, 0.0)
                aligned[i] = affine_transform(
                    slc,
                    matrix=matrix,
                    offset=offset,
                    output_shape=(ny, nx),
                    order=3,          # cubic interpolation
                    mode="constant",
                    cval=np.nan,
                    prefilter=True,
                )

            # Build the new header
            import copy
            new_header = copy.deepcopy(old_header)

            # Inject the corrected spatial WCS into the 3-D header
            spatial_hdr = new_spatial_wcs.to_header(relax=True)
            for key, val in spatial_hdr.items():
                # Only overwrite spatial axes (1 and 2); leave spectral axis (3) alone
                if any(key.endswith(s) for s in ("1", "2", "1_2", "2_1")):
                    new_header[key] = val
                elif key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
                            "CD1_1",  "CD1_2",  "CD2_1",  "CD2_2",
                            "CDELT1", "CDELT2", "PC1_1",  "PC1_2",
                            "PC2_1",  "PC2_2",  "CTYPE1",  "CTYPE2",
                            "CUNIT1", "CUNIT2"):
                    new_header[key] = val

            return aligned, new_header

        def _plot_alignment_diagnostics(
            synth_image,
            synth_wcs,
            synth_centroids,
            matched_synth_pix,
            cut_image,
            cut_wcs,
            ref_centroids,
            matched_ref_pix,
            original_cube,
            original_wcs,
            aligned_cube,
            aligned_wcs,
            filter_name,
        ):
            """Display a 2×2 diagnostic figure.

            Row 1 — Source detection:
                Left  : synthetic IFU image with all detected centroids (white circles)
                        and matched tie-points (yellow crosses) on a RA/Dec grid.
                Right : reference image cutout with all detected centroids (white
                        circles) and matched tie-points (yellow crosses) on a RA/Dec
                        grid.

            Row 2 — Before / after alignment:
                Left  : white-light collapse of the *original* cube on a RA/Dec grid.
                Right : white-light collapse of the *aligned* cube on a RA/Dec grid.
                        Matched source positions from the reference (yellow crosses)
                        are over-plotted in sky coordinates so any residual offset is
                        immediately visible.
            """
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            from astropy.visualization import ZScaleInterval, ImageNormalize
            from astropy.visualization.wcsaxes import WCSAxes

            zscale = ZScaleInterval()

            def _norm(img):
                try:
                    finite = img[np.isfinite(img)].value
                except:
                    finite = img[np.isfinite(img)]
                if finite.size == 0:
                    return ImageNormalize(vmin=0, vmax=1)
                return ImageNormalize(img.data, stretch=AsinhStretch(),
                                                vmin=0, vmax=np.percentile(img.data, 99))

            def _wl(cube):
                """Robust white-light image: nanmedian across wavelength axis."""
                return np.nanmedian(cube, axis=0)

            # Matched source positions in sky coords (for reuse in row 2)
            matched_ref_sky = _pixels_to_sky(matched_ref_pix, cut_wcs)

            fig = plt.figure(figsize=(14, 12))
            fig.suptitle(
                f"WCS Alignment Diagnostics — filter: {filter_name}",
                fontsize=14, fontweight="bold", y=0.98,
            )

            # ------------------------------------------------------------------
            # Row 1, Left: synthetic IFU image
            # ------------------------------------------------------------------
            ax1 = fig.add_subplot(2, 2, 1, projection=synth_wcs)
            norm1 = _norm(synth_image)
            ax1.imshow(synth_image, origin="lower", norm=norm1, cmap="gray_r",
                    interpolation="nearest")

            # All detected centroids
            if len(synth_centroids) > 0:
                ax1.scatter(
                    synth_centroids[:, 0], synth_centroids[:, 1],
                    transform=ax1.get_transform("pixel"),
                    s=60, facecolors="none", edgecolors="white", linewidths=1.2,
                    label=f"Detected ({len(synth_centroids)})",
                )
            # Matched tie-points
            if len(matched_synth_pix) > 0:
                ax1.scatter(
                    matched_synth_pix[:, 0], matched_synth_pix[:, 1],
                    transform=ax1.get_transform("pixel"),
                    marker="+", s=120, color="yellow", linewidths=1.8,
                    label=f"Matched ({len(matched_synth_pix)})",
                )

            _add_wcs_grid(ax1)
            ax1.set_title("Synthetic IFU image (source detection)", fontsize=10)
            ax1.legend(loc="upper right", fontsize=8, framealpha=0.6)

            # ------------------------------------------------------------------
            # Row 1, Right: reference cutout
            # ------------------------------------------------------------------
            ax2 = fig.add_subplot(2, 2, 2, projection=cut_wcs)

            norm2 = _norm(cut_image)
            ax2.imshow(cut_image, origin="lower", norm=norm2, cmap="gray_r",
                    interpolation="nearest")

            if len(ref_centroids) > 0:
                ax2.scatter(
                    ref_centroids[:, 0], ref_centroids[:, 1],
                    transform=ax2.get_transform("pixel"),
                    s=60, facecolors="none", edgecolors="white", linewidths=1.2,
                    label=f"Detected ({len(ref_centroids)})",
                )
            if len(matched_ref_pix) > 0:
                ax2.scatter(
                    matched_ref_pix[:, 0], matched_ref_pix[:, 1],
                    transform=ax2.get_transform("pixel"),
                    marker="+", s=120, color="yellow", linewidths=1.8,
                    label=f"Matched ({len(matched_ref_pix)})",
                )

            _add_wcs_grid(ax2)
            ax2.set_title("Reference image cutout (source detection)", fontsize=10)
            ax2.legend(loc="upper right", fontsize=8, framealpha=0.6)

            # ------------------------------------------------------------------
            # Row 2, Left: original cube white-light with old WCS
            # ------------------------------------------------------------------
            ax3 = fig.add_subplot(2, 2, 3, projection=original_wcs)
            wl_orig = _wl(original_cube)
            ax3.imshow(wl_orig, origin="lower", norm=_norm(wl_orig), cmap="inferno",
                    interpolation="nearest")

            # Show where the reference tie-points *should* land after alignment
            # (projected into the original WCS frame so any offset is visible)
            if len(matched_ref_sky) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    orig_pix = np.column_stack(
                        original_wcs.all_world2pix(
                            matched_ref_sky.ra.deg, matched_ref_sky.dec.deg, 0
                        )
                    )
                ax3.scatter(
                    orig_pix[:, 0], orig_pix[:, 1],
                    transform=ax3.get_transform("pixel"),
                    marker="+", s=120, color="cyan", linewidths=1.8,
                    label="Ref tie-points (orig WCS)",
                )

            _add_wcs_grid(ax3)
            ax3.set_title("Original cube — white-light collapse", fontsize=10)
            ax3.legend(loc="upper right", fontsize=8, framealpha=0.6)

            # ------------------------------------------------------------------
            # Row 2, Right: aligned cube white-light with new WCS
            # ------------------------------------------------------------------
            ax4 = fig.add_subplot(2, 2, 4, projection=aligned_wcs)
            wl_aligned = _wl(aligned_cube)
            ax4.imshow(wl_aligned, origin="lower", norm=_norm(wl_aligned), cmap="inferno",
                    interpolation="nearest")

            # Same reference tie-points — should now coincide with sources
            if len(matched_ref_sky) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    new_pix = np.column_stack(
                        aligned_wcs.all_world2pix(
                            matched_ref_sky.ra.deg, matched_ref_sky.dec.deg, 0
                        )
                    )
                ax4.scatter(
                    new_pix[:, 0], new_pix[:, 1],
                    transform=ax4.get_transform("pixel"),
                    marker="+", s=120, color="cyan", linewidths=1.8,
                    label="Ref tie-points (new WCS)",
                )

            _add_wcs_grid(ax4)
            ax4.set_title("Aligned cube — white-light collapse", fontsize=10)
            ax4.legend(loc="upper right", fontsize=8, framealpha=0.6)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        def _add_wcs_grid(ax):
            """Add a clean RA/Dec overlay grid to a WCSAxes panel."""
            ax.coords.grid(color="white", alpha=0.4, linestyle="--", linewidth=0.6)

            ra_ax  = ax.coords["ra"]
            dec_ax = ax.coords["dec"]

            ra_ax.set_axislabel("RA (J2000)", fontsize=8)
            dec_ax.set_axislabel("Dec (J2000)", fontsize=8)

            ra_ax.set_ticklabel(size=7, exclude_overlapping=True)
            dec_ax.set_ticklabel(size=7, exclude_overlapping=True)

            ra_ax.set_major_formatter("hh:mm:ss.s")
            dec_ax.set_major_formatter("dd:mm:ss")

            ra_ax.set_ticks(number=4)
            dec_ax.set_ticks(number=4)

        def _write_cube_to_fits(aligned_cube, new_header, out_file):
            """Write the aligned cube to a FITS file."""

            hdu = fits.PrimaryHDU(data=cube, header=header)
            hdu.writeto(filepath, overwrite=True)
            print(f"[align_cube_to_image] Aligned cube written to '{filepath}'.")


        if out_name is None:
            out_name = f"{cube_name}_aligned"

        # ------------------------------------------------------------------
        # 1.  Build the synthetic image from the cube
        # ------------------------------------------------------------------
        synth_name = f"synthetic_{filter_name}"
        self.create_synthetic_image(cube_name, filter_name, out_name=synth_name)

        synth_image: np.ndarray = self.images[synth_name]   # (ny_ifu, nx_ifu)
        synth_wcs: WCS = self.wcs[synth_name]               # 2-D spatial WCS

        # ------------------------------------------------------------------
        # 2.  Cut the reference image down to the IFU footprint
        # ------------------------------------------------------------------
        ref_image = image_object.images[image_name]    # (NY, NX) big array
        ref_wcs= image_object.wcs[image_name]                # 2-D WCS

        cutout = _make_ifu_footprint_cutout(
            ref_image, ref_wcs, synth_image, synth_wcs
        )
        cut_image= cutout.data                        # (ny_cut, nx_cut)
        cut_wcs= cutout.wcs                                  # WCS of the cutout
        # ------------------------------------------------------------------
        # 3.  Detect & fit centroids in both images
        # ------------------------------------------------------------------
        synth_centroids = _find_centroids(synth_image, fwhm=fwhm,
                                        threshold=detection_threshold)
        ref_centroids   = _find_centroids(cut_image,   fwhm=fwhm,
                                        threshold=detection_threshold)

        if len(synth_centroids) == 0 or len(ref_centroids) == 0:
            raise RuntimeError(
                "Source detection failed in one or both images. "
                "Try lowering detection_threshold or adjusting fwhm."
            )

        # ------------------------------------------------------------------
        # 4.  Convert pixel centroids → sky coords, then → pixels in the
        #     *other* frame, so we can match them in a common frame.
        #
        #     Strategy: project everything to sky (RA/Dec), then re-project
        #     into the reference (cutout) pixel frame for matching.
        # ------------------------------------------------------------------
        synth_sky = _pixels_to_sky(synth_centroids, synth_wcs)   # SkyCoord array
        ref_sky   = _pixels_to_sky(ref_centroids,   cut_wcs)

        # Re-project synthetic sources into the cutout's pixel frame for matching
        synth_in_ref_pix = np.column_stack(
            cut_wcs.all_world2pix(synth_sky.ra.deg, synth_sky.dec.deg, 0)
        )                                                          # (N, 2)  [x, y]

        ref_pix = ref_centroids                                    # already in cut_wcs frame

        # ------------------------------------------------------------------
        # 5.  Match centroids (nearest-neighbour with distance cap)
        # ------------------------------------------------------------------
        matched_synth_sky, matched_ref_pix = _match_centroids(
            synth_sky, synth_in_ref_pix,
            ref_sky,   ref_pix,
            max_dist_px=max_offset_pixels,
        )

        if len(matched_synth_sky) < min_matches:
            raise RuntimeError(
                f"Only {len(matched_synth_sky)} centroid matches found "
                f"(need ≥ {min_matches}).  "
                "Try raising max_offset_pixels or lowering detection_threshold."
            )

        # ------------------------------------------------------------------
        # 6.  Fit a new WCS for the synthetic image / cube spatial plane
        #     using the matched (sky → pixel-in-synth) pairs.
        #
        #     We re-derive the pixel coordinates in the *synthetic* frame for
        #     each matched sky position.
        # ------------------------------------------------------------------
        matched_synth_pix = np.column_stack(
            synth_wcs.all_world2pix(
                matched_synth_sky.ra.deg,
                matched_synth_sky.dec.deg,
                0,
            )
        )                                                         # (M, 2) [x, y]

        new_synth_wcs = _fit_new_wcs(
            sky_coords=matched_synth_sky,
            pixel_xy=matched_synth_pix,
            ref_sky=matched_synth_sky,
            ref_pix_in_ref_frame=matched_ref_pix,
            ref_wcs=cut_wcs,
            synth_wcs=synth_wcs,
        )

        # ------------------------------------------------------------------
        # 7.  Apply the derived transform to every wavelength slice of the
        #     cube and rebuild with the corrected WCS.
        # ------------------------------------------------------------------
        cube: np.ndarray = self.cubes[cube_name]           # (n_wave, ny, nx)
        old_cube_header  = self.headers[cube_name]

        aligned_cube, new_header = _remap_cube(
            cube, old_cube_header, synth_wcs, new_synth_wcs
        )

        new_header['BUNIT'] = 'W Hz-1 sr-1 m-2'
        new_wcs_3d = WCS(new_header)

        hdu = fits.PrimaryHDU(data=aligned_cube, header=new_header)
        
        self.headers[out_name] = new_header

        spectral_cube_obj = SpectralCube.read(fits.HDUList([hdu]))

        self.wcs[out_name]     = new_wcs_3d
        self.headers[out_name] = new_header
        self.cubes[out_name] = spectral_cube_obj
        self.wavelengths[out_name] = self.wavelengths[cube_name]
        if out_file is not None:
            _write_cube_to_fits(aligned_cube, new_header, out_file)
            self.files[out_name] = out_file

        # ------------------------------------------------------------------
        # 9.  Diagnostic plots
        # ------------------------------------------------------------------
        if show_diagnostic_plots:
            synthetic_image = self.create_synthetic_image(out_name, filter_name, warnings=True, counter='energy', out_name=f'synthetic_{filter_name}')

            _plot_alignment_diagnostics(
                synth_image=self.images[f'synthetic_{filter_name}'],
                synth_wcs=self.wcs[f'synthetic_{filter_name}'],
                synth_centroids=synth_centroids,
                matched_synth_pix=matched_synth_pix,
                cut_image=cut_image,
                cut_wcs=cut_wcs,
                ref_centroids=ref_centroids,
                matched_ref_pix=matched_ref_pix,
                original_cube=cube,
                original_wcs=synth_wcs,
                aligned_cube=aligned_cube,
                aligned_wcs=new_synth_wcs,
                filter_name=filter_name,
            )

        return aligned_cube
    
    def get_pa(self, wcs_name):

        if self.wcs[wcs_name].wcs.has_cd():
            M = self.wcs[wcs_name].wcs.cd
        else:
            pc = self.wcs[wcs_name].wcs.get_pc()
            cdelt = self.wcs[wcs_name].wcs.cdelt
            M = pc @ np.diag(cdelt)

        return np.degrees(np.arctan2(M[0,0], M[1,0]))
        
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

    def get_pix_scale(self, wcs_name):
        return self.wcs[wcs_name].wcs.cdelt[0]*3600*u.arcsec

    def get_spectrum(self, name, loc, radius, background_annulus_thickness=0,
        buffer=0*u.arcsec, replace_negatives = False, aperture_type='cyl', out_name=None):
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
        header = self.headers[name]
        wcs = self.wcs[name]
        cube = self.cubes[name]
        if aperture_type != 'cyl':
            print('Only circular apertures current coded in function')
            return None
        # === CONVERT RA/DEC TO PIXEL COORDINATES ===
        # Create SkyCoord object for spatial coordinates
        if (type(loc) == list) or (type(loc) == tuple):
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
        
        pix_area = self.get_pix_area(name)
        if 'CDELT1' in header:
            pixel_scale_deg = abs(header['CDELT1'])
        elif 'CD1_1' in header:
            pixel_scale_deg = abs(header['CD1_1'])
        else:
            print('Pixel size not found in header with key CDELT1 or CD1_1, aperture photometry failed')
            return

        source_radius_pixels = (radius.to_value(u.deg) / pixel_scale_deg)
        bg_inner_pixels = ((radius + buffer).to_value(u.deg) / pixel_scale_deg)
        bg_outer_pixels = ((radius + buffer + background_annulus_thickness).to_value(u.deg) / pixel_scale_deg)
        try:
            source_aperture = CircularAperture(
                (x, y),
                r=source_radius_pixels
            )
            source_area_pixels = source_aperture.area
            if background_annulus_thickness > 0:
                bg_annulus = CircularAnnulus(
                    (x, y),
                    r_in=bg_inner_pixels,
                    r_out=bg_outer_pixels
                )
                annulus_mask = bg_annulus.to_mask(method='exact')
                annulus_values = annulus_data[valid]
                annulus_weights = annulus_weights[valid]
        except:
            print('source aperture not valid')

        flux_density_spectrum = []
        nan_detected = 0
        for i in range(len(self.wavelengths[name])):
            image_quantity = cube[i]
            if replace_negatives is not False:
                if replace_negatives == 0:
                    image_quantity[image_quantity < 0] = 0
                else:
                    min_positive = np.nanmin(image_quantity[image_quantity > 0]) * 0.5
                    image_quantity[image_quantity < 0] = min_positive
                
            if ~np.isnan(image_quantity).sum() == 0:
                print(f'The entire wavelength slice for slice {i}:{self.wavelengths[name][i]} in the cube is NaNs')

            if background_annulus_thickness > 0:
                annulus_data = annulus_mask.multiply(image_quantity.value)
                annulus_weights = annulus_mask.data
                valid = (np.isfinite(annulus_data) & (annulus_weights > 0))
                intrinsic_pixel_values = (annulus_values / annulus_weights)
                background_per_pixel = np.nanmedian(
                    intrinsic_pixel_values
                ) * image_quantity.unit * pix_area

                annulus_area_pixels = np.sum(
                    annulus_weights
                )

                background_flux = (
                    background_per_pixel *
                    source_area_pixels
                )
            else:
                background_flux = 0
            source_flux = aperture_photometry(
                image_quantity,
                source_aperture,
                method='exact'
            )['aperture_sum'][0] * pix_area

            net_flux = source_flux - background_flux
            flux_density_spectrum.append(net_flux.value)
        units = net_flux.unit
        flux_density_spectrum = np.array(flux_density_spectrum)*units
        wavelength = self.wavelengths[name].to(u.m)
        frequency = (c / wavelength).to(u.Hz)

        F_lambda = (flux_density_spectrum * c / wavelength ** 2).to(u.W / (u.m ** 2 * u.m))

        spectrum = {
            'wavelength': wavelength,  # m
            'frequency': frequency,  # Hz
            'F_nu': flux_density_spectrum,  # W / m^2 / Hz
            'F_lambda': F_lambda,  # W / m^2 / m
            'location': loc,
            'radius': radius,
            'bg_annulus': background_annulus_thickness,
            'buffer': buffer,
            'aperture_type': aperture_type
        }
        if out_name is None:
            if f'{name}' in self.spectra.keys():
                out_name = f'{name}_{len(self.spectra.keys())}'
            else:
                out_name = f'{name}'
        self.spectra[out_name]=spectrum
        print(f'spectra saved to {self}.spectra[{out_name}]')
        return spectrum

    def apply_filter(self, spec_name, filter_name, warnings = True, counter = 'photons', out_name=None):
        '''get expected flux through filter. Assumes Fnu array is in W/m2/Hz and wl array is in meters. Otherwise, units will be weird.
        -------------
        
        Parameters
        -------------
        
        Returns
        -------------
        total_flux : type = float - Ideally in units of W/m2
        '''
        Fnu_array = self.spectra[spec_name]['F_nu']
        Fnu_array = np.nan_to_num(Fnu_array, nan=0.0)

        wl_array = self.spectra[spec_name]['wavelength']
        filter_wl_array, transmission_array = get_filter_data(filter_name)

        try:
            Fnu_units = Fnu_array.unit
        except:
            print('first argument of apply_filter() must have units')
            return None
        if ((filter_wl_array[0] < wl_array[0]) or (filter_wl_array[-1] > wl_array[-1])): #TJ Check if wavelengths are compatible with filter
            if warnings:
                print(f'filter goes from {filter_wl_array[0]} to {filter_wl_array[-1]}, but provided Fnu array goes from {wl_array[0]} to {wl_array[-1]}')
            idx_start = np.searchsorted(filter_wl_array, wl_array[0], side='left')
            idx_end = np.searchsorted(filter_wl_array, wl_array[-1], side='right')
            
            # Expand by one index if possible
            idx_start = max(0, idx_start - 1)  # Include one lower index
            idx_end = min(len(filter_wl_array), idx_end + 1)  # Include one higher index
            
            # Slice transmission data
            filter_wl_array = filter_wl_array[idx_start:idx_end]
            transmission_array = transmission_array[idx_start:idx_end]
        if len(filter_wl_array) == 0:
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
        filter_wl_array = np.array(filter_wl_array)

        #TJ Convert wavelength to frequency, reverse so freq increases left to right
        spec_freq_array = c / wl_array[::-1]
        Fnu_array = Fnu_array[::-1]
        filter_freq_array = c / filter_wl_array[::-1]
        transmission_array = transmission_array[::-1]

        #TJ Interpolate Fnu onto the transmission frequency grid
        #TJ this is because jwst transmission arrays are averages over BW widths which are much coarser than Fnu is.
        trans_interp = np.interp(
            spec_freq_array,
            filter_freq_array,
            transmission_array,
            left=0.0,
            right=0.0
        )

        if counter.lower() in ("photon", "photons", 'phot'):
            weight = trans_interp / spec_freq_array
        elif counter.lower() in ("energy", 'e'):
            weight = trans_interp
        else:
            raise ValueError("counter must be 'energy' or 'photon'")

        numerator = np.trapezoid(
            Fnu_array * weight,
            x=spec_freq_array,
            axis=0
        )

        denominator = np.trapezoid(
            weight,
            x=spec_freq_array
        )

        ab_mean_flux = numerator / denominator
        # Numerator: Fν * Transmission / nu integrated over frequency
        if out_name is None:
            if filter_name not in self.data:
                out_name = filter_name
            else:
                count = 1
                while out_name in self.data:
                    out_name = f'{filter_name}_{count}'
                    count+=1

        self.data[out_name] = ab_mean_flux*Fnu_units
        print(f'data saved to self.data[{out_name}]')
        return ab_mean_flux*Fnu_units

    def create_synthetic_image(self, cube_name, filter_name, warnings=True, counter='photon', out_name=None, out_file=None):
        """
        Apply a filter transmission curve to every spaxel in a cube, producing
        a synthetic image of the filter-weighted mean flux.

        Parameters
        ----------
        cube_name   : str  — key into self.cubes
        filter_name : str  — filter name passed to get_filter_data()
        warnings    : bool — warn if filter extends beyond cube wavelength range
        counter     : str  — 'energy' or 'photons' weighting

        Returns
        -------
        synth_image : 2D astropy Quantity array, shape (ny, nx), same flux units as cube
        """
        if filter_name is None:
            cube = self.cubes[cube_name]
            cube_units = cube.unit

            # (nz, ny, nx)
            cube_data = cube.unmasked_data[:].to(cube_units).value
            cube_data= np.nan_to_num(cube_data, nan=0.0)
            # ascending frequency axis
            freq = (c/self.wavelengths[name]).to(u.Hz)
            cube_data = cube_data[::-1]

            # ∫Fν dν
            numerator = np.trapezoid(
                cube_data,
                x=freq,
                axis=0
            )

            # ∫dν
            denominator = freq[-1] - freq[0]

            synth_image = (numerator / denominator) * cube_units

            if out_name is None:
                out_name = f"{cube_name}"

                count = 1
                while out_name in self.images:
                    out_name = f"{cube_name}_{count}"
                    count += 1

            self.images[out_name] = synth_image
            self.wcs[out_name] = self.wcs[cube_name].celestial

            print(f"Saved white-light image to self.images['{out_name}']")

        else:
            cube = self.cubes[cube_name]
            cube_units = cube.unit

            # --- get cube as plain (nz, ny, nx) numpy array ---
            cube_data = cube.unmasked_data[:].to(cube_units).value
            cube_data = np.nan_to_num(cube_data, nan=0.0)
            wl_array  = np.array(self.wavelengths[cube_name].to(u.m).value)      # (nz,)

            # --- load and trim transmission curve to cube wavelength range ---
            filter_wl_array, transmission_array = get_filter_data(filter_name)
            filter_wl_array     = np.array(filter_wl_array,     dtype=float)
            transmission_array = np.array(transmission_array, dtype=float)

            if filter_wl_array[0] < wl_array[0] or filter_wl_array[-1] > wl_array[-1]:
                if warnings:
                    print(f'filter goes from {filter_wl_array[0]:.4e} to {filter_wl_array[-1]:.4e} m, '
                        f'but cube goes from {wl_array[0]:.4e} to {wl_array[-1]:.4e} m')
                idx_start = max(0, np.searchsorted(filter_wl_array, wl_array[0],  side='left') - 1)
                idx_end   = min(len(filter_wl_array), np.searchsorted(filter_wl_array, wl_array[-1], side='right') + 1)
                filter_wl_array = filter_wl_array[idx_start:idx_end]
                transmission_array = transmission_array[idx_start:idx_end]

            if len(filter_wl_array) == 0:
                raise ValueError(f"No overlap between cube wavelengths and filter '{filter_name}'")

            spec_freq_array = (c.value / wl_array)[::-1]        # (nz,)
            cube_data = cube_data[::-1, :, :]                   # (nz, ny, nx)

            filter_freq_array = (c.value / filter_wl_array)[::-1]
            transmission_array = transmission_array[::-1]

            # ------------------------------------------------------------------
            # Interpolate FILTER onto the cube sampling
            # ------------------------------------------------------------------

            trans_interp = np.interp(
                spec_freq_array,
                filter_freq_array,
                transmission_array,
                left=0.0,
                right=0.0
            )

            # ------------------------------------------------------------------
            # Construct weighting function
            # ------------------------------------------------------------------

            if counter.lower() in ("photon", "photons"):
                weight = trans_interp / spec_freq_array
            elif counter.lower() == "energy":
                weight = trans_interp
            else:
                raise ValueError("counter must be 'energy' or 'photon'")

            # reshape for broadcasting over image dimensions
            weight3d = weight[:, None, None]

            # ------------------------------------------------------------------
            # Perform filter-weighted integration
            # ------------------------------------------------------------------

            numerator = np.trapezoid(
                cube_data * weight3d,
                x=spec_freq_array,
                axis=0
            )

            denominator = np.trapezoid(
                weight,
                x=spec_freq_array
            )

            synth_image = (numerator / denominator) * cube_units
            if out_name is None:
                if f'synth_{filter_name}' not in self.data:
                    out_name = f'synth_{filter_name}'
                else:
                    count = 1
                    while out_name in self.data:
                        out_name = f'synth_{filter_name}_{count}'
                        count+=1
            self.images[out_name] = synth_image
            self.wcs[out_name] = self.wcs[cube_name].celestial
            print(f'images and wcs saved to self.images/wcs[{out_name}]')
            header = self.cubes[cube_name].wcs.celestial.to_header()

            header["BUNIT"] = str(cube_units)
            header["FILTER"] = filter_name
            header["FROMCUBE"] = cube_name
            self.headers[out_name] = header
            if out_file is not None:

                hdu = fits.PrimaryHDU(
                    data=self.images[out_name].value,
                    header=header
                )

                hdu.writeto(out_file, overwrite=True)


                print(f'Saved synthetic image to : {out_file}')
        return synth_image

    def create_new_continuum_image(
        self,
        cube_name,
        wavelength,
        search_width=0.5*u.micron,
        window_width=0.1*u.micron,
        sigma_clip=3,
        fit_order=1,
        out_name=None,
        out_file=None
    ):
        """
        Estimate a continuum image at a given wavelength using robust
        continuum windows.

        The algorithm is fully vectorized across all spatial pixels.

        Parameters
        ----------
        cube_name : str

        wavelength : Quantity

        search_width : Quantity

        window_width : Quantity

        sigma_clip : float

        fit_order : int
            Currently supports 0 or 1.

        out_name : str or None

        Returns
        -------
        continuum_image : Quantity
        """

        cube = self.cubes[cube_name]
        cube_unit = cube.unit

        data = cube.unmasked_data[:].to_value(cube_unit)
        data = np.nan_to_num(data)

        wl = self.wavelengths[cube_name].to(u.micron).value
        target = wavelength.to(u.micron).value

        nz, ny, nx = data.shape
        npix = ny * nx

        cube2d = data.reshape(nz, npix)

        ###############################################################
        # Limit search region to cube coverage
        ###############################################################

        search = search_width.to(u.micron).value
        dw = window_width.to(u.micron).value

        left_edge = max(wl.min(), target-search)
        right_edge = min(wl.max(), target+search)

        centers = np.arange(left_edge, right_edge+dw, dw)

        ###############################################################
        # Build window masks once
        ###############################################################

        masks = []

        valid_centers = []

        for center in centers:

            m = np.abs(wl-center) < dw/2

            if np.sum(m) >= 5:

                masks.append(m)

                valid_centers.append(center)

        centers = np.asarray(valid_centers)

        nwin = len(centers)

        if nwin < fit_order+2:
            raise RuntimeError("Not enough continuum windows.")

        ###############################################################
        # Compute robust statistics
        ###############################################################

        medians = np.empty((nwin, npix))
        mads = np.empty((nwin, npix))

        for i, mask in enumerate(masks):

            vals = cube2d[mask]

            med = np.median(vals, axis=0)

            mad = 1.4826*np.median(
                np.abs(vals-med),
                axis=0
            )

            medians[i] = med
            mads[i] = mad

        ###############################################################
        # Reject windows containing emission/absorption features
        ###############################################################

        median_mad = np.median(mads, axis=0)

        good = mads < sigma_clip*median_mad

        ###############################################################
        # Constant continuum
        ###############################################################

        if fit_order == 0:

            continuum = np.nanmedian(
                np.where(good, medians, np.nan),
                axis=0
            )

        ###############################################################
        # Linear continuum
        ###############################################################

        elif fit_order == 1:

            x = centers[:, None]

            w = good.astype(float)

            S = np.sum(w, axis=0)

            Sx = np.sum(w*x, axis=0)

            Sy = np.sum(w*medians, axis=0)

            Sxx = np.sum(w*x*x, axis=0)

            Sxy = np.sum(w*x*medians, axis=0)

            denom = S*Sxx - Sx*Sx

            continuum = np.full(npix, np.nan)

            valid = (S >= 2) & (np.abs(denom) > 0)

            slope = np.zeros(npix)

            intercept = np.zeros(npix)

            slope[valid] = (
                S[valid]*Sxy[valid]
                - Sx[valid]*Sy[valid]
            ) / denom[valid]

            intercept[valid] = (
                Sy[valid]
                - slope[valid]*Sx[valid]
            ) / S[valid]

            continuum[valid] = (
                intercept[valid]
                + slope[valid]*target
            )

        else:

            raise NotImplementedError(
                "Only fit_order=0 or 1 currently supported."
            )

        ###############################################################
        # Save
        ###############################################################

        continuum_image = continuum.reshape(ny, nx)*cube_unit

        if out_name is not None:

            self.images[out_name] = continuum_image

            self.wcs[out_name] = self.wcs[cube_name].celestial

            header = self.cubes[cube_name].wcs.celestial.to_header()

            header["BUNIT"] = str(cube_unit)
            header["IM_TYPE"] = 'Continuum Image'
            header["FROMCUBE"] = cube_name
            self.headers[out_name] = header
            if out_file is not None:

                hdu = fits.PrimaryHDU(
                    data=self.images[out_name].value,
                    header=header
                )

                hdu.writeto(out_file, overwrite=True)

        return continuum_image

    def create_continuum_image(
        self,
        cube_name,
        wavelength,
        search_width=0.4*u.micron,
        median_width=0.05*u.micron,
        sigma_clip=3.0,
        fit_order=1,
        out_name=None,
        out_file=None,
    ):
        """
        Estimate a continuum image by masking emission/absorption lines and
        fitting the remaining local continuum.

        Parameters
        ----------
        cube_name : str

        wavelength : Quantity

        search_width : Quantity
            Width of the fitting region centered on the desired wavelength.

        median_width : Quantity
            Width of the median filter used to estimate the continuum.

        sigma_clip : float
            Threshold for masking spectral features.

        fit_order : int
            0 = constant
            1 = linear

        Returns
        -------
        continuum_image : Quantity
        """

        cube = self.cubes[cube_name]
        cube_unit = cube.unit

        data = cube.unmasked_data[:].to_value(cube_unit)
        wl = self.wavelengths[cube_name].to(u.micron).value

        target = wavelength.to(u.micron).value

        nz, ny, nx = data.shape
        npix = ny * nx

        cube2d = data.reshape(nz, npix)

        ############################################################
        # Median smoothing
        ############################################################

        dw = np.median(np.diff(wl))

        width_pix = int(
            np.round(
                median_width.to_value(u.micron) / dw
            )
        )

        width_pix = max(width_pix, 5)

        if width_pix % 2 == 0:
            width_pix += 1

        smooth = median_filter(
            cube2d,
            size=(width_pix, 1),
            mode="nearest"
        )

        ############################################################
        # Robust sigma
        ############################################################

        residual = cube2d - smooth

        mad = 1.4826 * np.nanmedian(
            np.abs(residual),
            axis=0
        )

        mad[mad == 0] = np.nanmedian(mad[mad > 0])

        ############################################################
        # Feature mask
        ############################################################

        mask = np.abs(residual) < sigma_clip * mad

        ############################################################
        # Restrict to local fitting region
        ############################################################

        left = max(wl.min(), target-search_width.to_value(u.micron))
        right = min(wl.max(), target+search_width.to_value(u.micron))

        local = (wl >= left) & (wl <= right)

        mask &= local[:, None]

        ############################################################
        # Polynomial fit
        ############################################################

        continuum = np.full(npix, np.nan)

        x = wl

        if fit_order == 0:

            continuum = np.nanmedian(
                np.where(mask, cube2d, np.nan),
                axis=0
            )

        elif fit_order == 1:

            for i in range(npix):

                good = mask[:, i]

                if np.sum(good) < 4:
                    continue

                coeff = np.polyfit(
                    x[good],
                    cube2d[good, i],
                    1
                )

                continuum[i] = np.polyval(
                    coeff,
                    target
                )

        else:

            raise ValueError("Only fit_order=0 or 1 supported.")

        ############################################################
        # Save
        ############################################################

        continuum_image = continuum.reshape(ny, nx) * cube_unit

        if out_name is not None:

            self.images[out_name] = continuum_image
            self.wcs[out_name] = self.wcs[cube_name].celestial

            header = self.cubes[cube_name].wcs.celestial.to_header()

            header["BUNIT"] = str(cube_unit)
            header["IM_TYPE"] = "Continuum Image"
            header["FROMCUBE"] = cube_name
            header["CONTWL"] = target
            header["FITORD"] = fit_order
            header["SIGCLIP"] = sigma_clip

            self.headers[out_name] = header

            if out_file is not None:

                fits.PrimaryHDU(
                    continuum_image.value,
                    header
                ).writeto(
                    out_file,
                    overwrite=True
                )

        return continuum_image

    def stitch_spectra(self, names, anchor_idx=0, method='add_shift', out_name=None):
        """
        Stitch spectra listed in `names`, keeping the anchor fixed.

        Parameters
        ----------
        names      : list of str — keys into self.spectra, in wavelength order
        anchor_idx : int         — index in names of the spectrum to keep fixed
        method     : str         — 'add_shift', 'mult_shift', or 'mean'
        out_name   : str         — key to save result under in self.spectra
        """
        if method not in ('add_shift', 'mult_shift', 'mean'):
            raise ValueError(f"method must be 'add_shift', 'mult_shift', or 'mean', got '{method}'")
        names = sorted(
            names,
            key=lambda n: np.nanmin(
                self.spectra[n]['wavelength'].value
            )
        )
        # --- Check metadata consistency, carry anchor values to output ---
        carried_keys = ['location', 'radius', 'bg_annulus', 'buffer', 'aperture_type']
        anchor_spec  = self.spectra[names[anchor_idx]]
        carried_meta = {}
        for key in carried_keys:
            anchor_val  = anchor_spec.get(key)
            mismatched  = [n for n in names if self.spectra[n].get(key) != anchor_val]
            if mismatched:
                print(f"Warning: '{key}' differs in {mismatched}, using anchor value '{anchor_val}'")
            carried_meta[key] = anchor_val

        # --- Strip units for internal arithmetic ---
        base_units = anchor_spec['F_nu'].unit
        wl_units   = anchor_spec['wavelength'].unit
        base = {
            'wavelength': np.asarray(anchor_spec['wavelength'].value, dtype=float),
            'F_nu':       np.asarray(anchor_spec['F_nu'].value,       dtype=float),
        }

        for i in range(anchor_idx - 1, -1, -1):
            print(f"Stitching LEFT  ({method}): {names[i]} → anchor")
            cur = self.spectra[names[i]]
            cur_d = {'wavelength': np.asarray(cur['wavelength'].value, dtype=float),
                    'F_nu':       np.asarray(cur['F_nu'].value,       dtype=float)}
            base = _combine_spectra(base, cur_d, side='left', method=method)

        for i in range(anchor_idx + 1, len(names)):
            print(f"Stitching RIGHT ({method}): anchor → {names[i]}")
            cur = self.spectra[names[i]]
            cur_d = {'wavelength': np.asarray(cur['wavelength'].value, dtype=float),
                    'F_nu':       np.asarray(cur['F_nu'].value,       dtype=float)}
            base = _combine_spectra(base, cur_d, side='right', method=method)

        # --- Rebuild derived arrays and reattach units ---
        wl   = base['wavelength'] * wl_units
        fnu  = base['F_nu']       * base_units
        freq = (c / wl).to(u.Hz)
        flam = (fnu * c / wl**2).to(u.W / (u.m**2 * u.m))

        print(f'Combined spectrum ({method}): {wl[0]:.4e} -- {wl[-1]:.4e}')

        if out_name is None:
            out_name = f'{names[0]}-{names[-1]}'
            if out_name in self.spectra:
                count = 1
                out_name_base = out_name.copy()
                while out_name in self.spectra:
                    out_name = f'{out_name_base}_{count}'
                    count+=1

        self.spectra[out_name] = {'wavelength': wl, 'frequency': freq,
                                'F_nu': fnu, 'F_lambda': flam, **carried_meta}
        print(f'Combined spectrum saved to self.spectra[{out_name}]')
        return self.spectra[out_name]

    def display(self, names, loc, radius, ncols=3, cmap='viridis', zoom=5, show_grid=False):
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

        fig = plt.figure(figsize=(5*ncols, 5*nrows))

        for i, name in enumerate(names):
            image = self.images[name].value
            wcs = self.wcs[name]

            try:
                pixel_scale = np.abs(wcs.wcs.cd[0][0]) * 3600
            except:
                pixel_scale = np.abs(wcs.wcs.cdelt[0]) * 3600

            cutout = Cutout2D(image, position=loc_sky, size=(radius*zoom, radius*zoom), wcs=wcs)

            # Each subplot gets its own WCS projection
            ax = fig.add_subplot(nrows, ncols, i+1, projection=cutout.wcs)

            x_img, y_img = cutout.wcs.world_to_pixel(loc_sky)
            pixels = cutout.data[np.isfinite(cutout.data)]
            med, sig = np.nanmedian(pixels), np.nanstd(pixels)
            norm = colors.Normalize(vmin=0, vmax=np.percentile(cutout.data, 99))
            im = ax.imshow(cutout.data, origin='lower', cmap=cmap,
                    norm=norm)

            if show_grid:
                ax.coords.grid(color='white', linestyle='--', linewidth=1, alpha=0.7)

            ax.add_patch(Circle((x_img, y_img),
                                (radius.to(u.arcsec).value) / pixel_scale,
                                ec='red', fc='none', lw=2, alpha=0.7))

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Flux (native units)", fontsize=10)
            ax.set_title(name, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        for i in range(n_images, nrows*ncols):
            fig.add_subplot(nrows, ncols, i+1).axis('off')

        plt.tight_layout()
        plt.show()
    
    def append_cubes(self, cube_names, wl_range=None, out_name=None):
        """
        Reproject and spectrally concatenate two cubes onto a common pixel grid.
        The first cube's WCS defines the output spatial grid; the second cube is
        reprojected to match it before concatenation.

        Parameters
        ----------
        cube_names : list of two str
            Keys into self.cubes. The first cube defines the output spatial WCS.
        wl_range   : list/tuple of two astropy Quantities, optional
            [wl_min, wl_max] — only keep slices within this wavelength range.
            e.g. wl_range=[2.0*u.micron, 5.0*u.micron]

        Returns
        -------
        combined_cube : SpectralCube
            Stitched cube on the first cube's spatial grid, optionally trimmed.
            Also stored under self.cubes[f'{cube_names[0]}+{cube_names[1]}'].
        """

        if len(cube_names) != 2:
            raise ValueError("append_cubes expects exactly two cube names")

        name_a, name_b = cube_names
        if self.wavelengths[name_a][0] < self.wavelengths[name_b][0]:
            cube_a = self.cubes[name_a]
            cube_b = self.cubes[name_b]
            wl_a = self.wavelengths[name_a]
            wl_b = self.wavelengths[name_b]
        else:
            cube_a = self.cubes[name_b]
            cube_b = self.cubes[name_a]
            wl_a = self.wavelengths[name_b]
            wl_b = self.wavelengths[name_a]
            name_a, name_b = name_b, name_a

        # --- check units are compatible ---
        if cube_a.unit != cube_b.unit:
            print(f"Warning: unit mismatch ({cube_a.unit} vs {cube_b.unit}), "
                f"converting {name_b} to {cube_a.unit}")
            cube_b = cube_b.to(cube_a.unit)

        # --- optionally trim each cube to wl_range before reprojection ---
        # (reduces memory and reprojection time significantly for large cubes)
        if wl_range is not None:
            print('wavelength cropping function not available, just use full cubes for now')
            wl_min, wl_max = wl_range[0].to(u.m), wl_range[1].to(u.m)

            mask_a = (wl_a >= wl_min) & (wl_a <= wl_max)
            mask_b = (wl_b >= wl_min) & (wl_b <= wl_max)
            cube_a = cube_a[(wl_a >= wl_min) & (wl_a <= wl_max)]
            cube_b = cube_b[mask_b]

        # --- reproject cube_b onto cube_a's spatial WCS slice by slice ---
        # Use cube_a's 2D celestial WCS as the target
        target_wcs    = cube_a.wcs.celestial
        target_shape  = cube_a.shape[1:]          # (ny, nx)
        nz_b          = cube_b.shape[0]
        cube_b_data   = np.array(cube_b.unmasked_data[:].value)   # (nz_b, ny_b, nx_b)
        reproj_b      = np.empty((nz_b, *target_shape), dtype=float)
        print(f"Reprojecting {name_b} ({nz_b} slices) onto {name_a} spatial grid...")
        for i in range(nz_b):
            reproj_b[i], _ = reproject_interp(
                (cube_b_data[i], cube_b.wcs.celestial),
                target_wcs,
                shape_out=target_shape
            )

        # --- interpolate cube_b onto cube_a's spectral grid and average in overlap ---
        cube_a_data = np.array(cube_a.unmasked_data[:].value)   # (nz_a, ny, nx)
        wl_a_vals   = wl_a.to_value(u.m)        # (nz_a,)
        wl_b_vals   = wl_b.to_value(u.m)        # (nz_b,)

        overlap_min = max(wl_a_vals[0],  wl_b_vals[0])
        overlap_max = min(wl_a_vals[-1], wl_b_vals[-1])
        has_overlap = overlap_max > overlap_min

        if has_overlap:
            # Interpolate cube_b onto cube_a's wavelength grid inside the overlap
            # reproj_b is (nz_b, ny, nx); we need (nz_a_overlap, ny, nx)
            ov_mask_a = (wl_a_vals >= overlap_min) & (wl_a_vals <= overlap_max)
            wl_a_ov   = wl_a_vals[ov_mask_a]                    # (nz_ov,)

            ny, nx    = target_shape
            nz_ov     = ov_mask_a.sum()

            # Vectorised interpolation: reshape spatial dims to one axis
            reproj_b_flat = reproj_b.reshape(nz_b, -1)          # (nz_b, ny*nx)
            interp_b_flat = np.empty((nz_ov, ny * nx), dtype=float)

            # searchsorted gives bracketing indices along the b spectral axis
            idx = np.searchsorted(wl_b_vals, wl_a_ov)
            idx = np.clip(idx, 1, nz_b - 1)
            t   = ((wl_a_ov - wl_b_vals[idx - 1]) /
                   (wl_b_vals[idx] - wl_b_vals[idx - 1]))       # (nz_ov,)

            for j in range(nz_ov):
                interp_b_flat[j] = (reproj_b_flat[idx[j] - 1] * (1 - t[j])
                                  + reproj_b_flat[idx[j]]     *      t[j])

            interp_b_ov = interp_b_flat.reshape(nz_ov, ny, nx)  # (nz_ov, ny, nx)

            # Average cube_a and interpolated cube_b in the overlap
            cube_a_ov              = cube_a_data[ov_mask_a]      # (nz_ov, ny, nx)
            averaged_ov            = np.where(
                np.isfinite(interp_b_ov) & np.isfinite(cube_a_ov),
                0.5 * (cube_a_ov + interp_b_ov),
                np.where(np.isfinite(cube_a_ov), cube_a_ov, interp_b_ov)
            )

            # Build final arrays: left tail (a only) + overlap (averaged) + right tail (b only)
            left_mask  = wl_a_vals < overlap_min
            right_mask = wl_b_vals > overlap_max

            all_wl   = np.concatenate([wl_a_vals[left_mask],
                                        wl_a_ov,
                                        wl_b_vals[right_mask]])
            all_data = np.concatenate([cube_a_data[left_mask],
                                        averaged_ov,
                                        reproj_b[right_mask_b := (wl_b_vals > overlap_max)]],
                                       axis=0)
        else:
            # No overlap — just sort and concatenate
            print("Warning: no spectral overlap between cubes, concatenating without averaging")
            all_wl   = np.concatenate([wl_a_vals, wl_b_vals])
            all_data = np.concatenate([cube_a_data, reproj_b], axis=0)
            order    = np.argsort(all_wl)
            all_wl   = all_wl[order]
            all_data = all_data[order]
        if not np.all(np.diff(all_wl) > 0):
            print("Warning: combined spectral axis is not strictly monotonic — "
                "check for overlapping wavelength ranges between cubes")

        # --- rebuild a SpectralCube with the correct WCS ---
        # Construct a new WCS by copying cube_a's and updating the spectral axis
        new_header = cube_a.wcs.to_header()
        new_header['NAXIS']  = 3
        new_header['NAXIS1'] = target_shape[1]
        new_header['NAXIS2'] = target_shape[0]
        new_header['NAXIS3'] = len(all_wl)

        # Update spectral WCS to a linear approximation of the new axis
        # (adequate for display/extraction; use a lookup table cube if you need exact values)
        new_header['CRPIX3'] = 1
        new_header['CRVAL3'] = all_wl[0]
        new_header['CDELT3'] = np.median(np.diff(all_wl))
        new_header['CTYPE3'] = 'WAVE'
        new_header['CUNIT3'] = 'm'

        new_wcs = WCS(new_header)
        combined_cube = SpectralCube(
            data  = all_data * cube_a.unit,
            wcs   = new_wcs,
            meta  = {'name': f'{name_a}+{name_b}'}
        )
        if out_name is None:
            if cube_names[0]+cube_names[1] not in self.cubes:
                out_name = cube_names[0]+cube_names[1]
            else:
                count = 1
                while out_name in self.cubes:
                    out_name = cube_names[0]+cube_names[1]+f'_{count}'
                    count+=1
        self.cubes[out_name]   = combined_cube
        self.headers[out_name] = new_header
        self.wcs[out_name]     = new_wcs
        self.wavelengths[out_name] = all_wl*u.m
        print(f"Stored combined cube under '{out_name}': \nNew cube goes from{self.wavelengths[out_name][0]} to {self.wavelengths[out_name][-1]}")
        return combined_cube

    def which_cubes(self, filter_name):
        """
        Return channel keys whose wavelength coverage overlaps the filter's
        effective halfwidth window [filter_mean - ew/2, filter_mean + ew/2].

        Parameters
        ----------

        Returns
        -------
        list of str — keys from self.spectra whose coverage overlaps the window
        """
        filter_wl, filter_trans = get_filter_data(filter_name)
        filt_lo = filter_wl[0]
        filt_hi = filter_wl[-1]

        covering = []

        for key in list(self.cubes.keys()):
            wl = self.wavelengths[key]
            if wl[0] <= filt_hi and wl[-1] >= filt_lo:
                covering.append(key)

        return covering

    def which_spectra(self, filter_name):
        """
        Return channel keys whose wavelength coverage overlaps the filter's
        effective halfwidth window [filter_mean - ew/2, filter_mean + ew/2].

        Parameters
        ----------

        Returns
        -------
        list of str — keys from self.spectra whose coverage overlaps the window
        """
        filter_wl, filter_trans = get_filter_data(filter_name)
        filt_lo = filter_wl[0]
        filt_hi = filter_wl[-1]

        covering = []

        for key in list(self.spectra.keys()):
            wl = self.spectra[key]['wavelength']
            if wl[0] <= filt_hi and wl[-1] >= filt_lo:
                covering.append(key)

        return covering

    def which_filters(self, obj, filter_list, must_all=False):
        keep = []
        for fil in filter_list:
            wl, trans = get_filter_data(fil)
            if not must_all:
                if type(obj) == SpectralCube:                    
                    if (obj.spectral_axis[-1] > wl[0]) and (obj.spectral_axis[0] < wl[-1]):
                        keep.append(fil)

                elif type(obj) == dict:
                    if (obj['wavelength'][-1] > wl[0]) and (obj['wavelength'][0] < wl[-1]):
                        keep.append(fil)
                elif type(obj.value) == np.ndarray:
                    if (obj[-1] > wl[0]) and (obj[0] < wl[-1]):
                        keep.append(fil)
                else:
                    print(f'type {type(obj)} not recognized as SpectralCube, dict, or np.ndarray')
            else:
                if type(obj) == SpectralCube:                    
                    if (obj.spectral_axis[0] > wl[0]) and (obj.spectral_axis[-1] < wl[-1]):
                        keep.append(fil)

                elif type(obj) == dict:
                    if (obj['wavelength'][0] > wl[0]) and (obj['wavelength'][-1] < wl[-1]):
                        keep.append(fil)
                elif type(obj.value) == np.ndarray:
                    if (obj[0] > wl[0]) and (obj[-1] < wl[-1]):
                        keep.append(fil)
                else:
                    print(f'type {type(obj)} not recognized as SpectralCube, dict, or np.ndarray')
        return keep
    
    def adjust_spectrum(self, name, filter_name, image_obj, image_name, location, radius, adjustment_operation = 'add', out_name=None):
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
            print(f'No filters found for {name}, returning unmodified spectrum')
            if out_name is None:
                if f'adjusted_{name}' not in self.spectra:
                    out_name = f'adjusted_{name}'
                else:
                    count = 1
                    while out_name in self.spectra:
                        out_name = f'adjusted_{name}_{count}'
                        count+=1
            raw_data = self.spectra[name]
            raw_data['correction'] = 0 if adjustment_operation == 'add' else 1
            raw_data['adjustment_method'] = adjustment_operation
            raw_data['F_lambda'] = None
            self.spectra[out_name] = raw_data
        else:

            raw_data = self.spectra[name]
            new_data = raw_data.copy()
            radius = raw_data['radius']
            bg_radius = raw_data['bg_annulus']
            buffer = raw_data['buffer']
            location = raw_data['location']
            filter_wl, filter_trans = get_filter_data(filter_name) #TJ this is the transmission vs wavelength function for this filter
            image_flux = image_obj.get_background_subtracted_flux(image_name, location, radius, bg_radius, buffer)['source_flux'] #TJ this is the flux we SHOULD get
            initial_synth_flux = self.apply_filter(name, filter_name, warnings = True, counter = 'energy', out_name=f'initial_synth_{filter_name}') #TJ this is the current synthetic flux we get
            
            if adjustment_operation == 'add':
                correction = image_flux - initial_synth_flux
                new_data['F_nu'] = new_data['F_nu'] + correction
            elif adjustment_operation == 'multiply':
                correction = image_flux/initial_synth_flux
                new_data['F_nu'] = new_data['F_nu']*correction
            else:
                print('adjustment operation not recognized, only "add" or "multiply" are currently implemented')
                return None
            if out_name is None:
                if f'adjusted_{name}' not in self.spectra:
                    out_name = f'adjusted_{name}'
                else:
                    count = 1
                    while out_name in self.spectra:
                        out_name = f'adjusted_{name}_{count}'
                        count+=1
            new_data['correction'] = correction
            new_data['adjustment_method'] = adjustment_operation
            new_data['F_lambda'] = None
            self.spectra[out_name] = new_data

    def get_largest_filter(self, spectrum_name, filter_list):
        '''Takes an ifu file and selects the filter with the largest bandpass that is entirely within it.
        -------------
        
        Parameters
        -------------
        ifu_file : type = string - string to location of ifu file
        filter_files : type = list - list of strings to filter files
        Returns
        -------------
        Filter name (ex. "F115W") corresponding to the largest filter entirely contained within the IFU file 
        '''
        filters = []
        widths = []
        for fil in filter_list:
            filter_wl, _, width, _, _ = get_filter_data(fil, aux_info=True)
            cube_range = [self.spectra[spectrum_name]['wavelength'][0], self.spectra[spectrum_name]['wavelength'][-1]]
            if (cube_range[0] < filter_wl[0]) and (cube_range[-1] > filter_wl[-1]):
                filters.append(fil)
                widths.append(width.value)
        if len(filters)<1:
            print(f'No filters entirely within {spectrum_name}')
            return None
        else:
            best_filter = filters[np.argmax(widths)]
            return best_filter

    def save_science(self, filename):
        """
        Save entire SpecScience object to disk.

        Parameters
        ----------
        filename : str
            Output filename, e.g. 'science.pkl'
        """

        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved SpecScience object to {filename}")
    
    def save_cube(self, cube_name, filename, overwrite=True):
        """
        Save a cube to a FITS file.

        Parameters
        ----------
        cube_name : str
            Name of cube in self.cubes.

        filename : str
            Output FITS filename.

        overwrite : bool
            Overwrite existing file.
        """

        cube = self.cubes[cube_name]

        # Convert Quantity -> ndarray
        data = cube.unmasked_data[:].value.astype(np.float32)

        # Build header from cube WCS
        header = cube.wcs.to_header()

        # Preserve units
        header["BUNIT"] = str(cube.unit)

        # Store wavelength units for convenience
        if cube_name in self.wavelengths:
            header["WAVEUNIT"] = str(self.wavelengths[cube_name].unit)

        hdu = fits.PrimaryHDU(
            data=data,
            header=header
        )

        hdu.writeto(filename, overwrite=overwrite)

        print(f"Saved cube '{cube_name}' to {filename}")

    def save_spectra(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.spectra, f)

    def load_all_spectra(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.spectra = pickle.load(f)

    def import_spec_data(self, name, filepath):
        '''Import already extracted spectra from a txt file and save as self.spectra[name]'''
        wl, fnu = np.genfromtxt(filepath)
        wl = wl*1e-6*u.m
        fnu
        frequency = (c / wavelength).to(u.Hz)
        
        F_lambda = (flux_density_spectrum * c / wavelength ** 2).to(u.W / (u.m ** 2 * u.m))
    @classmethod
    def load_science(cls, filename):
        """
        Load a saved SpecScience object.
        """

        with open(filename, "rb") as f:
            obj = pickle.load(f)

        print(f"Loaded SpecScience object from {filename}")
        return obj
