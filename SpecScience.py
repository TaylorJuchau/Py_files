import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.wcs import WCS
from spectral_cube import SpectralCube
from Functions import *
class SpecScience:
    'Tools for IFU cube analysis'
    def __init__(self):
        #TJ initialize dictionary of class attributes
        self.cubes = {}
        self.headers = {}
        self.files = {}
        self.wcs = {}

        
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


        self.cubes[name] = cube
        self.headers[name] = header
        self.wcs[name] = wcs

        print(
            f'Loaded: {name} '
            f'[cube units: {cube.unit}]'
        )
