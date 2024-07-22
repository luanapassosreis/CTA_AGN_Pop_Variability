from astro_constants import *

import numpy as np

## main imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## astro imports
import astropy.table
from astropy import units as u
from astropy.io import fits
from astropy.io import ascii
from astropy.table import QTable, Table

from astropy.time import Time,TimeUnix
from datetime import datetime

## other imports
import os
import csv
import glob
import math
import json
import statistics

import scipy.optimize as sp
import scipy.odr.odrpack as odrpack
from scipy import signal, integrate
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr

import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter


class JSONAnalyzer:
    
    def __init__(self, file_name, binning=['3-days','weekly','monthly'], index=['fixed','free']):
        self.file_name = file_name
        self.binning = binning
        self.index = index
        self.file = self.open_file(self.index)
        self.data = json.load(self.file)
        self.load_data()
        self.data_dict = self.create_dictionary()
        self.df = self.create_dataframe()

    def calculate_variability(self):
        dictionary = self.data_dict
        dataframe = self.df
        
        filtered_df = self.removing_outliers()
        
        ## selecting only non-NaN values from the DataFrame for flux and flux_error
        flux_non_nan_values = filtered_df.dropna(subset=['flux'])
        flux_error_non_nan_values = filtered_df.dropna(subset=['flux_error'])
        flux_ULs_non_nan_values = filtered_df.dropna(subset=['flux_upper_limits'])

        ## get the indexes (time) of the non-NaN values
        self.time_flux_non_nan = flux_non_nan_values.index
        self.time_flux_error_non_nan = flux_error_non_nan_values.index

        self.selected_flux_values = flux_non_nan_values['flux']
        self.selected_flux_error_values = flux_error_non_nan_values['flux_error']

        ##### normalized excess variance #####
        
        F_av = np.average(self.selected_flux_values)  # simple average
        n = len(self.selected_flux_values)
        
        if n != 1:
            s_squared = (1 / (n - 1)) * sum((F_i - F_av)**2 for F_i in self.selected_flux_values)
        else:
            s_squared = (1 / (n)) * sum((F_i - F_av)**2 for F_i in self.selected_flux_values)
            print(f'\nthe source {self.name} has only 1 flux point selected!')
            print(f'\n -> size ULs: {len(self.flux_upper_limits)}')
            print(f' -> size flux points: {len(self.flux)}')
            print(f'\n -> AFTER selection, size ULs: {len(flux_ULs_non_nan_values)}, size flux: {len(flux_non_nan_values)}')
            
        if n != 0:
            mse = (1/n) * sum(sigma_i**2 for sigma_i in self.selected_flux_error_values)
        else:
            n=1
            mse = (1/n) * sum(sigma_i**2 for sigma_i in self.selected_flux_error_values)
            print(f'\nthe source {self.name} has NO flux points selected!')
            print(f'\n -> size ULs: {len(self.flux_upper_limits)}')
            print(f' -> size flux points: {len(self.flux)}')
            print(f'\n -> AFTER selection, size ULs: {len(flux_ULs_non_nan_values)}, size flux: {len(flux_non_nan_values)}')
            
        excess_variance = s_squared - mse
        
        self.normalized_excess_variance = excess_variance / F_av**2
        
        if n != 0:
            term1 = np.sqrt(2/n) * ( mse / (F_av**2) )
            term2 = np.sqrt(mse/n) * ( 2 / F_av )
        else:
            n=1
            term1 = np.sqrt(2/n) * ( mse / (F_av**2) )
            term2 = np.sqrt(mse/n) * ( 2 / F_av )
            print(f'the source {self.name} has NO flux points selected! DO NOT trust this value!')
        
        self.unc_normalized_excess_variance = np.sqrt( (term1)**2 + ( (term2)**2 * self.normalized_excess_variance) )
        
        ##### Fractional Variability #####
        
        self.frac_variability = np.sqrt( max(self.normalized_excess_variance, 0) )  # 4FGL paper: max(term_max, 0)
        
        if n != 0:
            factor1 = np.sqrt( 1 / (2*n) ) * mse / ( F_av**2 )
            factor2 = np.sqrt( mse / n ) * ( 1 / F_av )
        else:
            n=1
            factor1 = np.sqrt( 1 / (2*n) ) * mse / ( F_av**2 )
            factor2 = np.sqrt( mse / n ) * ( 1 / F_av )
        
        if (self.frac_variability == 0):
            self.unc_frac_variability = 0.1
        else:
            self.unc_frac_variability = np.sqrt( ( (factor1)**2 / self.normalized_excess_variance ) + (factor2)**2 )
       
        return self.normalized_excess_variance, self.unc_normalized_excess_variance, self.frac_variability, self.unc_frac_variability

