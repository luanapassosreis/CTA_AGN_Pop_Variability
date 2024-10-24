

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


class Estimate_variability:
    
    def __init__(self, filtered_df):
        self.df = filtered_df
        self.drop_NaNs_from_df()

    def drop_NaNs_from_df(self):
        ## drop NaNs for ['time_flux', 'flux', 'flux_error']
        self.filtered_flux_df = self.df.dropna(subset=['time_flux', 'flux', 'flux_error'])
        
        ## drop NaNs for ['time_flux_upper_limits', 'flux_upper_limits']
        self.filtered_upper_limits_df = self.df.dropna(subset=['time_flux_upper_limits', 'flux_upper_limits'])
        return
    
    
    def calculate_variability(self):
        
        flux = self.filtered_flux_df['flux']
        flux_error = self.filtered_flux_df['flux_error'] 

        ##### normalized excess variance #####
        
        F_av = np.average(flux)  # simple average
        n = len(flux)
        
        s_squared = (1 / (n - 1)) * sum((F_i - F_av)**2 for F_i in flux)
            
        mse = (1/n) * sum(sigma_i**2 for sigma_i in flux_error)
            
        excess_variance = s_squared - mse
        
        self.normalized_excess_variance = excess_variance / F_av**2
        
        term1 = np.sqrt(2/n) * ( mse / (F_av**2) )
        term2 = np.sqrt(mse/n) * ( 2 / F_av )
        
        self.unc_normalized_excess_variance = np.sqrt( (term1)**2 + ( (term2)**2 * self.normalized_excess_variance) )
        
        ##### Fractional Variability #####
        
        self.frac_variability = np.sqrt( max(self.normalized_excess_variance, 0) )  # 4FGL paper: max(term_max, 0)
        
        factor1 = np.sqrt( 1 / (2*n) ) * mse / ( F_av**2 )
        factor2 = np.sqrt( mse / n ) * ( 1 / F_av )
        
        if (self.frac_variability == 0):
            self.unc_frac_variability = 0.1
        else:
            self.unc_frac_variability = np.sqrt( ( (factor1)**2 / self.normalized_excess_variance ) + (factor2)**2 )
       
        return self.normalized_excess_variance, self.unc_normalized_excess_variance, self.frac_variability, self.unc_frac_variability

