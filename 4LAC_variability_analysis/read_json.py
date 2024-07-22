

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



class read_json_file:
    
    def __init__(self, file_name, binning=['3-days','weekly','monthly'], index=['fixed','free']):
        self.file_name = file_name
        self.binning = binning
        self.index = index
        self.file = self.open_file(self.index)
        self.data = json.load(self.file)
        self.load_data()
        self.data_dict = self.create_dictionary()
        self.df = self.create_dataframe()
        
    def open_file(self, index=['fixed','free']):
        self.path_downloaded_lc_catalog = '../4LAC_lightcurve_downloader_v3/resulting_catalogs/input_lightcurve_downloads_v3'
        
        if self.index == 'fixed':
            self.path_folder = 'fixed_indexed_lightcurves'
            if self.binning == '3-days':
                return open(f'{self.path_downloaded_lc_catalog}/{self.path_folder}/3days_ts1_fixedindex_lightcurves/{self.file_name}')
            elif self.binning == 'weekly':
                return open(f'{self.path_downloaded_lc_catalog}/{self.path_folder}/weekly_ts1_fixedindex_lightcurves/{self.file_name}')
            elif self.binning == 'monthly':
                return open(f'{self.path_downloaded_lc_catalog}/{self.path_folder}/monthly_ts1_fixedindex_lightcurves/{self.file_name}')
            else:
                raise ValueError("Invalid binning option. Choose from '3-days', 'weekly', or 'monthly'.")
        
        elif self.index == 'free':
            self.path_folder = 'free_indexed_lightcurves'
            if self.binning == '3-days':
                return open(f'{self.path_downloaded_lc_catalog}/{self.path_folder}/3days_ts1_freeindex_lightcurves/{self.file_name}')
            elif self.binning == 'weekly':
                return open(f'{self.path_downloaded_lc_catalog}/{self.path_folder}/weekly_ts1_freeindex_lightcurves/{self.file_name}')
            elif self.binning == 'monthly':
                return open(f'{self.path_downloaded_lc_catalog}/{self.path_folder}/monthly_ts1_freeindex_lightcurves/{self.file_name}')
            else:
                raise ValueError("Invalid binning option. Choose from '3-days', 'weekly', or 'monthly'.")
            
    def load_data(self):
        self.name = self.file_name[5:-5]
        
        data = self.data
        ## ts = test statistics
        self.time_ts = np.array(data['ts'])[:, 0]      # [i][0]
        self.values_ts = np.array(data['ts'])[:, 1]    # [i][1]
        ## flux
        self.time_flux = np.array(data['flux'])[:,0]
        self.flux = np.array(data['flux'])[:,1]
        ## flux upper limits
        if np.array(data.get('flux_upper_limits')).ndim == 2:
            self.time_flux_upper_limits  = np.array(data['flux_upper_limits'])[:,0]
            self.flux_upper_limits  = np.array(data['flux_upper_limits'])[:,1]
        else:
            ## handle the case where "flux_upper_limits" is not present in the file
            self.time_flux_upper_limits = np.array(data['flux_upper_limits'])
            self.flux_upper_limits = np.array(data['flux_upper_limits'])
        ## flux low and high error
        self.time_flux_error = np.array(data['flux_error'])[:,0]  # [i][0]
        self.flux_low_error  = np.array(data['flux_error'])[:,1]  # [i][1]  - lower flux edge
        self.flux_high_error = np.array(data['flux_error'])[:,2]  # [i][2]  - high edge
        self.flux_error = self.low_and_high_errors()
        ## fit convergence
        self.time_fit_convergence = np.array(data['fit_convergence'])[:,0]
        self.fit_convergence = np.array(data['fit_convergence'])[:,1]  # [i][1] - should be zero!
        ## dlogl
        self.dlogl = np.array(data['dlogl'])
        
    def low_and_high_errors(self):
        ## selecting only the error bar: flux_error = flux - flux_low_error // flux_high_error - flux
        flux_point_low_error = self.flux - self.flux_low_error
        flux_point_high_error = self.flux_high_error - self.flux
        ## creating masks to select the larger error
        mask_high_larger_than_low = flux_point_high_error >= flux_point_low_error   # mask when high errors larger than low errors
        mask_low_larger_than_high = flux_point_low_error > flux_point_high_error    # low errors larger than high errors
        ## flux_error
        flux_error = np.zeros_like(self.flux_high_error)
        flux_error[mask_high_larger_than_low] = flux_point_high_error[mask_high_larger_than_low]
        flux_error[mask_low_larger_than_high] = flux_point_low_error[mask_low_larger_than_high]
        return flux_error
    
    def create_dictionary(self):
        data_dict = {
            'name': self.name,
            'time_flux': self.time_flux,
            'flux': self.flux,
            'time_flux_upper_limits': self.time_flux_upper_limits,
            'flux_upper_limits': self.flux_upper_limits,
            'time_flux_error': self.time_flux_error,
            'flux_low_error': self.flux_low_error,
            'flux_high_error': self.flux_high_error,
            'flux_error': self.flux_error,
            'time_ts': self.time_ts,
            'values_ts': self.values_ts,
            'time_fit_convergence': self.time_fit_convergence,
            'fit_convergence': self.fit_convergence,
            'dlogl': self.dlogl
        }
        return data_dict
    
    def create_dataframe(self):
        ## create a DataFrame for the outlier treatment
        df = pd.DataFrame()
        data_dict = self.data_dict

        ## assign 'time_fit_convergence' as index - the total number of observations
        df['time_fit_convergence'] = data_dict['time_fit_convergence']
        df.set_index('time_fit_convergence', inplace=True)
        df['fit_convergence'] = data_dict['fit_convergence']

        df['time_ts'] = data_dict['time_ts']
        df['values_ts'] = data_dict['values_ts']
        
        df['dlogl'] = data_dict['dlogl']

        ## insert columns with NaN values
        df['flux'] = np.nan
        df['flux_upper_limits'] = np.nan
        df['flux_error'] = np.nan

        ## fill in the values where 'time_flux' matches the index 'time_fit_convergence'
        mask_flux = df.index.isin(data_dict['time_flux'])
        df.loc[mask_flux, 'flux'] = data_dict['flux']

        ## 'time_upper_lim' matches the index 'time_fit_convergence'
        mask_upper_lim = df.index.isin(data_dict['time_flux_upper_limits'])
        df.loc[mask_upper_lim, 'flux_upper_limits'] = data_dict['flux_upper_limits']

        ## 'time_flux_error' matches the index
        mask_flux_error = df.index.isin(data_dict['time_flux_error'])
        df.loc[mask_flux_error, 'flux_error'] = data_dict['flux_error']
        
        return df
    
    def load_free_dataframe(self):
        self.file_free = self.open_file('free')
        self.data_free = json.load(self.file_free)
        self.load_data()
        self.data_dict_free = self.create_dictionary()
        self.df_free = self.create_dataframe()
        
        return self.df_free

    