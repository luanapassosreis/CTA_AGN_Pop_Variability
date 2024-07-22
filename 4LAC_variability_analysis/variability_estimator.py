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
        
    def open_file(self, index=['fixed','free']):
        self.path_4lac_dr3_catalog = '../4LAC_catalog_generator_v3/resulting_catalogs/agn_pop_4lac_dr3.ecsv'
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
        ## accessing instance variable
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

    def removing_outliers(self):
        dataframe = self.df
        df_free = self.load_free_dataframe()

        indices_to_remove_fit = (dataframe['fit_convergence'] != 0) # fit_convergence != 0
        indices_to_remove_flux_error = (dataframe['flux_error'] == 0) # flux_error == 0
        indices_to_remove = indices_to_remove_fit | indices_to_remove_flux_error

        dataframe.loc[indices_to_remove, ['flux', 'flux_upper_limits', 'flux_error']] = np.nan

        indices_to_replaceUL_ts = (dataframe['values_ts'] < 10) # TS < 10 -> point should be an UL
        dataframe.loc[indices_to_replaceUL_ts, 'flux_upper_limits'] = dataframe.loc[indices_to_replaceUL_ts, 'flux']
        dataframe.loc[indices_to_replaceUL_ts, ['flux', 'flux_error']] = np.nan
        
        ## Remove bins with exposure < 1e7 cm^2 s
        exposure = dataframe['flux'] / (dataframe['flux_error'] ** 2)
        indices_to_remove_exposure = (exposure < 1e7)
        dataframe.loc[indices_to_remove_exposure, ['flux', 'flux_upper_limits', 'flux_error']] = np.nan

        
        # indices_to_replacefree_dlogl = (dataframe['dlogl'] > 5) # 2*dlogl > 10 -> should have free index
        # dataframe.loc[indices_to_replacefree_dlogl, 'flux'] = df_free.loc[indices_to_replacefree_dlogl, 'flux']
        # dataframe.loc[indices_to_replacefree_dlogl, 'flux_error'] = df_free.loc[indices_to_replacefree_dlogl, 'flux_error']
        
        # print(f'{len(indices_to_replacefree_dlogl)} points were replaced in {self.name} fixed -> free index!')
        
        return dataframe

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

    def convert_MET_UTC(self, time_MET):
        time_Unix = Time(time_MET, format='unix', scale='utc')
        time_difference = Time('2001-01-01', format='iso', scale='utc')
        time_difference.format = 'unix'
        time_difference.value
        time_MET_copy = np.copy(time_MET)
        time_MET_copy += time_difference.value
        time_Unix = Time(time_MET_copy, format='unix', scale='utc')
        time_Unix.format = 'iso'
        time_Unix
        time_UTC = []
        for i in range(len(time_Unix.value)):
            time_UTC.append(datetime.strptime(time_Unix.value[i][:10], '%Y-%m-%d'))
        return time_UTC

    def plot_lc(self, ylim, flux_from_spectrum, convert_time=True):
        if convert_time:
            time = self.convert_MET_UTC(self.time_flux)
            time_error = self.convert_MET_UTC(self.time_flux_error)
            time_upper_lim = self.convert_MET_UTC(self.time_flux_upper_limits)
        else:
            time = self.time_flux
            time_error = self.time_flux_error
            time_upper_lim = self.time_flux_upper_limits

        plt.figure(figsize=(17,5), dpi=300)
        plt.plot(time, self.flux, '.', markersize=10, label='Flux Points')
        plt.plot(time, self.flux, linewidth=0.4, color='black')
        plt.plot(time_upper_lim, self.flux_upper_limits, 'v', color='gray', markersize=3, alpha=0.45, label='Upper Limits')
        plt.errorbar(time, self.flux, yerr=self.flux_high_error-self.flux, linewidth=0.2, color='black', alpha=0.9)
        plt.errorbar(time, self.flux, yerr=self.flux-self.flux_low_error, linewidth=0.2, color='black', alpha=0.9)
        plt.legend(fontsize=15)
        plt.ylim(0, ylim)
        plt.title(f'4FGL+{self.name} Light Curve', fontsize=20)
        plt.ylabel('Photon Flux (0.1-100 GeV ph $cm^{-2}$ $s^{-1}$)', fontsize=15)
        plt.xlabel('Date (UTC)', fontsize=15)
        return