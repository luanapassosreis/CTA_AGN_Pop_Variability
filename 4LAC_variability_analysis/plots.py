from source_filter import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.time import Time,TimeUnix
from datetime import datetime


# import seaborn as sns

# import astropy.table
# from astropy import units as u
# from astropy.io import fits
# from astropy.io import ascii
# from astropy.table import QTable, Table

# import os
# import csv
# import glob
# import math
# import json
# import statistics

# import scipy.optimize as sp
# import scipy.odr.odrpack as odrpack
# from scipy import signal, integrate
# from scipy.fft import fft, fftfreq
# from scipy.stats import pearsonr

# import matplotlib.ticker as mticker
# from matplotlib.ticker import FormatStrFormatter


class Plots:
    '''
    This class reads the 4lac_dr3 dataframe, the source dictionary and dataframe given
    and returns several plots according to what you may want.
    Arguments:
    source_name : source name in a format like 'J0001.2-0747'  (string);
    df_agn_pop_4lac_dr3   : ['3-days','weekly','monthly'] desired cadence to obtain the info (string);
    source_dictionary     : ['fixed','free'] desired index of the lightcurve (string);
    source_dataframe : (df).
    '''
    
    def __init__(self, source_name, df_agn_pop_4lac_dr3, source_dictionary, source_dataframe):
        self.source_name = source_name
        self.df_4lac = df_agn_pop_4lac_dr3
        self.source_dict = source_dictionary
        self.source_df = source_dataframe
    
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
    
    def lightcurve_complete(self, binning, ylim, convert_time=True):
        if convert_time:
            time_flux = self.convert_MET_UTC(self.source_dict['time_flux'])
            # time_flux_error = self.convert_MET_UTC(self.source_dict['time_flux_error'])
            time_flux_upper_limits = self.convert_MET_UTC(self.source_dict['time_flux_upper_limits'])
        else:
            time_flux = self.source_dict['time_flux']
            # time_flux_error = self.source_dict['time_flux_error']
            time_flux_upper_limits = self.source_dict['time_flux_upper_limits']

        plt.figure(figsize=(17,5), dpi=300)
        
        plt.plot(time_flux, self.source_dict['flux'], '.', markersize=10, label='Flux Points')
        plt.plot(time_flux, self.source_dict['flux'], linewidth=0.4, color='black')
        plt.plot(time_flux_upper_limits, self.source_dict['flux_upper_limits'], 'v',
                 color='gray', markersize=3, alpha=0.45, label='Upper Limits')
        
        plt.errorbar(time_flux, self.source_dict['flux'],
                     yerr=self.source_dict['flux_high_error']-self.source_dict['flux'],
                     linewidth=0.2, color='black', alpha=0.9)
        plt.errorbar(time_flux, self.source_dict['flux'],
                     yerr=self.source_dict['flux']-self.source_dict['flux_low_error'],
                     linewidth=0.2, color='black', alpha=0.9)
        
        plt.legend(fontsize=15)
        plt.ylim(0, ylim)
        plt.title(f'4FGL+{self.source_name} Light Curve -- {binning} cadence', fontsize=20)
        plt.ylabel('Photon Flux (0.1-100 GeV ph $cm^{-2}$ $s^{-1}$)', fontsize=15)
        if convert_time:
            plt.xlabel('Date (UTC)', fontsize=15)
        else:
            plt.xlabel('Date (MET seconds)', fontsize=15)
        return
    
    def drop_NaNs_from_df(self):
        ## drop NaNs for ['time_flux', 'flux', 'flux_error']
        self.filtered_flux_df = self.source_df.dropna(subset=['time_flux', 'flux', 'flux_error'])
        
        ## drop NaNs for ['time_flux_upper_limits', 'flux_upper_limits']
        self.filtered_upper_limits_df = self.source_df.dropna(subset=['time_flux_upper_limits', 'flux_upper_limits'])
        return
    
    def lightcurve_filtered(self, binning, ylim, convert_time=True):
        ## drop NaNs first
        self.drop_NaNs_from_df()
        
        if convert_time:
            time_flux = self.convert_MET_UTC(self.filtered_flux_df['time_flux'])
            time_flux_upper_limits = self.convert_MET_UTC(self.filtered_upper_limits_df['time_flux_upper_limits'])
        else:
            time_flux = self.filtered_flux_df['time_flux']
            time_flux_upper_limits = self.filtered_upper_limits_df['time_flux_upper_limits']

        plt.figure(figsize=(17,5), dpi=300)
        
        plt.plot(time_flux, self.filtered_flux_df['flux'], '.', markersize=10, label='Flux Points')
        plt.plot(time_flux, self.filtered_flux_df['flux'], linewidth=0.4, color='black')
        plt.plot(time_flux_upper_limits, self.filtered_upper_limits_df['flux_upper_limits'], 'v',
                 color='gray', markersize=3, alpha=0.45, label='Upper Limits')
        
        plt.errorbar(time_flux, self.filtered_flux_df['flux'],
                     yerr=self.filtered_flux_df['flux_error'],
                     linewidth=0.2, color='black', alpha=0.9)
        
        plt.legend(fontsize=15)
        plt.ylim(0, ylim)
        plt.title(f'4FGL+{self.source_name} Light Curve -- {binning} cadence', fontsize=20)
        plt.ylabel('Photon Flux (0.1-100 GeV ph $cm^{-2}$ $s^{-1}$)', fontsize=15)
        if convert_time:
            plt.xlabel('Date (UTC)', fontsize=15)
        else:
            plt.xlabel('Date (MET seconds)', fontsize=15)
        return

    
    def spectrum(self):
        
        return 
    
    
    
    def norm_excess_var_3days_monthly(self):
        
        return
    
    
    
    def exposure(self):
        
        return
    
    
    def test_statistics(self):
        
        return
    
    
    def delta_loglikelihood(self):
        
        return
