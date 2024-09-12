from source_filter import *
from spectrum_integrate import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.time import Time,TimeUnix
from datetime import datetime

from scipy.fft import fft, fftn, fftfreq
from scipy.signal import lombscargle
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


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

import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter


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
        self.filtered_df = filter_source_flux(self.source_df)
    
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
        plt.title(f'4FGL+{self.source_name} Complete Light Curve -- {binning} cadence', fontsize=20)
        plt.ylabel('Photon Flux (0.1-100 GeV ph $cm^{-2}$ $s^{-1}$)', fontsize=15)
        if convert_time:
            plt.xlabel('Date (UTC)', fontsize=15)
        else:
            plt.xlabel('Date (MET seconds)', fontsize=15)
        return
    
    def drop_NaNs_from_df(self):
        ## drop NaNs for ['time_flux', 'flux', 'flux_error']
        self.filtered_flux_df = self.filtered_df.dropna(subset=['time_flux', 'flux', 'flux_error'])
        
        ## drop NaNs for ['time_flux_upper_limits', 'flux_upper_limits']
        self.filtered_upper_limits_df = self.filtered_df.dropna(subset=['time_flux_upper_limits', 'flux_upper_limits'])
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
        plt.title(f'4FGL+{self.source_name} Filtered Light Curve -- {binning} cadence', fontsize=20)
        plt.ylabel('Photon Flux (0.1-100 GeV ph $cm^{-2}$ $s^{-1}$)', fontsize=15)
        if convert_time:
            plt.xlabel('Date (UTC)', fontsize=15)
        else:
            plt.xlabel('Date (MET seconds)', fontsize=15)
        return

    
    def spectrum(self, power, x_dlim, x_ulim, y_dlim, y_ulim):
    
        spectrum_flux, diff_flux, spec_type = integrate_spectrum_flux(self.source_name, self.df_4lac,
                                                                      y_min=0.1, y_max=100)
        
        plt.figure(figsize=(7,5), dpi=100)

        plt.plot(((E*u.erg).to('GeV')).value, diff_flux*10**(power), '+', markersize=2, color='black')
        plt.plot(((E*u.erg).to('GeV')).value, diff_flux*10**(power), '--', linewidth=0.4, color='black')

        plt.xscale('log')
        plt.yscale('log')
        plt.grid()

        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        plt.xlim(x_dlim, x_ulim)
        plt.ylim(y_dlim, y_ulim)


        plt.title(f'4FGL+{self.source_name} - {spec_type} Spectrum', fontsize='large')
        plt.ylabel(r'$\nu\ F_{\nu}$ [ $ 10^{- %d } $ erg $cm^{-2}$ $s^{-1}$]' % (power), fontsize=12)
        plt.xlabel('Energy (GeV)', fontsize=12)

        return
    
    def norm_excess_var_3days_monthly(self):
        
        return
    
    
    def exposure(self):
        x = self.filtered_flux_df['flux'] / (self.filtered_flux_df['flux_error']**2)
        y = self.filtered_flux_df['flux'] / np.median(self.filtered_flux_df['flux'])

        plt.figure(figsize=(8, 7))
        plt.scatter(x, y, alpha=0.5, s=50, color='green')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e2,1e12)
        plt.xlabel('phi / sigma^2')
        plt.ylabel('phi / phi_median')
        plt.title('Normalized Flux vs Flux/Median Flux')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return
    
    
    def test_statistics(self):
        
        return
    
    
    def delta_loglikelihood(self):
        
        return
    
    
    #------------------------------------- just trying a few things
    
    
    def fourier_transform_with_interpolation(self, flux, time):
        ## create an evenly spaced time grid
        time_grid = np.linspace(np.min(time), np.max(time), len(time))
        
        ## interpolate the flux data to this grid
        interp_func = interp1d(time, flux, kind='linear')
        flux_interp = interp_func(time_grid)
        
        ## detrend the flux data
        flux_detrended = flux_interp - np.mean(flux_interp)

        N = len(flux_detrended)
        T = np.mean(np.diff(time_grid)) # time step

        # Fourier Transform using fftn
        flux_fft = fftn(flux_detrended)
        frequencies = fftfreq(N, T) # frequency bins

        power_spectrum = np.abs(flux_fft)**2

        ## keep positive frequencies
        positive_freqs = frequencies > 0
        frequencies = frequencies[positive_freqs]
        power_spectrum = power_spectrum[positive_freqs]

        plt.figure(figsize=(10, 6))
        plt.loglog(frequencies, power_spectrum)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectrum')
        plt.title('Power Spectrum of Flux Variability (Interpolated Data)')
        plt.grid(True)
        plt.show()
    
    
    def fourier_transform(self, flux, time):
        ## check if the data is evenly spaced
        dt = np.diff(time)
        if not np.allclose(dt, dt[0]):
            raise ValueError("Time data must be evenly spaced for accurate Fourier Transform.")
            

        ## subtract the mean flux (detrending)
        flux_detrended = flux - np.mean(flux)

        N = len(flux_detrended) # sample
        T = np.mean(dt) # time step

        ## Fourier Transform using fft
        flux_fft = fft(flux_detrended)
        frequencies = fftfreq(N, T)  # Frequency bins

        power_spectrum = np.abs(flux_fft)**2

        ## keep positive frequencies
        positive_freqs = frequencies > 0
        frequencies = frequencies[positive_freqs]
        power_spectrum = power_spectrum[positive_freqs]

        plt.figure(figsize=(10, 6))
        plt.loglog(frequencies, power_spectrum)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectrum')
        plt.title('Power Spectrum of Flux Variability')
        plt.grid(True)
        plt.show()


    def lomb_scargle_transform(self, flux, time):
        ## normalize time and flux
        time = time - np.min(time)  # Shift time to start at 0
        flux_detrended = flux - np.mean(flux)  # Detrend the flux

        ## define frequency range
        min_freq = 1 / (max(time) - min(time))  
        max_freq = 0.5 / np.median(np.diff(time))  # Nyquist frequency
        frequencies = np.linspace(min_freq, max_freq, 1000)  # grid

        power_spectrum = lombscargle(time, flux_detrended, 2 * np.pi * frequencies)

        plt.figure(figsize=(10, 6))
        plt.loglog(frequencies, power_spectrum)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectrum')
        plt.title('Lomb-Scargle Power Spectrum of Flux Variability')
        plt.grid(True)
        plt.show()
        
        
    def lomb_scargle_transform_and_fit(self, flux, time):
        def power_law(frequency, A, alpha):
            """Power-law model: P(f) = A * f^(-alpha)"""
            return A * frequency ** (-alpha)

        def log_parabola(frequency, A, alpha, beta):
            """Log-parabola model: P(f) = A * f^(-alpha - beta * log(f))"""
            return A * frequency ** (-alpha - beta * np.log(frequency))
        
        ## normalize time and flux
        time = time - np.min(time)
        flux_detrended = flux - np.mean(flux)

        ## frequency range
        min_freq = 1 / (max(time) - min(time))
        max_freq = 0.5 / np.median(np.diff(time))
        frequencies = np.linspace(min_freq, max_freq, 1000)

        ## Lomb-Scargle periodogram
        power_spectrum = lombscargle(time, flux_detrended, 2 * np.pi * frequencies)

        ## fit Power-Law model
        popt_pl, pcov_pl = curve_fit(power_law, frequencies, power_spectrum)
        A_fit_pl, alpha_fit_pl = popt_pl

        ## fit Log-Parabola model
        # Provide initial guesses for the parameters
        p0_lp = [A_fit_pl, alpha_fit_pl, 0.5]
        popt_lp, pcov_lp = curve_fit(log_parabola, frequencies, power_spectrum, p0=p0_lp)
        A_fit_lp, alpha_fit_lp, beta_fit_lp = popt_lp

        plt.figure(figsize=(12, 8))
        plt.loglog(frequencies, power_spectrum, label='Lomb-Scargle Power Spectrum', color='black')
        plt.loglog(frequencies, power_law(frequencies, *popt_pl), label=f'Power-Law Fit: $P(f) = {A_fit_pl:.2e} f^{{-{alpha_fit_pl:.2f}}}$', linestyle='--')
        # plt.loglog(frequencies, log_parabola(frequencies, *popt_lp), label=f'Log-Parabola Fit: $P(f) = {A_fit_lp:.2e} f^{{-{alpha_fit_lp:.2f} - {beta_fit_lp:.2f} \log(f)}}$', linestyle='--')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectrum')
        plt.title('Lomb-Scargle Power Spectrum with Power-Law Fit')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Power-Law Fit: A = {A_fit_pl:.2e}, alpha = {alpha_fit_pl:.2f}")
        # print(f"Log-Parabola Fit: A = {A_fit_lp:.2e}, alpha = {alpha_fit_lp:.2f}, beta = {beta_fit_lp:.2f}")
