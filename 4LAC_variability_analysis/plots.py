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



Mdot_Edd = 1.45e18 * m # [g s-1] (KGS15 page 4)
R_s = 2.96e5 * m # [cm] = 2GM/c^2 (KGS15 page 4)


# Gamma = (1 + (va0 / c )**2 )**(-1/2) # KGS15 page 4
# q = ( 1 - ( 3 * R_s / R_x)**(1/2) )**(1/4) # KGS15 page 4


class Plots:
    def __init__(self, r_x, l, l_x, mdot, m):
        self.r_x = r_x
        self.l = l
        self.l_x = l_x
        self.mdot = mdot
        self.m = m
        self.Gamma = 1 / np.sqrt(2)
        # self.Gamma = (1 + (va0 / c )**2 )**(-1/2)
        self.q = ( 1 - ( 3 / self.r_x )**(1/2) )**(1/4)

    
    
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
    
    def lightcurve(self):
        '''Calculate inner disk magnetic field intensity.
        Eq.(2) of KGS15.'''
        # [G]
        return 9.96e8 * self.r_x**(-5/4) * self.mdot**(1/2) * self.m**(-1/2)

    
    def spectrum(self):
        '''Calculate coronal particle number density.
        Eq.(7) of KGS15.'''
         # [cm-3]
        return 8.02e18 * self.Gamma**(1/2) * self.r_x**(-3/8) * self.l**(-3/4) * self.q**(-2) * self.mdot**(1/4) * self.m**(-1)
