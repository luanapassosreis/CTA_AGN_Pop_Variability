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
