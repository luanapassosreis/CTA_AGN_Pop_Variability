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



Mdot_Edd = 1.45e18 * m # [g s-1] (KGS15 page 4)
R_s = 2.96e5 * m # [cm] = 2GM/c^2 (KGS15 page 4)


# Gamma = (1 + (va0 / c )**2 )**(-1/2) # KGS15 page 4
# q = ( 1 - ( 3 * R_s / R_x)**(1/2) )**(1/4) # KGS15 page 4


class plots:
    def __init__(self, r_x, l, l_x, mdot, m):
        self.r_x = r_x
        self.l = l
        self.l_x = l_x
        self.mdot = mdot
        self.m = m
        self.Gamma = 1 / np.sqrt(2)
        # self.Gamma = (1 + (va0 / c )**2 )**(-1/2)
        self.q = ( 1 - ( 3 / self.r_x )**(1/2) )**(1/4)

        
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

    
    