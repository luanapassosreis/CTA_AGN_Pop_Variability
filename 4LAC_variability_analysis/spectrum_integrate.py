import numpy as np
import pandas as pd

from astropy import units as u
from scipy import integrate

## defining Energy range in log-scale (0.1-100 GeV) [erg]
E = ((np.logspace(np.log10(0.1), np.log10(100), 100) * u.GeV).to('erg')).value

def integrate_spectrum_flux(source_name, dataframe, y_min, y_max):
    '''
    This function returns the total flux obtained by the integral of the SED.
    The calculation is done folowing the formulation of
    Abdollahi, S. et al., 2020, The Astrophysical Journal Supplement Series, Volume 247.
    Arguments:
    source_name : name of the source in the format''J0001.2+4741' (string);
    dataframe   : [4lac_dr3] dataframe that contains the source and its important info (df);
    y_min       : [GeV] lower limit of the integral in GeV (0.1 GeV) (float);
    y_max       : [GeV] upper limit of the integral in GeV (100 GeV) (float).
    Output:
    spectrum_flux: integrated flux from the SED (float);
    diff_flux    : differential flux dN/dE (float);
    spec_type    : [LogParabola/ PowerLaw] spectrum type (float).
    '''
    
    ## get the index of the source
    index = dataframe[dataframe['Source_Name'] == source_name].index

    for i in index:

        ## Pivot_Energy [erg]
        E_0 = ((dataframe.loc[i,'Pivot_Energy'] * u.MeV).to('erg')).value

        ## ---------- PowerLaw ----------
        if dataframe.loc[i,'SpectrumType'] == 'PowerLaw':
            spec_type = "PowerLaw"

            ## PL_Flux_Density [erg-1 cm-2 s-1]
            K = ((dataframe.loc[i,'PL_Flux_Density'] * u.MeV**-1 * u.cm**-2 * u.s**-1).to('erg-1 cm-2 s-1')).value
            
            ## PL_Index
            alpha = dataframe.loc[i,'PL_Index']
            
            ## if PowerLaw, beta = 0
            beta = 0

            
        ## ---------- LogParabola ----------
        elif dataframe.loc[i,'SpectrumType'] == 'LogParabola':
            spec_type = "LogParabola"

            ## LP_Flux_Density [erg-1 cm-2 s-1]
            K = ((dataframe.loc[i,'LP_Flux_Density'] * u.MeV**-1 * u.cm**-2 * u.s**-1).to('erg-1 cm-2 s-1')).value
            
            ## LP_Index
            alpha = dataframe.loc[i,'LP_Index']
            
            ## LP_beta
            beta = dataframe.loc[i,'LP_beta']

            
        ## ---------- in case there is an error ----------
        else:
            print('### error ###')
    
    
    dNdE = K * ((E/E_0)**(- alpha - beta * np.log(E/E_0)))
    
    diff_flux = E**2 * dNdE
    
    flux_from_spectrum = integrate.quad(lambda x: K * ((x/E_0)**(- alpha - beta * np.log(x/E_0))), 
                                          (y_min*u.GeV).to('erg').value, (y_max*u.GeV).to('erg').value)
    
    ## [ph cm-2 s-1]
    spectrum_flux = flux_from_spectrum[0] # returning only the first value of integrateflux
    
    return spectrum_flux, diff_flux, spec_type