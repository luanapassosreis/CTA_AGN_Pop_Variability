import numpy as np
import pandas as pd

import glob
import os


def generate_list_sources(binning=['3-days', 'weekly', 'monthly'], index=['fixed', 'free']):
    path_downloaded_lc_catalog = '../4LAC_lightcurve_downloader_v3/resulting_catalogs/input_lightcurve_downloads_v3'
    path_folder = f'{index}_indexed_lightcurves'
    
    if binning == '3-days':
        files = glob.glob(f'{path_downloaded_lc_catalog}/{path_folder}/3days_ts1_{index}index_lightcurves/*.json')
    elif binning == 'weekly':
        files = glob.glob(f'{path_downloaded_lc_catalog}/{path_folder}/weekly_ts1_{index}index_lightcurves/*.json')
    elif binning == 'monthly':
        files = glob.glob(f'{path_downloaded_lc_catalog}/{path_folder}/monthly_ts1_{index}index_lightcurves/*.json')
    else:
        raise ValueError("Invalid binning option. Choose from '3-days', 'weekly', or 'monthly'.")
    
    file_list = sorted([os.path.basename(file) for file in files])
    
    return file_list



def filter_outliers(source_dataframe):
    filtered_df = source_dataframe.copy()
    
    # print('# of bins:', len(source_dataframe))
    
    exposure = source_dataframe['flux'] / (source_dataframe['flux_error'] ** 2)
    
    ## to remove
    indices_ts = (source_dataframe['values_ts'] < 4) # TS < 4
    indices_ferror = (source_dataframe['flux_error'] == 0) # flux_error == 0
    indices_fit = (source_dataframe['fit_convergence'] != 0) # fit_convergence != 0
    indices_expo = (exposure < 1e7) # exposure < 1e7 cm2 s
    
    indices_to_remove = indices_ts | indices_ferror | indices_fit | indices_expo
    
    n_unconstrained = len(source_dataframe[indices_to_remove])

    ## unconstrained bins will become NaN
    filtered_df.loc[indices_to_remove, ['flux', 'flux_error']] = np.nan
    
    return filtered_df, n_unconstrained





def input_upperL(filtered_df, case=['average', 'zero']):
    inputed_df = filtered_df.copy()

    ## bins where 1<TS<2
    ts_condition = (filtered_df['values_ts'] > 1) & (filtered_df['values_ts'] < 2)
    
    ## median of 'flux' for bins satisfying the TS condition
    median_flux = filtered_df.loc[ts_condition, 'flux'].median()
    
    ## replace all initial NaN bins in 'flux_error' with the median_flux
    first_valid_index = inputed_df['flux_error'].first_valid_index()
    
    if first_valid_index is not None:
        ## fill NaNs in 'flux_error' before the first valid index with median_flux
        inputed_df.loc[:first_valid_index, 'flux_error'] = inputed_df['flux_error'].fillna(median_flux)

        
    ## remaining NaN values in 'flux'
    indices_UL = np.isnan(inputed_df['flux'])
    average_flux = inputed_df['flux'].dropna().mean()
    
    ## flux_errors
    
    ## replace NaN in flux_error by the previous bin flux (that's not NaN)
    ## if the first bin has a NaN, it will not be inputed
    inputed_df['flux_error'] = inputed_df['flux_error'].fillna(inputed_df['flux'].ffill())
    
    ## input average value for single bin - 3-days LCs
    
    ## flux    
        
    if case == 'average':
        inputed_df.loc[indices_UL, 'flux'] = average_flux  # Input the average of fluxes
    
    elif case == 'zero':
        inputed_df.loc[indices_UL, 'flux'] = 0  # Input zero
    
    else:
        raise ValueError("Invalid option for case. Choose either 'average' or 'zero'.")
    
    return inputed_df





## latest test

# def input_upperL(filtered_df, case=['average', 'zero']):
#     inputed_df = filtered_df.copy()
    
#     indices_UL = ~np.isnan(filtered_df['flux_upper_limits']) # inverted mask
    
#     inputed_df.loc[indices_UL, ['flux_error']] = filtered_df.loc[indices_UL, 'flux_upper_limits']
#     inputed_df.loc[indices_UL, ['flux_upper_limits']] = np.nan
    
#     indices_input = np.isnan(filtered_df['flux'])
    
#     average_flux = np.average(filtered_df['flux'].dropna())
#     # average_unc = np.average(filtered_df['flux_error'].dropna())
        
#     if case == 'average':
#         # inputed_df.loc[indices_UL, ['flux_error']] = average_unc
#         inputed_df.loc[indices_input, 'flux'] = average_flux  # input the average of fluxes
    
#     elif case == 'zero':
#         inputed_df.loc[indices_input, 'flux'] = 0  # Input zero
    
#     else:
#         raise ValueError("Invalid option for case. Choose either 'average' or 'zero'.")
    
#     return inputed_df
    

    
## here the unconstrained points uncertainty were not taken into account

# def input_upperL(filtered_df, case=['average', 'zero']):
#     inputed_df = filtered_df.copy()
    
#     indices_UL = np.isnan(filtered_df['flux'])
    
#     average = np.average(filtered_df['flux'].dropna())
    
#     inputed_df.loc[indices_UL, ['flux_error']] = filtered_df.loc[indices_UL, 'flux_upper_limits']
#     inputed_df.loc[indices_UL, ['flux_upper_limits']] = np.nan
        
#     if case == 'average':
#         inputed_df.loc[indices_UL, 'flux'] = average  # input the average of fluxes
    
#     elif case == 'zero':
#         inputed_df.loc[indices_UL, 'flux'] = 0  # Input zero
    
#     else:
#         raise ValueError("Invalid option for case. Choose either 'average' or 'zero'.")
    
#     return inputed_df
