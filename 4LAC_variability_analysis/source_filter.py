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
    

def filter_source_flux(source_dataframe):
    filtered_df = source_dataframe.copy()
    
    ## to turn the point into an Upper Limit
    indices_to_replaceUL_ts = (source_dataframe['values_ts'] < 4) # TS < 10 -> point should be an UL
    filtered_df.loc[indices_to_replaceUL_ts,
                         'time_flux_upper_limits'] = source_dataframe.loc[indices_to_replaceUL_ts, 'time_flux']
    filtered_df.loc[indices_to_replaceUL_ts,
                         'flux_upper_limits'] = source_dataframe.loc[indices_to_replaceUL_ts, 'flux']
    filtered_df.loc[indices_to_replaceUL_ts, ['time_flux', 'flux', 'flux_error']] = np.nan

    ## remove bins with exposure < 1e7 cm^2 s
    exposure = source_dataframe['flux'] / (source_dataframe['flux_error'] ** 2)
    indices_to_remove_exposure = (exposure < 1e7)
    
    ## to remove points
    indices_to_remove_fit = (source_dataframe['fit_convergence'] != 0) # fit_convergence != 0
    indices_to_remove_flux_error = (source_dataframe['flux_error'] == 0) # flux_error == 0
    
    indices_to_remove = indices_to_remove_fit | indices_to_remove_flux_error | indices_to_remove_exposure

    ## make the point an UL, with unc_flu_UL = 0 and flux_UL = 0
    filtered_df.loc[indices_to_remove,
                         'time_flux_upper_limits'] = source_dataframe.loc[indices_to_remove, 'time_flux']
    filtered_df.loc[indices_to_remove,
                         'flux_upper_limits'] = source_dataframe.loc[indices_to_remove, 'flux']
    filtered_df.loc[indices_to_remove, ['time_flux', 'flux', 'flux_error']] = np.nan


    # indices_to_replacefree_dlogl = (source_dataframe['dlogl'] > 5) # 2*dlogl > 10 -> should have free index
    # filtered_df.loc[indices_to_replacefree_dlogl, 'flux'] = df_free.loc[indices_to_replacefree_dlogl, 'flux']
    # filtered_df.loc[indices_to_replacefree_dlogl, 'flux_error'] = df_free.loc[indices_to_replacefree_dlogl, 'flux_error']

    # print(f'{len(indices_to_replacefree_dlogl)} points were replaced in {self.name} fixed -> free index!')

    return filtered_df
