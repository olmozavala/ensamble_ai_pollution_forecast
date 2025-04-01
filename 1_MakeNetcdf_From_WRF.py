# %%
# %load_ext autoreload
# %autoreload 2
from proj_preproc.wrf import crop_variables_xr, crop_variables_xr_cca_reanalisis
from conf.localConstants import wrfFileType
import os
import xarray as xr
from multiprocessing import Pool
from utils.plotting import plot_variables
from io_netcdf.inout import read_wrf_files_names, read_wrf_old_files_names
from proj_preproc.wrf import calculate_relative_humidity_metpy
import matplotlib.pyplot as plt
import numpy as np
from metpy.units import units

def process_single_year(args):
    """
    Process all WRF files for a single year.
    
    Args:
        args (tuple): Contains:
            - year (int): Year to process
            - variable_names (list): Variables to extract
            - output_folder (str): Path for output netCDF files
            - output_folder_imgs (str): Path for output images
            - bbox (list): Bounding box coordinates
            - generate_images (bool): Whether to generate plots
    """
    year, orig_variable_names, output_folder, output_folder_imgs, bbox, generate_images = args
    
    print(f"Processing year {year}")
    
    if year <= 2018:
        print(f"Processing year {year} with old WRF format")
        mode = wrfFileType.old
        # input_folder = '/ServerData/CHACMOOL/Reanalisis/RESPALDO_V4' # Path to ZION
        input_folder = '/unity/f1/ozavala/DATA/AirPollutionData/WRF_Data/RESPALDO_V4'
        result_dates, result_files, result_files_coords, result_paths = read_wrf_old_files_names(
            input_folder, f'{year}-01-01', f'{year + 1}-01-01')
    else:
        print(f"Processing year {year} with new WRF format")
        mode = wrfFileType.new
        # input_folder = '/ServerData/WRF_2017_Kraken' # Path to ZION
        input_folder = '/unity/f1/ozavala/DATA/AirPollutionData/WRF_Data/WRF_2017_Kraken'
        result_dates, result_files, result_paths = read_wrf_files_names(
            input_folder, f'{year}-01-01', f'{year + 1}-01-01')
        result_files_coords = None

    successful_files = 0

    for file_idx in range(len(result_paths)):
        variable_names = orig_variable_names.copy()
        # we need to shift them by 6 hours
        times = range(6, 30)
        
        print(F"================ {result_files[file_idx]} ================================ ")

        # If it is the old reanalisis we need to open two files since it only saves 24 hours per file
        if mode == wrfFileType.old:
            # If it is the first of the year we need to save all the hours 
            if file_idx == 0:
                times = range(30)
            # Check if we have at least 2 files to merge
            if file_idx < len(result_paths) - 1:
                # Open three consecutive files without using dask
                cur_xr_ds = xr.open_mfdataset(
                    result_paths[file_idx:file_idx+2], 
                    decode_times=False,
                    combine='nested',
                    concat_dim='Time',
                    parallel=True
                )
            else:
                times = range(6, 24) # If it is one of the last files
                cur_xr_ds = xr.open_dataset(result_paths[file_idx], decode_times=False)
        else:
            cur_xr_ds = xr.open_dataset(result_paths[file_idx], decode_times=False)

        # If in the 'desired' variables we have RAINC and RAINNC, we need to sum them to get the total rainfall and
        # also make them not cumulative
        if 'RAINC' in variable_names and 'RAINNC' in variable_names:
            print(F"Summing RAINC and RAINNC to get the total rainfall")
            variable_names.append('RAIN')
            # Sum RAINC and RAINNC to get the total rainfall
            cur_xr_ds['RAIN'] = cur_xr_ds['RAINC'] + cur_xr_ds['RAINNC']
            # Make RAIN not cumulative
            rain_values = cur_xr_ds['RAIN'].values
            rain_diff = np.zeros_like(rain_values)
            rain_diff[1:,:,:] = rain_values[1:,:,:] - rain_values[:-1,:,:]
            cur_xr_ds['RAIN'] = xr.DataArray(rain_diff, dims=cur_xr_ds['RAIN'].dims, coords=cur_xr_ds['RAIN'].coords)
            # Set negative values to 0
            cur_xr_ds['RAIN'] = cur_xr_ds['RAIN'].where(cur_xr_ds['RAIN'] > 0, 0)
            # Remove RAINC and RAINNC
            cur_xr_ds = cur_xr_ds.drop_vars(['RAINC', 'RAINNC'])
            variable_names.remove('RAINC')
            variable_names.remove('RAINNC')

        # If U10 and V10 are in the variables, we need to calculate the wind speed
        if 'U10' in variable_names and 'V10' in variable_names:
            print(F"Calculating wind speed from U10 and V10")
            cur_xr_ds['WS10'] = np.sqrt(cur_xr_ds['U10']**2 + cur_xr_ds['V10']**2)
            variable_names.append('WS10')

        if 'RH' in variable_names:
            print(F"Calculating relative humidity from T2 and PSFC")
            # Calculate relative humidity from T2 and PSFC
            T2 = cur_xr_ds['T2'].values  # Temperature in K
            PSFC = cur_xr_ds['PSFC'].values  # Pressure in Pa
            Q2 = cur_xr_ds['Q2'].values  # Mixing ratio in kg/kg
            
            RH = calculate_relative_humidity_metpy(T2, PSFC, Q2)
            
            # Add back to dataset as DataArray
            cur_xr_ds['RH'] = xr.DataArray(RH, dims=cur_xr_ds['T2'].dims, coords=cur_xr_ds['T2'].coords)

        try:
            if mode == wrfFileType.new:
                cropped_xr_ds, newLAT, newLon = crop_variables_xr(cur_xr_ds, variable_names, bbox, times=times)
            elif mode == wrfFileType.old:
                cur_xr_ds_coords = xr.open_dataset(result_files_coords[file_idx])
                LAT = cur_xr_ds_coords.XLAT.values[0,:,0]
                LON = cur_xr_ds_coords.XLONG.values[0,0,:]
                cropped_xr_ds, newLAT, newLon = crop_variables_xr_cca_reanalisis(cur_xr_ds, variable_names, bbox,
                                                                                    times=times, LAT=LAT, LON=LON)

            # Print the dimensions of the cropped dataset
            print(f"Cropped dataset dimensions: {cropped_xr_ds.dims}")

            if generate_images:
                n_timesteps = 3 # Only plot 3 time steps
                plot_variables(cropped_xr_ds, variable_names, output_folder_imgs, result_dates[file_idx], 
                               n_timesteps=n_timesteps, times=times)

            # Save the data as a single day netcdf file
            output_name = F"{output_folder}/{result_dates[file_idx].strftime('%Y-%m-%d')}.nc"
            cropped_xr_ds.to_netcdf(output_name)

            # Clear the memory
            del cropped_xr_ds
            del cur_xr_ds
            del cur_xr_ds_coords
            del LAT
            del LON
            del newLAT
            del newLon

        except Exception as e:
            print(F"ERROR!!!!! Failed to crop file {result_paths[file_idx]}: {e}")
            return False

        successful_files += 1
            
    return year, successful_files, len(result_paths)

def process_files(years, variable_names, output_folder, output_sizes, bbox, generate_images=False, parallel=False):
    """
    Process WRF files for multiple years, cropping them to a specified region and saving as netCDF files.
    
    Args:
        years (range or list): Years to process
        variable_names (list): List of variables to extract from WRF files
        output_folder (str): Path to save output netCDF files
        output_sizes (list): List of dictionaries specifying output dimensions
        bbox (list): Bounding box coordinates [lat_min, lat_max, lon_min, lon_max]
        generate_images (bool, optional): Whether to generate plots of the variables. Defaults to False.
        parallel (bool, optional): Whether to process files in parallel by year. Defaults to False.
    """
    # Create output folders if they don't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder_imgs):
        os.makedirs(output_folder_imgs)

    if parallel:
        # Create arguments list for parallel processing by year
        process_args = [(year, variable_names, output_folder, output_folder_imgs, 
                        bbox, generate_images) for year in years]

        with Pool(processes=min(len(years), 10)) as pool:
            results = pool.map(process_single_year, process_args)
        
        # Print summary of processing
        for year, successful, total in results:
            print(f"Year {year}: Successfully processed {successful} out of {total} files")
    else:
        for year in years:
            year, successful, total = process_single_year(
                (year, variable_names, output_folder, output_folder_imgs, bbox, generate_images))
            print(f"Year {year}: Successfully processed {successful} out of {total} files")

if __name__== '__main__':
    # Reads user configuration

    variable_names = ['T2', 'U10', 'V10', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW', 'RH']
    # output_folder = '/ZION/AirPollutionData/Data/WRF_NetCDF'
    # output_folder_imgs= '/ZION/AirPollutionData/Data/WRF_NetCDF_imgs'

    output_folder = '/unity/f1/ozavala/DATA/AirPollutionData/Preproc/WRF_NetCDF'
    output_folder_imgs= '/unity/f1/ozavala/DATA/AirPollutionData/Preproc/WRF_NetCDF_imgs'

    output_sizes = [{"rows": 100, "cols": 100}]
    sm_bbox = [18.75, 20,-99.75, -98.5]
    large_bbox = [14.568, 21.628, -106.145, -93.1004]
    bbox = sm_bbox
    # years = range(2010, 2025)
    years = [2010]
    generate_images = True
    parallel = False

    process_files(years, variable_names, output_folder, output_sizes, bbox, generate_images, parallel)