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

def process_single_file(args):
    """
    Process a single WRF file by cropping it to the specified region and saving it as a netCDF file.
    
    Args:
        args (tuple): Contains the following parameters:
            - file_idx (int): Index of the file to process
            - result_paths (list): List of full file paths
            - result_files (list): List of file names
            - result_files_coords (list): List of coordinate files (for old WRF format)
            - result_dates (list): List of dates corresponding to each file
            - mode (wrfFileType): Indicates if files are in new or old WRF format
            - variable_names (list): List of variables to extract
            - bbox (list): Bounding box coordinates [lat_min, lat_max, lon_min, lon_max]
            - generate_images (bool): Whether to generate plots of the variables
            - output_folder (str): Path to save output netCDF files
            - output_folder_imgs (str): Path to save output images
    
    Returns:
        bool: True if processing successful, False otherwise
    """
    file_idx, result_paths, result_files, result_files_coords, result_dates, mode, variable_names, bbox, generate_images, output_folder, output_folder_imgs = args

    # These are the times we want to keep. But, since the WRF files are 6 hours apart, 
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
                combine='by_coords',
                parallel=True# Don't use parallel processing
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
        # cur_xr_ds = cur_xr_ds.drop_vars(['RAINC', 'RAINNC'])

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
            plot_variables(cropped_xr_ds, variable_names, output_folder_imgs, result_dates[file_idx], n_timesteps=n_timesteps)

        # Save the data as a single day netcdf file
        output_name = F"{output_folder}/{result_dates[file_idx].strftime('%Y-%m-%d')}.nc"
        cropped_xr_ds.to_netcdf(output_name)
        return True

    except Exception as e:
        print(F"ERROR!!!!! Failed to crop file {result_paths[file_idx]}: {e}")
        return False

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
        parallel (bool, optional): Whether to process files in parallel. Defaults to False.
    
    Notes:
        - For years <= 2018, uses old WRF format from CHACMOOL/Reanalisis
        - For years > 2018, uses new WRF format from WRF_2017_Kraken
        - When parallel=True, uses 10 processes to speed up processing
    """

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder_imgs):
        os.makedirs(output_folder_imgs)

    for year in years:
        if year <= 2018:
            print(F"Processing year {year} with old WRF format")
            mode = wrfFileType.old
            input_folder = '/ServerData/CHACMOOL/Reanalisis/RESPALDO_V4/'
            result_dates, result_files, result_files_coords, result_paths = read_wrf_old_files_names(input_folder, F'{year}-01-01', F'{year + 1}-01-01')
        else:
            print(F"Processing year {year} with new WRF format")
            mode = wrfFileType.new
            input_folder = '/ServerData/WRF_2017_Kraken/'
            result_dates, result_files, result_paths = read_wrf_files_names(input_folder, F'{year}-01-01', F'{year + 1}-01-01')
            result_files_coords = None

        # Itereate over each file and preprocess them
        print("Processing files...")

        # Process files in parallel using Pool
        if parallel:
            # Create arguments list for parallel processing
            process_args = [(i, result_paths, result_files, result_files_coords, result_dates, 
                            mode, variable_names, bbox, generate_images, output_folder, 
                            output_folder_imgs) for i in range(len(result_paths))]

            with Pool(processes=10) as pool:
                results = pool.map(process_single_file, process_args)
            
            print(f"Successfully processed {sum(results)} out of {len(results)} files")

        elif not parallel:
            for i in range(len(result_paths)):
                args = (i, result_paths, result_files, result_files_coords, result_dates, 
                        mode, variable_names, bbox, generate_images, output_folder, 
                        output_folder_imgs)
                process_single_file(args)
                # Break if i > 5
                if i > 5:
                    break

if __name__== '__main__':
    # Reads user configuration

    variable_names = ['T2', 'U10', 'V10', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW', 'RH']
    output_folder = '/ZION/AirPollutionData/Data/WRF_NetCDF'
    output_folder_imgs= '/ZION/AirPollutionData/Data/WRF_NetCDF_imgs'
    output_sizes = [{"rows": 100, "cols": 100}]
    sm_bbox = [18.75, 20,-99.75, -98.5]
    large_bbox = [14.568, 21.628, -106.145, -93.1004]
    bbox = sm_bbox
    # years = range(2010, 2025)
    years = [2010]
    generate_images = True
    parallel = False

    process_files(years, variable_names, output_folder, output_sizes, bbox, generate_images, parallel)