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
import dask
import pandas as pd

def process_single_file(args):
    """
    Process a single WRF file.
    
    Args:
        args (tuple): Contains:
            - file_path (str): Path to the WRF file
            - file_idx (int): Index of the file
            - mode (wrfFileType): Type of WRF file (old or new)
            - orig_variable_names (list): Variables to extract
            - output_folder (str): Path for output netCDF files
            - output_folder_imgs (str): Path for output images
            - bbox (list): Bounding box coordinates
            - generate_images (bool): Whether to generate plots
            - result_dates (list): List of dates
            - result_files_coords (list): List of coordinate files (for old format)
    """
    (file_path, file_idx, mode, orig_variable_names, output_folder, output_folder_imgs, 
     bbox, generate_images, result_dates, result_files_coords, resolution) = args
    
    times = range(24)
    try:
        print(f"Processing file {file_path}....")
        # Load single file with proper time decoding
        cur_xr_ds = xr.open_dataset(file_path, decode_times=False)

        
        variable_names = orig_variable_names.copy()

        # If in the 'desired' variables we have RAINC and RAINNC, we need to sum them
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

        # If U10 and V10 are in the variables, calculate wind speed
        if 'U10' in variable_names and 'V10' in variable_names:
            print("Calculating wind speed from U10 and V10")
            cur_xr_ds['WS10'] = np.sqrt(cur_xr_ds['U10']**2 + cur_xr_ds['V10']**2)
            variable_names.append('WS10')

        if 'RH' in variable_names:
            print("Calculating relative humidity from T2 and PSFC")
            T2 = cur_xr_ds['T2'].values
            PSFC = cur_xr_ds['PSFC'].values
            Q2 = cur_xr_ds['Q2'].values
            
            RH = calculate_relative_humidity_metpy(T2, PSFC, Q2)
            cur_xr_ds['RH'] = xr.DataArray(RH, dims=cur_xr_ds['T2'].dims, 
                                         coords=cur_xr_ds['T2'].coords)

        # Crop the dataset
        if mode == wrfFileType.new:
            cropped_xr_ds, newLAT, newLon = crop_variables_xr(cur_xr_ds, variable_names, 
                                                             bbox, times=times)
        else:
            cur_xr_ds_coords = xr.open_dataset(result_files_coords[file_idx])
            LAT = cur_xr_ds_coords.XLAT.values[0,:,0]
            LON = cur_xr_ds_coords.XLONG.values[0,0,:]
            cropped_xr_ds, newLAT, newLon = crop_variables_xr_cca_reanalisis(
                cur_xr_ds, variable_names, bbox, times=times, LAT=LAT, LON=LON)

        
        # Since the first hour is GMT 0, we need to add the time zone by subtracting 6 hours    
        first_datetime = result_dates[file_idx].replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(hours=6)
        # Update the attributes for time, lat, and lon to follow CF conventions
        cropped_xr_ds['time'].attrs.update({
            'units': f'hours since {first_datetime.strftime("%Y-%m-%d %H:%M:%S")}',
            'calendar': 'standard',
            'axis': 'T',
            'long_name': 'time',
            'standard_name': 'time'
        })
        cropped_xr_ds['lat'].attrs.update({
            'units': 'degrees_north',
            'axis': 'Y',
            'long_name': 'latitude',
            'standard_name': 'latitude'
        })  
        cropped_xr_ds['lon'].attrs.update({
            'units': 'degrees_east',
            'axis': 'X',
            'long_name': 'longitude',
            'standard_name': 'longitude'
        })

        # Instead of resampling, create new coordinates at the desired resolution
        new_lat = np.arange(bbox[0], bbox[1], resolution)  # From lat_min to lat_max
        new_lon = np.arange(bbox[2], bbox[3], resolution)  # From lon_min to lon_max
        
        # Interpolate the dataset to the new coordinates
        cropped_xr_ds = cropped_xr_ds.interp(
            lat=new_lat,
            lon=new_lon,
            method='linear'  
        )

        # Save the data
        cur_date = result_dates[file_idx]
        output_name = f"{output_folder}/{cur_date.strftime('%Y-%m-%d')}.nc"
        cropped_xr_ds.to_netcdf(output_name)

        if generate_images:
            plot_variables(cropped_xr_ds, variable_names, output_folder_imgs, n_timesteps=3)

        # Clean up
        cur_xr_ds.close()
        if mode == wrfFileType.old:
            cur_xr_ds_coords.close()
        
        return True, file_path
        
    except Exception as e:
        print(f"Failed to process file {file_path}: {str(e)}")
        return False, file_path

    print(f"Finished processing file {file_path}")

def process_single_year(args):
    """
    Process all WRF files for a single year using parallel processing.
    """
    year, orig_variable_names, output_folder, output_folder_imgs, bbox, generate_images, resolution = args
    
    print(f"Processing year {year}")
    
    if year <= 2018:
        print(f"Processing year {year} with old WRF format")
        mode = wrfFileType.old
        # input_folder = '/unity/f1/ozavala/DATA/AirPollutionData/WRF_Data/RESPALDO_V4' # Path in Skynet
        # input_folder = '/ServerData/CHACMOOL/Reanalisis/RESPALDO_V4' # Path to ZION
        input_folder = '/CHACMOOL/DATOS/RESPALDO_V4' # Path in Quetzal
        result_dates, result_files, result_files_coords, result_paths = read_wrf_old_files_names(
            input_folder, f'{year}-01-01', f'{year + 1}-01-01')
    else:
        print(f"Processing year {year} with new WRF format")
        mode = wrfFileType.new
        # input_folder = '/ServerData/WRF_2017_Kraken' # Path to ZION
        # input_folder = '/unity/f1/ozavala/DATA/AirPollutionData/WRF_Data/WRF_2017_Kraken'
        input_folder = '/LUSTRE/OPERATIVO/EXTERNO-salidas/WRF' # Path in Quetzal
        result_dates, result_files, result_paths = read_wrf_files_names(
            input_folder, f'{year}-01-01', f'{year + 1}-01-01')
        result_files_coords = None

    # Print some debugging information
    print(f"Number of files found: {len(result_paths)}")
    print(f"First few files just to check:")
    for path in result_paths[:3]:
        print(f"  {path}")
        try:
            with xr.open_dataset(path, decode_times=False) as ds:
                print(f"{path} - Successfully opened. ")
        except Exception as e:
            print(f"    Failed to open: {str(e)}")

    # Sort the result_paths by name
    result_paths = sorted(result_paths, key=lambda x: x.split('/')[-1])
    
    # Create arguments for parallel processing
    process_args = [(file_path, idx, mode, orig_variable_names, output_folder, 
                    output_folder_imgs, bbox, generate_images, result_dates, 
                    result_files_coords, resolution) 
                   for idx, file_path in enumerate(result_paths)]

    # Process files sequentially
    results = [process_single_file(args) for args in process_args]
    
    # Count successful files
    successful_files = sum(1 for success, _ in results if success)
    
    return year, successful_files, len(result_paths)

def process_files(years, variable_names, output_folder, output_folder_imgs, resolution, bbox, generate_images=False, parallel=False):
    """
    Process WRF files for multiple years, cropping them to a specified region and saving as netCDF files.
    
    Args:
        years (range or list): Years to process
        variable_names (list): List of variables to extract from WRF files
        output_folder (str): Path to save output netCDF files
        resolution (float): Resolution of the output netCDF files
        bbox (list): Bounding box coordinates [lat_min, lat_max, lon_min, lon_max]
        generate_images (bool, optional): Whether to generate plots of the variables. Defaults to False.
        parallel (bool, optional): Whether to process files in parallel by year. Defaults to False.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder_imgs):
        os.makedirs(output_folder_imgs)

    if parallel:
        # Create arguments list for parallel processing by year
        process_args = [(year, variable_names, output_folder, output_folder_imgs, 
                        bbox, generate_images, resolution) for year in years]

        with Pool(processes=min(len(years), 10)) as pool:
            results = pool.map(process_single_year, process_args)
        
        # Print summary of processing
        for year, successful, total in results:
            print(f"Year {year}: Successfully processed {successful} out of {total} files")
    else:
        for year in years:
            year, successful, total = process_single_year(
                (year, variable_names, output_folder, output_folder_imgs, 
                 bbox, generate_images, resolution))  # Added resolution here
            print(f"Year {year}: Successfully processed {successful} out of {total} files")

if __name__== '__main__':
    # Reads user configuration

    variable_names = ['T2', 'U10', 'V10', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW', 'RH']
    # output_folder = '/ZION/AirPollutionData/Data/WRF_NetCDF'
    # output_folder_imgs= '/ZION/AirPollutionData/Data/WRF_NetCDF_imgs'

    # output_folder = '/unity/f1/ozavala/DATA/AirPollutionData/Preproc/WRF_NetCDF'
    # output_folder_imgs= '/unity/f1/ozavala/DATA/AirPollutionData/Preproc/WRF_NetCDF_imgs'

    output_folder = '/home/olmozavala/DATA/AirPollution/WRF_NetCDF'
    output_folder_imgs= '/home/olmozavala/DATA/AirPollution/WRF_NetCDF_imgs'

    resolution = 1/20 # degrees
    sm_bbox = [18.75, 20,-99.75, -98.5]
    large_bbox = [14.568, 21.628, -106.145, -93.1004]
    bbox = sm_bbox
    years = range(2010, 2025)
    # years = [2021]
    generate_images = True
    parallel = False

    process_files(years, variable_names, output_folder, output_folder_imgs, resolution, bbox, generate_images, parallel)