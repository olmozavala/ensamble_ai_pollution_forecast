from conf.MakeWRF_and_DB_CSV_UserConfiguration import getPreprocWRFParams
from conf.params import PreprocParams
from proj_preproc.wrf import crop_variables_xr, crop_variables_xr_cca_reanalisis, subsampleData
from proj_preproc.utils import getStringDates
from conf.localConstants import constants
from conf.localConstants import wrfFileType
import os
from os.path import join
import xarray as xr
from multiprocessing import Pool
from utils.plotting import plot_variables
from io_netcdf.inout import read_wrf_files_names, read_wrf_old_files_names, saveFlattenedVariables

def process_single_file(args):
    file_idx, result_paths, result_files, result_files_coords, result_dates, mode, variable_names, bbox, generate_images, output_folder, output_folder_imgs = args

    times = range(24) # Only saving 24 hours per file
    
    print(F"================ {result_files[file_idx]} ================================ ")
    # Read file as xarray
    cur_xr_ds = xr.open_dataset(result_paths[file_idx], decode_times=False)
    print(F"\tCropping...")
    
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
            plot_variables(cropped_xr_ds, variable_names, output_folder_imgs, result_dates[file_idx])

        # Save the data as a single day netcdf file
        output_name = F"{output_folder}/{result_dates[file_idx].strftime('%Y-%m-%d')}.nc"
        cropped_xr_ds.to_netcdf(output_name)
        return True

    except Exception as e:
        print(F"ERROR!!!!! Failed to crop file {result_paths[file_idx]}: {e}")
        return False

def process_files(years, variable_names, output_folder, output_sizes, bbox, generate_images=False, parallel=False):
    """
    It will process the corresponding WRF files and save the outputs by year. Depending on the years requested
    is the folder to be used and the files to be read.
    """

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder_imgs):
        os.makedirs(output_folder_imgs)

    for year in years:
        if year <= 2018:
            mode = wrfFileType.old
            input_folder = '/ServerData/CHACMOOL/Reanalisis/RESPALDO_V4/'
            result_dates, result_files, result_files_coords, result_paths = read_wrf_old_files_names(input_folder, F'{year}-01-01', F'{year + 1}-01-01')
        else:
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
                break


if __name__== '__main__':
    # Reads user configuration

    variable_names = ['T2', 'U10', 'V10', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW']
    output_folder = '/ZION/AirPollutionData/Data/WRF_NetCDF'
    output_folder_imgs= '/ZION/AirPollutionData/Data/WRF_NetCDF_imgs'
    output_sizes = [{"rows": 100, "cols": 100}]
    sm_bbox = [18.75, 20,-99.75, -98.5]
    large_bbox = [14.568, 21.628, -106.145, -93.1004]
    bbox = sm_bbox
    years = range(2010, 2025)
    generate_images = True
    parallel = True

    process_files(years, variable_names, output_folder, output_sizes, bbox, generate_images, parallel)

    # # The max range is from 1980 to present
    # start_year = 2010
    # end_year = 2011

    # # Run this process in parallel splitting separating by years
    # NUMBER_PROC = 20
    # p = Pool(NUMBER_PROC)
    # p.map(runParallel, range(start_year, end_year))


# dataset 
# parametros
# horas previas (int) Cuantas horas previas se van a usar
# horas previas meteorologia (int) Cuantas horas previas de meteorologia se van a usar
# horas posteriores (int) Cuantas horas posteriores se van a usar en el autoregresivos
# contaminantes (list of strings) Lista de contaminantes a usar ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
# meteorologia (list of strings) Lista de meteorologia a usar ['T2', 'RH2', 'WS10', 'WD10', 'PSFC', 'U10', 'V10']
# otras_varialbes (list of strings) ['meteorogila_estacion', 'incendios',...]
# perc_noise (float) Porcentaje de ruido a agregar a los datos
#
# Salidas del dataset
# contaminacion [horas_previas, estaciones, contaminantes] (original esta flatten)
# cyclic time [horas_previas, 12] (sin(year), cos(year), sin(day), cos(day), sin(week), cos(week), 
#        half_sin(year), half_cos(year), half_sin(week), half_cos(week), half_sin(day), half_cos(day))
# meteorologia [horas_previas_meteorologia + horas_posteriores, variables, width, height]   
# Y [estaciones, contaminantes]
# Y_flags [estaciones, contaminantes]

# ---- ACTUAL SIN METEOROLOGIA ----
# INPUT
# array [horas_previas * (estaciones * contaminantes + ciclic_time) ]
# OUTPUT
# array [estaciones*contaminantes]

# ---- CON METEOROLOGIA ----
# INPUT
# [[horas_previas * (estaciones * contaminantes + ciclic_time) ], (meteorologia)[horas previas meteorologia, variables, width, height]]
# OUTPUT
# array [estaciones*contaminantes]
