from enum import Enum

class constants(Enum):
    index_label = "fecha"
    wrf_output_folder = "WRF_CSV"
    wrf_each_quadrant_name = "WRF_CSV"
    db_output_folder = "DataPollutionDB_CSV"
    training_output_folder = "Training"
    merge_output_folder = "MergedDataCSV"
    date_format = '%Y-%m-%d'
    datetime_format = '%Y-%m-%d %H:%M:%S'


class metricsNames(Enum):
    mse = 1

class wrfFileType(Enum):
    old = 0
    new = 1
