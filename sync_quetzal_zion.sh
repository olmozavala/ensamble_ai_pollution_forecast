# rsync -avz --progress /home/olmozavala/DATA/AirPollution/WRF_NetCDF/ pedro@132.248.8.98:/FOLDER/ -e "ssh -p 22722"
# rsync -avz --progress /home/olmozavala/DATA/AirPollution/WRF_NetCDF_imgs/ pedro@132.248.8.98:/FOLDER/ -e "ssh -p 22722"

rsync -avz --progress olmozavala@132.248.8.98:/ZION/AirPollutionData/Data/MergedDataCSV/16/Imputed/bk1/ /home/olmozavala/DATA/AirPollution/PollutionCSV/ -e "ssh -p 22722"
