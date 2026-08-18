
'''dependencies
todo:
os.chdir(measured_illuminance_path) revert this after operation is done, genssky residues here. 
resolve AIM FRAME Data stuff.
'''


from __future__ import absolute_import, division, print_function
import os 
import subprocess
import pandas as pd
import sys
import json
import importlib.util
import datetime
import numpy as np
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
import pytz
from astropy.coordinates import get_sun, AltAz, EarthLocation # for dn_illuminance , dh_illuminance.
from Luminous_eff_models import irr2ill_P90
import warnings
# import GHIsplit model
# ghi_split_models_path = os.path.join('/tudelft.net/staff-umbrella/AIM FRAME Data/RQ0_UofT_collaboration/', "01_process", "01_weatherdata", "GHI_split_models.py")
# spec = importlib.util.spec_from_file_location("GHIsplit", ghi_split_models_path)
# GHIsplit = importlib.util.module_from_spec(spec)
# sys.modules["GHIsplit"] = GHIsplit
# spec.loader.exec_module(GHIsplit)

from GHI_split_models import Skartveit86

# for Toronto location at november 2023.
LAT = 43.6596179
LON = -79.400956



class site_measurements:
    """ A class for processing the measured data(?) from the on-site measurements
    
    Attributes:
    site_measurements: a dictionary containing the HDR information per room:
    ---------
    site_measurements
    ├── DA230
    │   ├── HDR_info
    │   ├── weather_info 
    │   ├── sensor_info: it is made calibrated (calibrate-v2.py) raw sensor data (e.g. nPC_C.csv)
    │   └── sensor_info_postprocessed  --> applies action spectra
    ├── DA_RA
    ├── DA_200
    ├── DA_230(2)
    ├── DA-Café
    ├── MH_RA
    ├── MH_440
    └── DA_321
    
    predefined inputs:
    sensor_json: a json file containing the paths to the raw sensor data per sensor
    file: a path to the excel file containing the HDR capture times and room names
    weather_path: a path to the weather station data
    
    Methods:
    get_Lark_input: returns DNI_DHI, GHI_Max_solar_radiation, DateTime, Dew_point for Lark Simulation as lists
    """
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        weather_path = os.path.join(self.base_path, "Nov22log.txt")
        sensor_json = os.path.join(self.base_path, "raw_sonsor_illuminance", "sensor_raw_dict.json")
        file = os.path.join(self.base_path, "HDRs_measured_illuminance.xlsx")
        self.sensor_raw_path_dict = json.load(open(sensor_json, 'r'))
        self.file = file # PATH TO THE EXCEL FILE CONTAINING THE HDR CAPTURE TIMES and room names
        self.toronto_tz = pytz.timezone('America/Toronto')
        self.weather = pd.read_csv(weather_path, delimiter=',')
        self.site_measurements = {}
        self.room_names = []
        
        '''process weather station data'''
        date = self.weather['Date (dd/mm/yy)']
        Day_of_year=[]
        for i in range(len(date)):
            test = datetime.datetime.strptime(date[i],'%d-%m-%y')
            Day_of_year.append(test.strftime('%j'))
        self.weather['Day_of_year'] = Day_of_year
        # remove Day_of_year variable from memory
        del Day_of_year
        # 1. calculate Zenith angle
        date_and_time_astropy=[]
        date_and_time=[]
        for i in range(len(date)):
            date_and_time_astropy.append(Time(datetime.datetime.strptime(str(self.weather.iloc[i]['Date (dd/mm/yy)']) + ' ' + str(self.weather.iloc[i]['Time']), '%d-%m-%y %H:%M')))
            utc_time = datetime.datetime.strptime(str(self.weather.iloc[i]['Date (dd/mm/yy)']) + ' ' + str(self.weather.iloc[i]['Time']), '%d-%m-%y %H:%M') 
            toronto_tz = pytz.timezone('America/Toronto') # Create a timezone object for Toronto
            toronto_time = utc_time.replace(tzinfo=pytz.utc).astimezone(toronto_tz)# Convert UTC time to Toronto time
            date_and_time.append(toronto_time)
        self.weather["DateTime_astropy"] = date_and_time_astropy
        self.weather["DateTime_torontoTZ"] = date_and_time
        zenith_angle = []
        solar_altitude = []
        for i in range(len(self.weather)):
            now = self.weather["DateTime_astropy"][i]
            altaz = coord.AltAz(location=coord.EarthLocation(lon=LON * u.deg, lat=LAT * u.deg), obstime=now)
            sun = coord.get_sun(now)
            zenith = sun.transform_to(altaz).zen.degree
            zenith_angle.append(zenith)
            solar_altitude.append(90 - zenith)
        self.weather['zenith'] = zenith_angle
        self.weather['solar_altitude'] = solar_altitude
        
        #'''calculating the DNI, DHI, DNE, DHE'''
        dn = []
        dh = []
        dn_max = []
        dh_max = []
        
        dne = []
        dhe = []
        dne_max = []
        dhe_max = []
        
        for i in range(len(self.weather)):
            _, dni, dhi =         Skartveit86(      ghi     = self.weather['Solar Radiation (W/m^2)'][i], 
                                                    solalt  = self.weather['solar_altitude'][i],
                                                    dn      = int(self.weather['Day_of_year'][i]))
            
            _, dni_max, dhi_max = Skartveit86(      ghi     = self.weather['Max Solar radiation (W/m^2)'][i], 
                                                    solalt  = self.weather['solar_altitude'][i],
                                                    dn      = int(self.weather['Day_of_year'][i]))
            
            ghe_, dne_, dhe_, zlum_ = irr2ill_P90(      ghi     = self.weather['Solar Radiation (W/m^2)'][i],
                                                    dhi     = dhi,
                                                    solalt  = self.weather['solar_altitude'][i])
            
            ghemax, dnemax, dhemax, zlummax = irr2ill_P90(      ghi     = self.weather['Max Solar radiation (W/m^2)'][i],
                                                    dhi     = dhi_max,
                                                    solalt  = self.weather['solar_altitude'][i])
            
            dn.append(dni)
            dh.append(dhi)
            
            dn_max.append(dni_max)
            dh_max.append(dhi_max)
            
            dhe_max.append(dhemax)
            dne_max.append(dnemax)
            
            dhe.append(dhe_)
            dne.append(dne_)
        
        self.weather['dn_Skartveit86'] = dn
        self.weather['dh_Skartveit86'] = dh
        self.weather['dh_max_Skartveit86'] = dh_max
        self.weather['dn_max_Skartveit86'] = dn_max
        self.weather['dhe[lx]'] = dhe
        self.weather['dne[lx]'] = dne
        self.weather['dhe[lx]_max'] = dhe_max
        self.weather['dne[lx]_max'] = dne_max
        self.weather = self.weather.drop(['Date (dd/mm/yy)', 'Time', 'DateTime_astropy'], axis=1)
        
        '''Process excel file'''
        df_from_excel = pd.read_excel(file, sheet_name='HDR_NAME_TIME_ROOM')
        datetime_ = []
        for i in range(len(df_from_excel)):
            date_ttime = datetime.datetime.strptime(str(df_from_excel['endHDRcapturetime'][i]), "%Y-%m-%d %H:%M:%S")
            toronto_tz = pytz.timezone('America/Toronto')
            toronto_datetime = toronto_tz.localize(date_ttime) #only HDR times included(subset of total sensor times )
            datetime_.append(toronto_datetime) #assign timezone. 
        df_from_excel["datetime"] = datetime_
        df_from_excel["datetime"]
        room_names = self.__unique(df_from_excel['Room']) # list containing the room names
        for room_name in room_names:
            df_selected = None
            df_selected = df_from_excel[df_from_excel['Room'] == room_name]
            self.site_measurements[room_name]={}
            self.site_measurements[room_name]['HDR_info'] = df_selected
            start, delta = toronto_tz.localize(min(self.site_measurements[room_name]['HDR_info']['endHDRcapturetime'])),  max(self.site_measurements[room_name]['HDR_info']['endHDRcapturetime']) - min(self.site_measurements[room_name]['HDR_info']['endHDRcapturetime'])
            start_weather = self.__find_closest(start, self.weather['DateTime_torontoTZ'].to_list()) #stores the timestamp closest to the START of HDR/EML measurements. 
            end_weather = self.__find_closest(start_weather + delta, self.weather['DateTime_torontoTZ'].to_list())#stores the timestamp closest to the END of HDR/EML measurements. 
            self.site_measurements[room_name]['weather_info'] = self.weather.iloc[self.weather[self.weather['DateTime_torontoTZ']==start_weather].index.values[0] : self.weather[self.weather['DateTime_torontoTZ']==end_weather].index.values[0], :] # A pandas dataframe containing only the dataframe for one room

        ################################################ temp: sensor calibration and post-processing ##################################################
        # Define paths
        calibrated_csv_sensor_df_dict = {}
        sensor_df_dict_2CIE = {}
        for room in room_names:
                     self.site_measurements[room]['sensor_info']= {} 
                     self.site_measurements[room]['sensor_info_postprocessed']= {}
        for key in self.sensor_raw_path_dict: # for each sensor
            try:
                input = os.path.join(self.base_path, "raw_sonsor_illuminance", self.sensor_raw_path_dict[key]) # e.g. nPC_C.csv
                # print if input exists
                script_path = os.path.join(self.base_path, "01_Measured_Illuminance", "calibrate-v2.py") # Spectral irradiance in μW/cm²/nm?
                measured_illuminance_path = os.path.join(self.base_path, "01_Measured_Illuminance") 
                # Set the current working directory to measured_illuminance_path
                os.chdir(measured_illuminance_path)
                # 1. _calibrated.csv | Calibrate / write to csv / read calibrated csv (TEMP1)
                calibrated_path = rf"{input[:-4]}_calibrated.csv"
                subprocess.run(["python", script_path, input, calibrated_path]) # this makes _calibrated.csv
                df_calibrated = pd.read_csv(calibrated_path) # read the _calibrated.csv
                # 2. _calibrated_above780Removed.csv | calibrated -> Remove above 780nm -> Write to CSV (TEMP2)
                metadata_columns = ['date', 'time', 'manual', 'int_time', 'frame_avg', 'ae', 'is_saturated', 'is_dark', 'X.original', 'Y.original', 'Z.original']
                df_calibrated_above780_removed = df_calibrated[metadata_columns + [col for col in df_calibrated.columns if col.isdigit() and int(col) <= 780]]
                df_calibrated_above780_removed.to_csv(rf"{input[:-4]}_calibrated_above780Removed.csv", index=False)
                # 3. _calibrated_above780Removed_workplane_cie.csv | 
                subprocess.run(["python", "nPC_to_cie_new_Fversion_Nima.py", '-i', rf"{input[:-4]}_calibrated_above780Removed.csv", '-o', rf"{input[:-4]}_calibrated_above780Removed"]) # this makes <sensor>_calibrated_above780Removed_workplane_cie.csv
                os.remove(rf"{input[:-4]}_calibrated_above780Removed.csv") # remove TEMP2 
                df_calibrated_above780_removed_workplane_cie = pd.read_csv(rf"{input[:-4]}_calibrated_above780Removed_workplane_cie.csv") 
                os.remove(rf"{input[:-4]}_calibrated_above780Removed_workplane_cie.csv") # remove TEMP3
                # ---------------------------------------------------------------> TEMP block 1 replacement
                li = []
                datetime_str = df_calibrated['date'] + ' ' + df_calibrated['time']
                datetime_objs = [datetime.datetime.strptime(dt, '%d/%m/%Y %H:%M:%S') for dt in datetime_str]
                df_calibrated['DateTime'] = pd.to_datetime(datetime_objs) #convert date and time columns to datetime format
                df_calibrated['DateTime'] = df_calibrated['DateTime'] + pd.DateOffset(months=3)# add three months to the datetime column !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                toronto_tz = pytz.timezone('America/Toronto') # set timezone to Toronto
                df_calibrated['DateTime'] = df_calibrated['DateTime'].dt.tz_localize(toronto_tz)
                df_calibrated['Toronto_Time'] = df_calibrated['DateTime'] # assign the Toronto time to a new column called "Toronto_Time"
                cols = list(df_calibrated.columns) # move the "Toronto_Time" column to the first column of the dataframe
                cols = [cols[-1]] + cols[:-1] 
                df_calibrated = df_calibrated[cols]
                li.append(df_calibrated)
                calibrated_csv_sensor_df_dict[key] = pd.concat(li, axis=0, ignore_index=True)
                # ---------------------------------------------------------------> TEMP block 1 replacement
                # ---------------------------------------------------------------> TEMP block 3 replacement

                li = [] # A list to store the dataframes for each sensor
                datetime_str = df_calibrated_above780_removed_workplane_cie['Date'] + ' ' + df_calibrated_above780_removed_workplane_cie['Time']
                datetime_objs = [datetime.datetime.strptime(dt, '%d/%m/%Y %H:%M:%S') for dt in datetime_str]
                df_calibrated_above780_removed_workplane_cie['DateTime'] = pd.to_datetime(datetime_objs) # Convert date and time columns to datetime format
                df_calibrated_above780_removed_workplane_cie['DateTime'] = df_calibrated_above780_removed_workplane_cie['DateTime'] + pd.DateOffset(months=3) # Add three months to the datetime column
                toronto_tz = pytz.timezone('America/Toronto') # Set timezone to Toronto
                df_calibrated_above780_removed_workplane_cie['DateTime'] = df_calibrated_above780_removed_workplane_cie['DateTime'].dt.tz_localize(toronto_tz)
                df_calibrated_above780_removed_workplane_cie['Toronto_Time'] = df_calibrated_above780_removed_workplane_cie['DateTime'] # Assign the Toronto time to a new column called "Toronto_Time"
                cols = list(df_calibrated_above780_removed_workplane_cie.columns) 
                cols = [cols[-1]] + cols[:-1] # Move the "Toronto_Time" column to the first column of the dataframe
                df_calibrated_above780_removed_workplane_cie = df_calibrated_above780_removed_workplane_cie[cols] 
                li.append(df_calibrated_above780_removed_workplane_cie)
                sensor_df_dict_2CIE[key] = pd.concat(li, axis=0, ignore_index=True) # A pandas dataframe containing only the dataframe for one room
                # ---------------------------------------------------------------> TEMP block 3 replacement
                # ---------------------------------------------------------------> TEMP block 2 replacement
                for i in room_names:
                    start, delta = toronto_tz.localize(min(self.site_measurements[i]['HDR_info']['endHDRcapturetime'])),  max(self.site_measurements[i]['HDR_info']['endHDRcapturetime']) - min(self.site_measurements[i]['HDR_info']['endHDRcapturetime'])
                    start_sensorData = self.__find_closest(start, calibrated_csv_sensor_df_dict[key]['Toronto_Time'].to_list()) #stores the timestamp closest to the START of HDR/EML measurements. 
                    end_sensorData = self.__find_closest(start_sensorData + delta, calibrated_csv_sensor_df_dict[key]['Toronto_Time'].to_list())#stores the timestamp closest to the END of HDR/EML measurements.
                    start_weather = self.__find_closest(start, self.weather['DateTime_torontoTZ'].to_list()) #stores the timestamp closest to the START of HDR/EML measurements.        
                    end_weather = self.__find_closest(start_weather + delta, self.weather['DateTime_torontoTZ'].to_list())#stores the timestamp closest to the END of HDR/EML measurements. 
                    self.site_measurements[i]['sensor_info'][key] = calibrated_csv_sensor_df_dict[key].iloc[calibrated_csv_sensor_df_dict[key][calibrated_csv_sensor_df_dict[key]['Toronto_Time']==start_sensorData].index.values[0] : calibrated_csv_sensor_df_dict[key][calibrated_csv_sensor_df_dict[key]['Toronto_Time']==end_sensorData].index.values[0], :] # A pandas dataframe containing only the dataframe for one room
                    self.site_measurements[i]['sensor_info_postprocessed'][key] = sensor_df_dict_2CIE[key].iloc[sensor_df_dict_2CIE[key][sensor_df_dict_2CIE[key]['Toronto_Time']==start_sensorData].index.values[0] : sensor_df_dict_2CIE[key][sensor_df_dict_2CIE[key]['Toronto_Time']==end_sensorData].index.values[0], :] # A pandas dataframe containing only the dataframe for one room
                    for timeStamp in self.site_measurements[i]['weather_info']['DateTime_torontoTZ']: # 
                        array_to_search = self.site_measurements[i]['sensor_info'][key]["Toronto_Time"]
                        data_to_find = timeStamp
                    try:
                        df_calibrated = df_calibrated.append(self.site_measurements[i]['sensor_info'][key][self.site_measurements[i]['sensor_info'][key]["Toronto_Time"] == self.__find_closest(data_to_find, array_to_search)])
                        warnings.simplefilter('ignore', FutureWarning)
                        df_calibrated_above780_removed_workplane_cie = df_calibrated_above780_removed_workplane_cie.append(self.site_measurements[i]['sensor_info_postprocessed'][key][self.site_measurements[i]['sensor_info_postprocessed'][key]["Toronto_Time"] == self.__find_closest(data_to_find, array_to_search)])
                        warnings.simplefilter('ignore', FutureWarning)
                    except:
                        warnings.simplefilter('ignore', FutureWarning)
                    self.site_measurements[i]['sensor_info'][key] = df_calibrated
                    self.site_measurements[i]['sensor_info_postprocessed'][key] = df_calibrated_above780_removed_workplane_cie
                print(f"Calibrated sensor data for {key} has been processed")
            except pd.errors.EmptyDataError:
                print(f"EmptyDataError: The sensor data for {key} is empty, passed")
                pass
            except KeyError:
                print(f"KeyError: The sensor data for {key} is empty, passed")
                pass
            os.remove(calibrated_path) # remove TEMP1
    @classmethod
    def __unique(self,list1):
        unique_list = [] # initialize a null list
        for x in list1: # traverse for all elements
            if x not in unique_list: # check if exists in unique_list or not
                unique_list.append(x)
        return unique_list
    def __find_closest(self,dt, list_of_dt): 
        '''finds the closest date in the weather dataframe to a timestamp'''
        return min(list_of_dt, key=lambda x: abs(x - dt))
    def get_Lark_input(self, room_name, prefix = None):
        '''returns the site_measurements for Lark Simulation'''
        DNI_DHI = []
        DateTime = []
        dne_dhe = []
        GHI_Max_solar_radiation = self.site_measurements[room_name]['weather_info']['Max Solar radiation (W/m^2)'].tolist()
        self.site_measurements[room_name]['weather_info']['dh_max_Skartveit86'].values
        for i in range(len(self.site_measurements[room_name]['weather_info']['dh_max_Skartveit86'])):
            DNI_DHI.append(str(round(self.site_measurements[room_name]['weather_info']['dn_max_Skartveit86'].values[i],2))+"_"+str(round(self.site_measurements[room_name]['weather_info']['dh_max_Skartveit86'].values[i],2)))
            dne_dhe.append(str(round(self.site_measurements[room_name]['weather_info']['dne[lx]_max'].values[i],2))+"_"+str(round(self.site_measurements[room_name]['weather_info']['dhe[lx]_max'].values[i],2)))
        self.site_measurements[room_name]['weather_info']['dh_max_Skartveit86']
        for i in list(self.site_measurements[room_name]['weather_info']['DateTime_torontoTZ']): 
            DateTime.append(str(i.month)+"_"+str(i.day)+"_"+str(round(i.hour + i.minute/60, 2)))
        Dew_point = list(self.site_measurements[room_name]['weather_info']['Dew point (C)'].values)
        if prefix == None: # write to json file
            return DNI_DHI, GHI_Max_solar_radiation, DateTime, Dew_point, dne_dhe
        else: # write the {room_name}_Lark_input.json file to the prefix folder
            writing_path = os.path.join(prefix, f"{room_name}_Lark_input.json")
            with open(writing_path, 'w') as f:
                json.dump({"DNI_DHI": DNI_DHI, "GHI_Max_solar_radiation": GHI_Max_solar_radiation, "DateTime": DateTime, "Dew_point": Dew_point}, f)
