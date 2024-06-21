from utilities import load_object, save_object, preprocess_long_lat, scale_long_lat, process_time
import os
import numpy as np
import pandas as pd

all_subdirs = os.listdir() 
 
def read_heading(nf1, nf2): 
    all_mine = dict()
    for subdir_name in all_subdirs: 
        if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
            continue
        
        all_files = os.listdir(subdir_name + "/cleaned_csv/") 
        bad_rides_filenames = set()
        if os.path.isfile(subdir_name + "/bad_rides_filenames"):
            bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
        gap_rides_filenames = set()
        if os.path.isfile(subdir_name + "/gap_rides_filenames"):
            gap_rides_filenames = load_object(subdir_name + "/gap_rides_filenames")
        val_rides = set()
        if os.path.isfile(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1)):
            val_rides = load_object(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1))
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file not in val_rides: 
                continue
        
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
            directions = list(file_with_ride["fields_direction"]) 
            direction_int = [np.round(direction, 0) for direction in directions]
            all_mine[subdir_name + "/cleaned_csv/" + some_file] = direction_int
    
    if not os.path.isdir("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
        os.makedirs("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")
    
    save_object("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_val_direction", all_mine)
 
def read_latitude_no_abs(nf1, nf2): 
    all_mine = dict()
    for subdir_name in all_subdirs: 
        if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
            continue
        
        all_files = os.listdir(subdir_name + "/cleaned_csv/") 
        bad_rides_filenames = set()
        if os.path.isfile(subdir_name + "/bad_rides_filenames"):
            bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
        gap_rides_filenames = set()
        if os.path.isfile(subdir_name + "/gap_rides_filenames"):
            gap_rides_filenames = load_object(subdir_name + "/gap_rides_filenames")
        val_rides = set()
        if os.path.isfile(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1)):
            val_rides = load_object(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1))
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file not in val_rides: 
                continue
        
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
            longitudes = list(file_with_ride["fields_longitude"]) 
            latitudes = list(file_with_ride["fields_latitude"]) 
            longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
            longitudes, latitudes = scale_long_lat(longitudes, latitudes, 0.1, 0.1, True)
            latitude_int = [np.round(latitudes[latitude_index + 1] - latitudes[latitude_index], 10) for latitude_index in range(len(latitudes) - 1)]
            all_mine[subdir_name + "/cleaned_csv/" + some_file] = latitude_int
   
    if not os.path.isdir("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
        os.makedirs("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")
    
    save_object("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_val_latitude_no_abs", all_mine)
  
def read_longitude_no_abs(nf1, nf2): 
    all_mine = dict()
    for subdir_name in all_subdirs: 
        if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
            continue
        
        all_files = os.listdir(subdir_name + "/cleaned_csv/") 
        bad_rides_filenames = set()
        if os.path.isfile(subdir_name + "/bad_rides_filenames"):
            bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
        gap_rides_filenames = set()
        if os.path.isfile(subdir_name + "/gap_rides_filenames"):
            gap_rides_filenames = load_object(subdir_name + "/gap_rides_filenames")
        val_rides = set()
        if os.path.isfile(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1)):
            val_rides = load_object(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1))
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file not in val_rides: 
                continue
        
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
            longitudes = list(file_with_ride["fields_longitude"]) 
            latitudes = list(file_with_ride["fields_latitude"]) 
            longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
            longitudes, latitudes = scale_long_lat(longitudes, latitudes, 0.1, 0.1, True)
            longitude_int = [np.round(longitudes[longitude_index + 1] - longitudes[longitude_index], 10) for longitude_index in range(len(longitudes) - 1)]
            all_mine[subdir_name + "/cleaned_csv/" + some_file] = longitude_int
    
    if not os.path.isdir("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
        os.makedirs("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")
    
    save_object("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_val_longitude_no_abs", all_mine)
  
def read_speed(nf1, nf2): 
    all_mine = dict()
    for subdir_name in all_subdirs: 
        if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
            continue
        
        all_files = os.listdir(subdir_name + "/cleaned_csv/") 
        bad_rides_filenames = set()
        if os.path.isfile(subdir_name + "/bad_rides_filenames"):
            bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
        gap_rides_filenames = set()
        if os.path.isfile(subdir_name + "/gap_rides_filenames"):
            gap_rides_filenames = load_object(subdir_name + "/gap_rides_filenames")
        val_rides = set()
        if os.path.isfile(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1)):
            val_rides = load_object(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1))
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file not in val_rides: 
                continue
        
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
            speeds = list(file_with_ride["fields_speed"]) 
            speed_int = [np.round(speed, 0) for speed in speeds] 
            all_mine[subdir_name + "/cleaned_csv/" + some_file] = speed_int
    
    if not os.path.isdir("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
        os.makedirs("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")
    
    save_object("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_val_speed", all_mine)

def read_time(nf1, nf2): 
    all_mine = dict()
    for subdir_name in all_subdirs: 
        if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
            continue
        
        all_files = os.listdir(subdir_name + "/cleaned_csv/") 
        bad_rides_filenames = set()
        if os.path.isfile(subdir_name + "/bad_rides_filenames"):
            bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
        gap_rides_filenames = set()
        if os.path.isfile(subdir_name + "/gap_rides_filenames"):
            gap_rides_filenames = load_object(subdir_name + "/gap_rides_filenames")
        val_rides = set()
        if os.path.isfile(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1)):
            val_rides = load_object(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1))
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file not in val_rides: 
                continue
        
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
            times = list(file_with_ride["time"])
            times_processed = [process_time(time_new) for time_new in times] 
            time_int = [np.round(times_processed[time_index + 1] - times_processed[time_index], 3) for time_index in range(len(times_processed) - 1)] 
            for time_index in range(len(time_int)):
                    if time_int[time_index] == 0: 
                        time_int[time_index] = 10 ** -20 
            all_mine[subdir_name + "/cleaned_csv/" + some_file] = time_int
    
    if not os.path.isdir("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
        os.makedirs("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")
    
    save_object("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_val_time", all_mine)
 
sf1, sf2 = 5, 5
for nf1 in range(sf1):
    for nf2 in range(sf2):
        read_heading(nf1, nf2)
        read_latitude_no_abs(nf1, nf2)
        read_longitude_no_abs(nf1, nf2)
        read_speed(nf1, nf2)
        read_time(nf1, nf2)