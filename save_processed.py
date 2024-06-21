from utilities import load_object, save_object, preprocess_long_lat, scale_long_lat
import os
import pandas as pd
        
all_subdirs = os.listdir()

dict_long_lat = dict()

sf1, sf2 = 5, 5
for nf1 in range(sf1):
    for nf2 in range(sf2):
        for subdir_name in all_subdirs: 
            if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
                continue
            dict_long_lat[subdir_name] = dict()
            all_files = os.listdir(subdir_name + "/cleaned_csv/") 
            bad_rides_filenames = set()
            if os.path.isfile(subdir_name + "/bad_rides_filenames"):
                bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
            gap_rides_filenames = set()
            if os.path.isfile(subdir_name + "/gap_rides_filenames"):
                gap_rides_filenames = load_object(subdir_name + "/gap_rides_filenames")
            test_rides = set()
            if os.path.isfile(subdir_name + "/test_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1)):
                test_rides = load_object(subdir_name + "/test_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1))
            val_rides = set()
            if os.path.isfile(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1)):
                val_rides = load_object(subdir_name + "/val_rides_" + str(nf1 + 1) + "_" + str(nf2 + 1))

            test_rides_veh = []
            train_rides_veh = []
            rides_veh = []
                
            for some_file in all_files:  

                if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames:
                    continue
                file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)

                longitudes = list(file_with_ride["fields_longitude"]) 
                latitudes = list(file_with_ride["fields_latitude"]) 

                longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
                longitudes, latitudes = scale_long_lat(longitudes, latitudes, 0.1, 0.1, True)

                if some_file in test_rides:
                    dict_long_lat[subdir_name][some_file] = [longitudes, latitudes, "test"]
                else:
                    if some_file in val_rides:
                        dict_long_lat[subdir_name][some_file] = [longitudes, latitudes, "val"]
                    else:
                        dict_long_lat[subdir_name][some_file] = [longitudes, latitudes, "train"]

        if not os.path.isdir("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
            os.makedirs("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")

        save_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_traj", dict_long_lat)