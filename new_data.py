from utilities import load_object, process_time, save_object
import os
import pandas as pd
import numpy as np
    
all_subdirs = os.listdir() 

for subdir_name in all_subdirs:
    if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
        continue
    print(subdir_name)
     
    max_start_found = process_time("2012-07-01 00:00:00.000")
    min_start_found = process_time("2032-07-01 00:00:00.000")  
 
    asc_rides = set()
    desc_rides = set()  
    mixed_rides = set() 
 
    ride_for_file = dict() 

    files_seen = set()  
    good_rides_filenames = dict()
    
    bad_rides_filenames = dict()
    if os.path.isfile(subdir_name + "/bad_rides_filenames"):
        bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames") 

    all_files = os.listdir(subdir_name + "/cleaned_csv/")  

    max_gap = process_time("2012-07-01 00:00:00.000") - process_time("2012-07-01 00:00:00.000")
    avg_gap = []
    lens = []
    start_time = []
    end_time = []

    good_avg_gap = []
    good_lens = []
    good_start_time = []
    good_end_time = []

    sorted_set = set()
    reverse_set = set()
    unsorted_set = set()       
           
    for some_file in all_files: 
        if some_file[0] == "e" and ".csv" in some_file: 
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file) 
            list_time = list(file_with_ride["time"])
            start_time.append(min(list_time))
            end_time.append(max(list_time))
            lens.append(len(list_time))

            file_name_new = subdir_name + "/cleaned_csv/" + some_file
 
            sgn_set = set()
            for x in range(len(list_time)):
                list_time[x] = process_time(list_time[x])
                if x > 0:
                    sgn_set.add(list_time[x] > list_time[x - 1]) 
           
            if len(sgn_set) == 1 and False in sgn_set:
                reverse_set.add(some_file)
            if len(sgn_set) == 1 and True in sgn_set:
                sorted_set.add(some_file) 
            if len(sgn_set) > 1:
                unsorted_set.add(some_file)
            #print(sgn_set)
        
            if some_file in reverse_set:
                list_time_sorted = [list_time[len(list_time) - 1 - x] for x in range(len(list_time))]
            if some_file in sorted_set:
                list_time_sorted = [ele for ele in list_time]
            if some_file in unsorted_set:
                list_time_sorted = sorted(list_time)
            
            max_gap_for_ride = process_time("2012-07-01 00:00:00.000") - process_time("2012-07-01 00:00:00.000")
            avg_gap_for_ride = []
            for x in range(len(list_time_sorted) - 1):
                gap_for_ride = list_time_sorted[x + 1] - list_time_sorted[x]
                avg_gap.append(gap_for_ride)
                avg_gap_for_ride.append(gap_for_ride)
                max_gap_for_ride = max(max_gap_for_ride, gap_for_ride)
                max_gap = max(max_gap, max_gap_for_ride)
            #print("max gap for ride", max_gap_for_ride)

            if max_gap_for_ride < 4:
                if len(list_time) < 60:
                    bad_rides_filenames[file_name_new] = -1000
                else:
                    if file_name_new not in bad_rides_filenames or (file_name_new in bad_rides_filenames and bad_rides_filenames[file_name_new] > 0):
                        good_rides_filenames[file_name_new] = max_gap_for_ride
                    if file_name_new in bad_rides_filenames and bad_rides_filenames[file_name_new] > 0: 
                        bad_rides_filenames.pop(file_name_new)
                    good_avg_gap.append(avg_gap[-1])
                    good_lens.append(lens[-1])
                    good_end_time.append(end_time[-1])
                    good_start_time.append(start_time[-1])
            else:
                if file_name_new not in bad_rides_filenames or (file_name_new in bad_rides_filenames and bad_rides_filenames[file_name_new] > 0):
                    bad_rides_filenames[file_name_new] = max_gap_for_ride
            ride_for_file[some_file] = file_name_new
  
    #print(len(sorted_set), len(reverse_set), len(unsorted_set)) 
    print(len(bad_rides_filenames), len(good_rides_filenames), len(ride_for_file))
    save_object(subdir_name + "/bad_rides_filenames", bad_rides_filenames) 
    if len(avg_gap) == 0: 
        continue
    #print(min(start_time), max(start_time))
    #print(min(end_time), max(end_time))
    print(np.average(avg_gap), min(start_time), max(end_time), min(lens))
    ls = 0
    for l in lens: 
        if l < 60:
            ls += 1
    if ls != 0:
        print(ls)
    if len(good_avg_gap) == 0: 
        continue
    print(np.average(good_avg_gap), min(good_start_time), max(good_end_time), min(good_lens))
    lsg = 0
    for l in good_lens: 
        if l < 60:
            lsg += 1
    if lsg != 0:
        print(lsg)