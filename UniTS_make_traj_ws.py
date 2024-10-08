import pandas as pd
import os  
from utilities import load_object, save_object, get_sides_from_angle
from pytorch_utilities import get_XY

dictangle = dict()
def change_angle(angle, name_file):
        
    if name_file not in dictangle:
        file_with_ride = pd.read_csv(name_file) 
        x_dir = list(file_with_ride["fields_longitude"])[0] < list(file_with_ride["fields_longitude"])[-1]
        y_dir = list(file_with_ride["fields_latitude"])[0] < list(file_with_ride["fields_latitude"])[-1]
        dictangle[name_file] = (x_dir, y_dir)
    else:
        x_dir, y_dir = dictangle[name_file]
        
    new_dir = (90 - angle + 360) % 360 
    if not x_dir: 
        new_dir = (180 - new_dir + 360) % 360
    if not y_dir: 
        new_dir = 360 - new_dir 

    return new_dir

ws_range = [2, 3, 4, 5, 10, 20, 30]

sf1, sf2 = 5, 5

for nf1 in range(sf1):

    for nf2 in range(sf2):

        predicted_all = dict()
        y_test_all = dict()
        ws_all = dict()

        for dirnam in os.listdir("UniTS_final_res/" + str(nf1 + 1) + "/" + str(nf2 + 1)):
            
            if os.path.isdir("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam):
                continue

            model_name = "UniTS_" + dirnam

            if os.path.isfile("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/predicted_all"):
                predicted_all = load_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/predicted_all")
                
            if os.path.isfile("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/y_test_all"):
                y_test_all = load_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/y_test_all")

            if os.path.isfile("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/ws_all"):
                ws_all = load_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/ws_all")

            for varname in ["speed", "direction", "longitude_no_abs", "latitude_no_abs", "time"]:

                if varname not in predicted_all:
                    predicted_all[varname] = dict()

                if varname not in y_test_all:
                    y_test_all[varname] = dict()

                if varname not in ws_all:
                    ws_all[varname] = dict()

                if model_name not in predicted_all[varname]:
                    predicted_all[varname][model_name] = dict()
                    
                if model_name not in y_test_all[varname]:
                    y_test_all[varname][model_name] = dict()

                if model_name not in ws_all[varname]:
                    ws_all[varname][model_name] = dict()
                
                for ws_use in ws_range:

                    if ws_use not in predicted_all[varname][model_name]:
                        predicted_all[varname][model_name][ws_use] = dict()
                        
                    if ws_use not in y_test_all[varname][model_name]:
                        y_test_all[varname][model_name][ws_use] = dict()

                    if ws_use not in ws_all[varname][model_name]:
                        ws_all[varname][model_name][ws_use] = dict()

                    startdir = "UniTS_final_res_ws2/"

                    if not os.path.isdir("UniTS_final_res_ws2/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/" + str(ws_use)):
                        startdir = "UniTS_final_res_ws/"

                    if not os.path.isdir("UniTS_final_res_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/" + str(ws_use)):
                        startdir = "UniTS_final_res/"

                    if varname != "time":
                        
                        final_test_data = pd.read_csv(startdir + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/" + str(ws_use) + "/" + varname + ".csv", index_col = False)

                    #else:
                    
                        #final_test_data = pd.read_csv(startdir + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam.replace("longlat", "offsets") + "/" + str(ws_use) + "/" + varname + ".csv", index_col = False)

                    file_object_test = load_object("actual/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_" + varname)

                    len_total = 0

                    lastk = list(file_object_test.keys())[-1]

                    for k in file_object_test:
                        
                        ws_all[varname][model_name][ws_use][k] = ws_use
                        
                        if k == lastk:
                            
                            y_test_all[varname][model_name][ws_use][k] = file_object_test[k][:-(ws_use-1)]

                        else:

                            y_test_all[varname][model_name][ws_use][k] = file_object_test[k]

                        if varname != "time":
                            
                            predicted_all[varname][model_name][ws_use][k] = list(final_test_data["predicted"][len_total:len_total + len(y_test_all[varname][model_name][ws_use][k])])
                        
                        else:

                            predicted_all[varname][model_name][ws_use][k] = y_test_all[varname][model_name][ws_use][k]

                        len_total += len(y_test_all[varname][model_name][ws_use][k])  

            if not os.path.isdir("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam):
                os.makedirs("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam)

            save_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/predicted_all", predicted_all)
            save_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/y_test_all", y_test_all)
            save_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/ws_all", ws_all)

            predicted_long = dict()
            predicted_lat = dict()

            actual_long = dict()
            actual_lat = dict() 

            if os.path.isfile("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/predicted_long"):
                predicted_long = load_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/predicted_long")

            if os.path.isfile("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/predicted_lat"):
                predicted_lat = load_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/predicted_lat")

            if os.path.isfile("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/actual_long"):
                actual_long = load_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/actual_long")

            if os.path.isfile("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/actual_lat"):
                actual_lat = load_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/actual_lat")
                    
            if model_name not in actual_long:
                actual_long[model_name] = dict()
            if model_name not in actual_lat:
                actual_lat[model_name] = dict() 

            if model_name not in predicted_long:
                predicted_long[model_name] = dict()
            if model_name not in predicted_lat:
                predicted_lat[model_name] = dict()  

            for ws_use in ws_range:

                if ws_use not in actual_long[model_name]:
                    actual_long[model_name][ws_use] = dict()
                if ws_use not in actual_lat[model_name]:
                    actual_lat[model_name][ws_use] = dict()  
            
                for k in y_test_all["longitude_no_abs"][model_name][ws_use]:
                    print(nf1 + 1, nf2 + 1, model_name, k, "actual")
                    actual_long[model_name][ws_use][k] = [0]
                    actual_lat[model_name][ws_use][k] = [0]
                    
                    max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name][ws_use][k], ws_all["latitude_no_abs"][model_name][ws_use][k])
                    long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name][ws_use][k]
                    lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name][ws_use][k]
                    range_long = len(y_test_all["longitude_no_abs"][model_name][ws_use][k]) - long_offset
                    range_lat = len(y_test_all["latitude_no_abs"][model_name][ws_use][k]) - lat_offset
                    min_range_long_lat = min(range_long, range_lat)

                    for ix in range(min_range_long_lat):
                        actual_long[model_name][ws_use][k].append(actual_long[model_name][ws_use][k][-1] + y_test_all["longitude_no_abs"][model_name][ws_use][k][ix + long_offset])
                        actual_lat[model_name][ws_use][k].append(actual_lat[model_name][ws_use][k][-1] + y_test_all["latitude_no_abs"][model_name][ws_use][k][ix + lat_offset])

                if ws_use not in predicted_long[model_name]:
                    predicted_long[model_name][ws_use] = dict()
                if ws_use not in predicted_lat[model_name]:
                    predicted_lat[model_name][ws_use] = dict()  
                    
                if "long no abs" not in predicted_long[model_name][ws_use]:
                    predicted_long[model_name][ws_use]["long no abs"] = dict()
                if "lat no abs" not in predicted_lat[model_name][ws_use]:
                    predicted_lat[model_name][ws_use]["lat no abs"] = dict()  

                for k in predicted_all["longitude_no_abs"][model_name][ws_use]:
                    print(nf1 + 1, nf2 + 1, model_name, k, "long no abs")
                    predicted_long[model_name][ws_use]["long no abs"][k] = [0]
                    predicted_lat[model_name][ws_use]["lat no abs"][k] = [0]
                    
                    max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name][ws_use][k], ws_all["latitude_no_abs"][model_name][ws_use][k])
                    long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name][ws_use][k]
                    lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name][ws_use][k]
                    range_long = len(y_test_all["longitude_no_abs"][model_name][ws_use][k]) - long_offset
                    range_lat = len(y_test_all["latitude_no_abs"][model_name][ws_use][k]) - lat_offset
                    min_range_long_lat = min(range_long, range_lat)

                    for ix in range(min_range_long_lat):
                        predicted_long[model_name][ws_use]["long no abs"][k].append(predicted_long[model_name][ws_use]["long no abs"][k][-1] + predicted_all["longitude_no_abs"][model_name][ws_use][k][ix + long_offset])
                        predicted_lat[model_name][ws_use]["lat no abs"][k].append(predicted_lat[model_name][ws_use]["lat no abs"][k][-1] + predicted_all["latitude_no_abs"][model_name][ws_use][k][ix + lat_offset])

                if "long speed actual dir" not in predicted_long[model_name][ws_use]:
                    predicted_long[model_name][ws_use]["long speed actual dir"] = dict()
                if "lat speed actual dir" not in predicted_lat[model_name][ws_use]:
                    predicted_lat[model_name][ws_use]["lat speed actual dir"] = dict()  

                for k in predicted_all["speed"][model_name][ws_use]:
                    print(nf1 + 1, nf2 + 1, model_name, k, "long speed actual dir")
                    predicted_long[model_name][ws_use]["long speed actual dir"][k] = [0]
                    predicted_lat[model_name][ws_use]["lat speed actual dir"][k] = [0]
                
                    max_offset_speed_dir_time = max(max(ws_all["speed"][model_name][ws_use][k], ws_all["direction"][model_name][ws_use][k]), ws_all["time"][model_name][ws_use][k])
                    speed_offset_time = max_offset_speed_dir_time - ws_all["speed"][model_name][ws_use][k]
                    dir_offset_time = max_offset_speed_dir_time - ws_all["direction"][model_name][ws_use][k]
                    time_offset_time = max_offset_speed_dir_time - ws_all["time"][model_name][ws_use][k]
                    range_speed_time = len(y_test_all["speed"][model_name][ws_use][k]) - speed_offset_time
                    range_dir_time = len(y_test_all["direction"][model_name][ws_use][k]) - dir_offset_time
                    range_time_time = len(y_test_all["time"][model_name][ws_use][k]) - time_offset_time
                    min_range_speed_dir_time = min(min(range_speed_time, range_dir_time), range_time_time)

                    for ix in range(min_range_speed_dir_time):
                        new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][ws_use][k][ix + speed_offset_time] / 111 / 0.1 / 3600 * y_test_all["time"][model_name][ws_use][k][ix + time_offset_time], change_angle(predicted_all["direction"][model_name][ws_use][k][ix + dir_offset_time], k))
                        predicted_long[model_name][ws_use]["long speed actual dir"][k].append(predicted_long[model_name][ws_use]["long speed actual dir"][k][-1] + new_long)
                        predicted_lat[model_name][ws_use]["lat speed actual dir"][k].append(predicted_lat[model_name][ws_use]["lat speed actual dir"][k][-1] + new_lat)

                if "long speed ones dir" not in predicted_long[model_name][ws_use]:
                    predicted_long[model_name][ws_use]["long speed ones dir"] = dict()
                if "lat speed ones dir" not in predicted_lat[model_name][ws_use]:
                    predicted_lat[model_name][ws_use]["lat speed ones dir"] = dict()  

                for k in predicted_all["speed"][model_name][ws_use]:
                    print(nf1 + 1, nf2 + 1, model_name, k, "long speed ones dir")
                    predicted_long[model_name][ws_use]["long speed ones dir"][k] = [0]
                    predicted_lat[model_name][ws_use]["lat speed ones dir"][k] = [0]
                
                    max_offset_speed_dir = max(ws_all["speed"][model_name][ws_use][k], ws_all["direction"][model_name][ws_use][k])
                    speed_offset = max_offset_speed_dir - ws_all["speed"][model_name][ws_use][k]
                    dir_offset = max_offset_speed_dir - ws_all["direction"][model_name][ws_use][k]
                    range_speed = len(y_test_all["speed"][model_name][ws_use][k]) - speed_offset
                    range_dir = len(y_test_all["direction"][model_name][ws_use][k]) - dir_offset
                    min_range_speed_dir = min(range_speed, range_dir)

                    for ix in range(min_range_speed_dir):
                        new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][ws_use][k][ix + speed_offset] / 111 / 0.1 / 3600, change_angle(predicted_all["direction"][model_name][ws_use][k][ix + dir_offset], k))
                        predicted_long[model_name][ws_use]["long speed ones dir"][k].append(predicted_long[model_name][ws_use]["long speed ones dir"][k][-1] + new_long)
                        predicted_lat[model_name][ws_use]["lat speed ones dir"][k].append(predicted_lat[model_name][ws_use]["lat speed ones dir"][k][-1] + new_lat)

                continue

                if "long speed dir" not in predicted_long[model_name][ws_use]:
                    predicted_long[model_name][ws_use]["long speed dir"] = dict()
                if "lat speed dir" not in predicted_lat[model_name][ws_use]:
                    predicted_lat[model_name][ws_use]["lat speed dir"] = dict()  

                for k in predicted_all["speed"][model_name][ws_use]:
                    print(nf1 + 1, nf2 + 1, model_name, k, "long speed dir")
                    predicted_long[model_name][ws_use]["long speed dir"][k] = [0]
                    predicted_lat[model_name][ws_use]["lat speed dir"][k] = [0]
                
                    max_offset_speed_dir_time = max(max(ws_all["speed"][model_name][ws_use][k], ws_all["direction"][model_name][ws_use][k]), ws_all["time"][model_name][ws_use][k])
                    speed_offset_time = max_offset_speed_dir_time - ws_all["speed"][model_name][ws_use][k]
                    dir_offset_time = max_offset_speed_dir_time - ws_all["direction"][model_name][ws_use][k]
                    time_offset_time = max_offset_speed_dir_time - ws_all["time"][model_name][ws_use][k]
                    range_speed_time = len(y_test_all["speed"][model_name][ws_use][k]) - speed_offset_time
                    range_dir_time = len(y_test_all["direction"][model_name][ws_use][k]) - dir_offset_time
                    range_time_time = len(y_test_all["time"][model_name][ws_use][k]) - time_offset_time
                    min_range_speed_dir_time = min(min(range_speed_time, range_dir_time), range_time_time)

                    for ix in range(min_range_speed_dir_time):
                        new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][ws_use][k][ix + speed_offset_time] / 111 / 0.1 / 3600 * predicted_all["time"][model_name][ws_use][k][ix + time_offset_time], change_angle(predicted_all["direction"][model_name][ws_use][k][ix + dir_offset_time], k))
                        predicted_long[model_name][ws_use]["long speed dir"][k].append(predicted_long[model_name][ws_use]["long speed dir"][k][-1] + new_long)
                        predicted_lat[model_name][ws_use]["lat speed dir"][k].append(predicted_lat[model_name][ws_use]["lat speed dir"][k][-1] + new_lat)

            if not os.path.isdir("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1)):
                os.makedirs("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1))

            save_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/actual_long", actual_long)
            save_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/actual_lat", actual_lat)
            save_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/predicted_long", predicted_long)
            save_object("UniTS_final_result_ws/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + dirnam + "/predicted_lat", predicted_lat)