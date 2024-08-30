import pandas as pd
import os  
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utilities import load_object, save_object
import numpy as np

ws_range = [2, 3, 4, 5, 10, 20, 30]

hidden_range = [256]

model_list = ["GRU", "LSTM", "RNN"]

modes = ["Reference", "Third", "Linear", "Twice"]

dicti_to_print = dict()

dicti_all = dict()
if os.path.isfile("dicti_all"):
    dicti_all = load_object("dicti_all")

sf1, sf2 = 5, 5
for nf1 in range(sf1):
    
    dicti_to_print[nf1]  = dict()

    if nf1 not in dicti_all:
        dicti_all[nf1] = dict()

    for nf2 in range(sf2):

        dicti_to_print[nf1][nf2] = dict()

        if nf2 not in dicti_all[nf1]:
            dicti_all[nf1][nf2] = dict()

        for varname in os.listdir("train_pytorch/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/Reference/"):
                
            dicti_to_print[nf1][nf2][varname] = dict()

            if varname not in dicti_all[nf1][nf2]:
                dicti_all[nf1][nf2][varname] = dict()

            for mod_use in modes:
                
                final_test_NRMSE = []
                final_test_RMSE = []
                final_test_R2 = []
                final_test_MAE = []
                
                hidden_arr = []
                ws_arr = []
                model_arr = []

                all_mine = load_object("actual/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_" + varname)
                all_mine_flat = []
                for filename in all_mine: 
                    for val in all_mine[filename]:
                        all_mine_flat.append(val)
                        
                for model_name in model_list:
                    for ws_use in ws_range:
                        for hidden_use in hidden_range:

                            if model_name + "_" + mod_use + "_" + str(hidden_use) not in dicti_all[nf1][nf2][varname]:
                                dicti_all[nf1][nf2][varname][model_name + "_" + mod_use + "_" + str(hidden_use)] = dict()

                            if str(ws_use) not in dicti_all[nf1][nf2][varname][model_name + "_" + mod_use + "_" + str(hidden_use)]:
                                print(nf1, nf2, varname, model_name + "_" + mod_use + "_" + str(hidden_use), ws_use)
                                dicti_all[nf1][nf2][varname][model_name + "_" + mod_use + "_" + str(hidden_use)][str(ws_use)] = dict()
                            else:
                                continue
            
                            final_test_data = pd.read_csv("train_pytorch/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + mod_use + "/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_test.csv", sep = ";", index_col = False)
                    
                            is_a_nan = False
                            for val in final_test_data["predicted"]:
                                if str(val) == 'nan':
                                    is_a_nan = True
                                    break

                            hidden_arr.append(hidden_use)
                            ws_arr.append(ws_use)
                            model_arr.append(model_name + "_" + mod_use)

                            final_test_data_predicted = [float(x.split(",")[0]) for x in final_test_data["predicted"]]
                                
                            if is_a_nan:
                                final_test_MAE.append(1000000)
                                final_test_R2.append(1000000)
                                final_test_NRMSE.append(1000000)
                                final_test_RMSE.append(1000000)
                            else:
                                final_test_MAE.append(mean_absolute_error(final_test_data["actual"], final_test_data_predicted))
                                final_test_R2.append(r2_score(final_test_data["actual"], final_test_data_predicted))
                                final_test_NRMSE.append(math.sqrt(mean_squared_error(final_test_data["actual"], final_test_data_predicted)) / (max(all_mine_flat) - min(all_mine_flat)))
                                final_test_RMSE.append(math.sqrt(mean_squared_error(final_test_data["actual"], final_test_data_predicted)))
                
                for mini_ix_val in range(len(final_test_RMSE)):
                    if model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val]) not in dicti_to_print[nf1][nf2][varname]:
                        dicti_to_print[nf1][nf2][varname][model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val])] = dict()

                    dicti_all[nf1][nf2][varname][model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val])][str(ws_arr[mini_ix_val])] = dict()
                    dicti_all[nf1][nf2][varname][model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val])][str(ws_arr[mini_ix_val])]["NRMSE"] = final_test_NRMSE[mini_ix_val] 
                    dicti_all[nf1][nf2][varname][model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val])][str(ws_arr[mini_ix_val])]["RMSE"] = final_test_RMSE[mini_ix_val] 
                    dicti_all[nf1][nf2][varname][model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val])][str(ws_arr[mini_ix_val])]["R2"] = final_test_R2[mini_ix_val]  
                    dicti_all[nf1][nf2][varname][model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val])][str(ws_arr[mini_ix_val])]["MAE"] = final_test_MAE[mini_ix_val]  
                    
                    save_object("dicti_all", dicti_all)
                    
                    dicti_to_print[nf1][nf2][varname][model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val])][str(ws_arr[mini_ix_val])] = dict()
                    dicti_to_print[nf1][nf2][varname][model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val])][str(ws_arr[mini_ix_val])]["NRMSE"] = final_test_NRMSE[mini_ix_val] 
                    dicti_to_print[nf1][nf2][varname][model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val])][str(ws_arr[mini_ix_val])]["RMSE"] = final_test_RMSE[mini_ix_val] 
                    dicti_to_print[nf1][nf2][varname][model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val])][str(ws_arr[mini_ix_val])]["R2"] = final_test_R2[mini_ix_val]  
                    dicti_to_print[nf1][nf2][varname][model_arr[mini_ix_val] + "_" + str(hidden_arr[mini_ix_val])][str(ws_arr[mini_ix_val])]["MAE"] = final_test_MAE[mini_ix_val]  
                    print(model_arr[mini_ix_val], hidden_arr[mini_ix_val], ws_arr[mini_ix_val], np.round(final_test_NRMSE[mini_ix_val] * 100, 2), np.round(final_test_R2[mini_ix_val] * 100, 2), np.round(final_test_MAE[mini_ix_val], 6), np.round(final_test_RMSE[mini_ix_val], 6))

rv_metric = {"R2": 2, "RMSE": 6, "MAE": 6, "NRMSE": 2}
mul_metric = {"R2": 100, "RMSE": 1, "MAE": 1, "NRMSE": 100}
list_ws = sorted([int(x) for x in dicti_all[0][0]["speed"]["LSTM_Reference_256"]])
list_ws = [2, 3, 4, 5, 10, 20, 30]

for nf1 in range(sf1):
    for nf2 in range(sf2):
        for metric_name_use in list(rv_metric.keys()):
            for varname in dicti_all[nf1][nf2]:
                str_pr = ""
                first_line = str(nf1 + 1) + " " +  str(nf2 + 1) + " " + metric_name_use + " " + varname
                for model_name_use in dicti_all[nf1][nf2][varname]:
                    for val_ws in list_ws:
                        first_line += " & $" + str(val_ws) + "$s"
                    break
                #print(first_line + " \\\\ \\hline")
                for model_name_use in dicti_all[nf1][nf2][varname]:
                    str_pr += str(nf1 + 1) + " " +  str(nf2 + 1) + " " + varname + " " + metric_name_use + " " + model_name_use
                    for val_ws in list_ws:
                        vv = dicti_all[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use]
                        if nf1 not in dicti_all:
                            dicti_all[nf1]  = dict()
                        if nf2 not in dicti_all[nf1]:
                            dicti_all[nf1][nf2] = dict()
                        if varname not in dicti_all[nf1][nf2]:
                            dicti_all[nf1][nf2][varname] = dict()
                        if model_name_use not in dicti_all[nf1][nf2][varname]:
                            dicti_all[nf1][nf2][varname][model_name_use] = dict()
                        if str(val_ws) not in dicti_all[nf1][nf2][varname][model_name_use]:
                            dicti_all[nf1][nf2][varname][model_name_use][str(val_ws)] = dict()
                        dicti_all[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use] = vv
                        vv = np.round(vv * mul_metric[metric_name_use], rv_metric[metric_name_use])
                        str_pr += " & $" + str(vv) + "$"
                    str_pr += " \\\\ \\hline\n"
                #print(str_pr)

        for metric_name_use in list(rv_metric.keys()):
            for model_name_use in dicti_all[0][0]["speed"]:
                str_pr = ""
                first_line = str(nf1 + 1) + " " +  str(nf2 + 1) + " " + metric_name_use + " " + model_name_use
                for varname in dicti_all[nf1][nf2]:
                    for val_ws in list_ws:
                        first_line += " & $" + str(val_ws) + "$s"
                    break
                print(first_line + " \\\\ \\hline")
                for varname in dicti_all[nf1][nf2]:
                    str_pr += str(nf1 + 1) + " " +  str(nf2 + 1) + " " + varname + " " + metric_name_use + " " + model_name_use
                    for val_ws in list_ws: 
                        vv = dicti_all[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use]
                        if nf1 not in dicti_all:
                            dicti_all[nf1]  = dict()
                        if nf2 not in dicti_all[nf1]:
                            dicti_all[nf1][nf2] = dict()
                        if varname not in dicti_all[nf1][nf2]:
                            dicti_all[nf1][nf2][varname] = dict()
                        if model_name_use not in dicti_all[nf1][nf2][varname]:
                            dicti_all[nf1][nf2][varname][model_name_use] = dict()
                        if str(val_ws) not in dicti_all[nf1][nf2][varname][model_name_use]:
                            dicti_all[nf1][nf2][varname][model_name_use][str(val_ws)] = dict()
                        dicti_all[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use] = vv
                        vv = np.round(vv * mul_metric[metric_name_use], rv_metric[metric_name_use])
                        str_pr += " & $" + str(vv) + "$"
                    str_pr += " \\\\ \\hline\n"
                print(str_pr)