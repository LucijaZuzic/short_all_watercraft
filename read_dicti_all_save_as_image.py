from utilities import load_object
import numpy as np
import matplotlib.pyplot as plt
import os

dicti_all = load_object("dicti_all")
ord_metric = ["UniTS_longlat_speed_direction"]
metric_dicti = {"NRMSE": 2, "R2": 2, "MAE": 0, "RMSE": 0}
metric_translate = {"NRMSE": "NRMSE (\%)", "R2": "$R^{2} (\%)$", "MAE": "MAE", "RMSE": "RMSE"}
list_ws = [2, 3, 4, 5, 10, 20, 30] 
sf1, sf2 = 5, 5

metrictouse = ["MAE", "R2"]
vartouse = ["direction", "speed", "longitude_no_abs", "latitude_no_abs"]
translate_varname = {"direction": "heading", "speed": "speed", "longitude_no_abs": "$x$ offset", "latitude_no_abs": "$y$ offset", "time": "time intervals"}
start_of_table = "\\begin{figure}[!t]\n\t\\centering\n\t\\includegraphics[width = 0.99\linewidth]{FILENAME}"
end_of_table = "\n\t\\caption{METRICNAME for the VARNAME estimated on the k-fold testing datasets by different RNN models using different forecasting times, and k-fold validation datasets.}\n\t\\label{fig:k-fold_test_VARNAME_METRICNAME}\n\\end{figure}\n"
for metric_name_use in metrictouse:
    for varname in vartouse:
        line_for_model = dict()
        duplicate_val_all = True
        duplicate_val = True
        too_small = True
        mul_metric = 0
        rv_metric = 2
        while too_small or duplicate_val_all:
            set_values_all = set()
            set_values = dict()
            for val_ws in list_ws:
                set_values[val_ws] = set()
            max_col = dict()
            for val_ws in list_ws:
                max_col[val_ws] = -1000000
            min_col = dict()
            for val_ws in list_ws:
                min_col[val_ws] = 1000000
            duplicate_val_all = False
            duplicate_val = False
            too_small = False
            str_pr = ""
            first_line = metric_name_use + " " + varname + " 10^{" + str(mul_metric) + "} " + str(rv_metric)
            first_line = "\t\t\\begin{tabular}{|c|} \\hline\n\t\t\tModel"
            longc = "c"
            for model_name_use in ord_metric:
                for val_ws in list_ws:
                    first_line += " & $" + str(val_ws) + "$s"
                    longc += "|c"
                break
            first_line = first_line.replace("{|c|}", "{|"+ longc + "|}")
            for model_name_use in ord_metric:
                if "offsets" in model_name_use:
                    continue
                for nf1 in range(sf1):
                    for nf2 in range(sf2):
                        str_pr += "\t\t\t" + model_name_use.replace("_", " ").replace(" 256", "").replace(" longlat speed direction", "")
                        line_for_model[model_name_use + " test " + str(nf1 + 1) + " val " + str(nf2 + 1)] = []
                        for val_ws in list_ws: 
                            line_for_model[model_name_use + " test " + str(nf1 + 1) + " val " + str(nf2 + 1)].append(dicti_all[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use] * (10 ** metric_dicti[metric_name_use]))
                            vv = dicti_all[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use]  
                            vv = np.round(vv * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                            str_pr += " & $" + str(vv) + "$"
                            if vv in set_values[val_ws]:
                                duplicate_val = True
                            if vv in set_values_all:
                                duplicate_val_all = True
                            if "$0." in str_pr:
                                too_small = True
                            set_values[val_ws].add(vv)
                            set_values_all.add(vv)
                            if vv > max_col[val_ws]:
                                max_col[val_ws] = vv
                            if vv < min_col[val_ws]:
                                min_col[val_ws] = vv
                        str_pr += " \\\\ \\hline\n"
            if "R2" not in metric_name_use and "NRMSE" not in metric_name_use:
                if too_small:
                    mul_metric += 1
                    rv_metric = 2
                elif duplicate_val_all:
                    rv_metric += 1
            else: 
                rv_metric += 1
            if ("R2" in metric_name_use or "NRMSE" in metric_name_use) and (rv_metric > 3 or mul_metric > 3):
                break
            if rv_metric > 3 or mul_metric > 6:
                break
        if not os.path.isdir("new_img"):
            os.makedirs("new_img")
        plt.figure(figsize = (7, 15))
        for nf1 in range(sf1):
            plt.subplot(sf1 + 1, 1, nf1 + 1)
            if nf1 == 0:
                plt.title(metric_translate[metric_name_use] + "\n" + translate_varname[varname].capitalize())
            plt.ylabel("Test " + str(nf1 + 1))
            for model_name_use in ord_metric:
                for nf2 in range(sf2):
                    plt.plot(list_ws, line_for_model[model_name_use + " test " + str(nf1 + 1) + " val " + str(nf2 + 1)], label =  "val " + str(nf2 + 1))
            plt.xticks(list_ws)
            if nf1 == sf1 - 1:
                plt.xlabel("Forecasting time")
                plt.legend(ncol = 2)
        plt.savefig("new_img/" + metric_name_use + "_" + varname + ".png", bbox_inches = "tight")
        plt.close()
        newend = end_of_table.replace("METRICNAME", metric_name_use).replace("VARNAME_", varname + "_").replace("NRMSE ", "NRMSE (\%) ").replace("R2 ", "$R^{2}$ (\%) ").replace("VARNAME ", translate_varname[varname] + " ")
        newstart = start_of_table.replace("FILENAME", "new_img/" + metric_name_use + "_" + varname + ".png")
        print(newstart + newend)