from utilities import load_object, save_object
import numpy as np

dicti_all = load_object("dicti_all")
dicti_all_ws = load_object("dicti_all_ws")
dicti_all_latest = dict()
dicti_all_traj = load_object("dicti_all_traj")
dicti_all_traj_ws = load_object("dicti_all_traj_ws")
dicti_all_traj_latest = dict()

ord_metric_traj = dicti_all_traj[0][0]["long no abs"].keys()
ord_metric = dicti_all[0][0]["speed"].keys()
metric_dicti_traj = {"Euclid": 0, "R2": 2, "MAE": 0, "RMSE": 0, "R2_wt": 2, "MAE_wt": 0, "RMSE_wt": 0}
metric_dicti = {"NRMSE": 2, "R2": 2, "MAE": 0, "RMSE": 0}
list_ws = [2, 3, 4, 5, 10, 20, 30] 
 
sf1, sf2 = 5, 5
for nf1 in range(sf1):
    dicti_all_latest[nf1] = dict()
    for nf2 in range(sf2):
        dicti_all_latest[nf1][nf2] = dict()
        for varname in dicti_all[nf1][nf2]:
            dicti_all_latest[nf1][nf2][varname] = dict()
            for model_name_use in ord_metric:
                dicti_all_latest[nf1][nf2][varname][model_name_use] = dict()
                for val_ws in list_ws:
                    dicti_all_latest[nf1][nf2][varname][model_name_use][str(val_ws)] = dict()
                    for metric_name_use in list(metric_dicti.keys()):
                        vv = dicti_all[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use]
                        if model_name_use in dicti_all_ws[nf1][nf2][varname]:
                            vv = dicti_all_ws[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use]
                        dicti_all_latest[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use] = vv
                      
save_object("dicti_all_latest", dicti_all_latest)

for nf1 in range(sf1):
    dicti_all_traj_latest[nf1] = dict()
    for nf2 in range(sf2):
        dicti_all_traj_latest[nf1][nf2] = dict()
        for varname in dicti_all_traj[nf1][nf2]:
            dicti_all_traj_latest[nf1][nf2][varname] = dict()
            for model_name_use in ord_metric_traj:
                dicti_all_traj_latest[nf1][nf2][varname][model_name_use] = dict()
                for val_ws in list_ws:
                    dicti_all_traj_latest[nf1][nf2][varname][model_name_use][str(val_ws)] = dict()
                    for metric_name_use in list(metric_dicti_traj.keys()):
                        vv = dicti_all_traj[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use]
                        if model_name_use in dicti_all_traj_ws[nf1][nf2][varname]:
                            vv = dicti_all_traj_ws[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use]
                        dicti_all_traj_latest[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use] = vv

save_object("dicti_all_traj_latest", dicti_all_traj_latest)

dicti_all_latest_short = dict()
dicti_all_traj_latest_short = dict()

dicti_all_latest_avg = dict()
dicti_all_traj_latest_avg = dict()

dicti_all_latest_std = dict()
dicti_all_traj_latest_std = dict()

for varname in dicti_all[0][0]:
    dicti_all_latest_short[varname] = dict()
    dicti_all_latest_avg[varname] = dict()
    dicti_all_latest_std[varname] = dict()
    for model_name_use in ord_metric:
        dicti_all_latest_short[varname][model_name_use] = dict()
        dicti_all_latest_avg[varname][model_name_use] = dict()
        dicti_all_latest_std[varname][model_name_use] = dict()
        for val_ws in list_ws:
            dicti_all_latest_short[varname][model_name_use][str(val_ws)] = dict()
            dicti_all_latest_avg[varname][model_name_use][str(val_ws)] = dict()
            dicti_all_latest_std[varname][model_name_use][str(val_ws)] = dict()
            for metric_name_use in list(metric_dicti.keys()):
                arr = []
                for nf1 in range(sf1):
                    for nf2 in range(sf2):
                        arr.append(dicti_all_latest[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use])
                dicti_all_latest_short[varname][model_name_use][str(val_ws)][metric_name_use] = arr
                dicti_all_latest_avg[varname][model_name_use][str(val_ws)][metric_name_use] = np.mean(arr)
                dicti_all_latest_std[varname][model_name_use][str(val_ws)][metric_name_use] = np.std(arr)

save_object("dicti_all_latest_short", dicti_all_latest_short)
save_object("dicti_all_latest_avg", dicti_all_latest_avg)
save_object("dicti_all_latest_std", dicti_all_latest_std)

for varname in dicti_all_traj[0][0]:
    dicti_all_traj_latest_short[varname] = dict()
    dicti_all_traj_latest_avg[varname] = dict()
    dicti_all_traj_latest_std[varname] = dict()
    for model_name_use in ord_metric_traj:
        dicti_all_traj_latest_short[varname][model_name_use] = dict()
        dicti_all_traj_latest_avg[varname][model_name_use] = dict()
        dicti_all_traj_latest_std[varname][model_name_use] = dict()
        for val_ws in list_ws:
            dicti_all_traj_latest_short[varname][model_name_use][str(val_ws)] = dict()
            dicti_all_traj_latest_avg[varname][model_name_use][str(val_ws)] = dict()
            dicti_all_traj_latest_std[varname][model_name_use][str(val_ws)] = dict()
            for metric_name_use in list(metric_dicti_traj.keys()):
                arr = []
                for nf1 in range(sf1):
                    for nf2 in range(sf2):
                        arr.append(dicti_all_traj_latest[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use])
                dicti_all_traj_latest_short[varname][model_name_use][str(val_ws)][metric_name_use] = arr
                dicti_all_traj_latest_avg[varname][model_name_use][str(val_ws)][metric_name_use] = np.mean(arr)
                dicti_all_traj_latest_std[varname][model_name_use][str(val_ws)][metric_name_use] = np.std(arr)

save_object("dicti_all_traj_latest_short", dicti_all_traj_latest_short)
save_object("dicti_all_traj_latest_avg", dicti_all_traj_latest_avg)
save_object("dicti_all_traj_latest_std", dicti_all_traj_latest_std)

total_errs = set()

metrictouse = ["MAE", "R2"]
vartouse = ["direction", "speed", "longitude_no_abs", "latitude_no_abs"]
translate_varname = {"direction": "heading", "speed": "speed", "longitude_no_abs": "$x$ offset", "latitude_no_abs": "$y$ offset", "time": "time intervals"}
start_of_table = "\\begin{table*}[!t]\n\t\\begin{center}\n\t\t\\caption{The average METRICNAME across k-fold validation datasets, with standard deviation in brackets, for the VARNAME estimated on the k-fold testing datasets by different RNN models using different forecasting times.}\n\t\t\\label{tab:val_VARNAME_METRICNAME}\n\t\t\\resizebox{\linewidth}{!}{"
end_of_table = "\t\t\\end{tabular}}\n\t\\end{center}\n\\end{table*}\n"
for metric_name_use in metrictouse:
    for varname in vartouse:
        duplicate_val_all = True
        duplicate_val = True
        too_small = True
        mul_metric = 0
        rv_metric = 2
        while too_small or duplicate_val_all:
            errs = set()
            set_values_all = set()
            set_values = dict()
            for val_ws in list_ws:
                set_values[val_ws] = set()
            max_col = dict()
            for val_ws in list_ws:
                max_col[val_ws] = (-1000000, 0)
            min_col = dict()
            for val_ws in list_ws:
                min_col[val_ws] = (1000000, 0)
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
                str_pr += "\t\t\t\multirow{2}{*}{" + model_name_use.replace("_", " ").replace(" 256", "").replace(" longlat speed direction", "") + "}"
                for val_ws in list_ws: 
                    vv = dicti_all_latest_avg[varname][model_name_use][str(val_ws)][metric_name_use]  
                    vv = np.round(vv * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                    vv2 = dicti_all_latest_std[varname][model_name_use][str(val_ws)][metric_name_use]  
                    vv2 = np.round(vv2 * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                    old_str = vv
                    if "e+" in str(vv):
                        parts_12 = str(vv).split("e+")
                        main_part = float(parts_12[0])
                        exp_part = int(parts_12[1])
                        vv = str(np.round(main_part, rv_metric)) + " \times 10^{" + str(exp_part) + "}"
                        errs.add((varname, model_name_use, str(val_ws), metric_name_use, old_str, vv))
                    if "e+" in str(vv2):
                        old_str2 = vv2
                        parts_12 = str(vv2).split("e+")
                        main_part = float(parts_12[0])
                        exp_part = int(parts_12[1])
                        vv2 = str(np.round(main_part, rv_metric)) + " \times 10^{" + str(exp_part) + "}"
                        errs.add((varname, model_name_use, str(val_ws), metric_name_use, old_str2, vv2))
                    str_pr += " & $" + str(vv) + "$"
                    vv = old_str
                    if vv in set_values[val_ws]:
                        duplicate_val = True
                    if vv in set_values_all:
                        duplicate_val_all = True
                    if "$0." in str_pr:
                        too_small = True
                    set_values[val_ws].add(vv)
                    set_values_all.add(vv)
                    if vv > max_col[val_ws][0]:
                        max_col[val_ws] = (vv, vv2)
                    if vv < min_col[val_ws][0]:
                        min_col[val_ws] = (vv, vv2)
                str_pr += " \\\\ \\hline\n"
                str_pr += "\t\t\t"
                for val_ws in list_ws: 
                    vv = dicti_all_latest_std[varname][model_name_use][str(val_ws)][metric_name_use]  
                    vv = np.round(vv * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                    if "e+" in str(vv):
                        old_str = vv
                        parts_12 = str(vv).split("e+")
                        main_part = float(parts_12[0])
                        exp_part = int(parts_12[1])
                        vv = str(np.round(main_part, rv_metric)) + " \times 10^{" + str(exp_part) + "}"
                        errs.add((varname, model_name_use, str(val_ws), metric_name_use, old_str, vv))
                    str_pr += " & $" + str(vv) + "$"
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
        if "R2" in metric_name_use:
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(max_col[val_ws][0]) + "$", "$\\mathbf{" + str(max_col[val_ws][0]) + "}$") 
                str_pr = str_pr.replace("($" + str(max_col[val_ws][1]) + "$)", "($\\mathbf{" + str(max_col[val_ws][1]) + "}$)") 
        else:
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(min_col[val_ws][0]) + "$", "$\\mathbf{" + str(min_col[val_ws][0]) + "}$") 
                str_pr = str_pr.replace("($" + str(min_col[val_ws][1]) + "$)", "($\\mathbf{" + str(min_col[val_ws][1]) + "}$)") 
        newstart = start_of_table.replace("METRICNAME", metric_name_use).replace("VARNAME_", varname + "_").replace("NRMSE ", "NRMSE (\%) ").replace("R2 ", "$R^{2}$ (\%) ").replace("VARNAME ", translate_varname[varname] + " ")
        if "R2" not in metric_name_use and "NRMSE" not in metric_name_use and mul_metric != 0:
            newstart = newstart.replace(metric_name_use + " ", metric_name_use + " ($\\times 10^{-" + str(mul_metric) + "}$) ")
        print(newstart)
        print(first_line + " \\\\ \\hline")
        print(str_pr + end_of_table)
        for e in errs:
            total_errs.add(e)

metrictouse = ["Euclid", "MAE", "R2"]
vartouse = ["long speed actual dir", "long no abs"]
translate_varname = {"long speed ones dir": "speed, heading, a fixed one-second time interval",
                    "long speed dir": "speed, heading, time intervals",
                    "long speed actual dir": "speed, heading, the actual time interval",
                    "long no abs": "$x$ and $y$ offset"}
start_of_table = "\\begin{table*}[!t]\n\t\\begin{center}\n\t\t\\caption{The average METRICNAME across k-fold validation datasets, with standard deviation in brackets, for the trajectories in the k-fold testing datasets estimated using VARNAME, different RNN models, and different forecasting times.}\n\t\t\\label{tab:val_VARNAME_METRICNAME}\n\t\t\\resizebox{\linewidth}{!}{"
end_of_table = "\t\t\\end{tabular}}\n\t\\end{center}\n\\end{table*}\n"
for metric_name_use in metrictouse:
    for varname in vartouse:
        duplicate_val_all = True
        duplicate_val = True
        too_small = True
        mul_metric = 0
        rv_metric = 2
        while too_small or duplicate_val_all:
            errs = set()
            set_values_all = set()
            set_values = dict()
            for val_ws in list_ws:
                set_values[val_ws] = set()
            max_col = dict()
            for val_ws in list_ws:
                max_col[val_ws] = (-1000000, 0)
            min_col = dict()
            for val_ws in list_ws:
                min_col[val_ws] = (1000000, 0)
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
            for model_name_use in ord_metric_traj:
                if "offsets" in model_name_use:
                    continue
                str_pr += "\t\t\t\multirow{2}{*}{" + model_name_use.replace("_", " ").replace(" 256", "").replace(" longlat speed direction", "") + "}"
                for val_ws in list_ws: 
                    vv = dicti_all_traj_latest_avg[varname][model_name_use][str(val_ws)][metric_name_use]   
                    vv = np.round(vv * (10 ** metric_dicti_traj[metric_name_use]) * (10 ** mul_metric), rv_metric)
                    vv2 = dicti_all_traj_latest_std[varname][model_name_use][str(val_ws)][metric_name_use] 
                    vv2 = np.round(vv2 * (10 ** metric_dicti_traj[metric_name_use]) * (10 ** mul_metric), rv_metric)
                    old_str = vv
                    if "e+" in str(vv):
                        parts_12 = str(vv).split("e+")
                        main_part = float(parts_12[0])
                        exp_part = int(parts_12[1])
                        vv = str(np.round(main_part, rv_metric)) + " \times 10^{" + str(exp_part) + "}"
                        errs.add((varname, model_name_use, str(val_ws), metric_name_use, old_str, vv))
                    if "e+" in str(vv2):
                        old_str2 = vv2
                        parts_12 = str(vv2).split("e+")
                        main_part = float(parts_12[0])
                        exp_part = int(parts_12[1])
                        vv2 = str(np.round(main_part, rv_metric)) + " \times 10^{" + str(exp_part) + "}"
                        errs.add((varname, model_name_use, str(val_ws), metric_name_use, old_str2, vv2))
                    str_pr += " & $" + str(vv) + "$"
                    vv = old_str
                    if vv in set_values[val_ws]:
                        duplicate_val = True
                    if vv in set_values_all:
                        duplicate_val_all = True
                    if "$0." in str_pr:
                        too_small = True
                    set_values[val_ws].add(vv)
                    set_values_all.add(vv)
                    if vv > max_col[val_ws][0]:
                        max_col[val_ws] = (vv, vv2)
                    if vv < min_col[val_ws][0]:
                        min_col[val_ws] = (vv, vv2)
                str_pr += " \\\\ \\hline\n"
                str_pr += "\t\t\t"
                for val_ws in list_ws: 
                    vv = dicti_all_traj_latest_std[varname][model_name_use][str(val_ws)][metric_name_use]  
                    vv = np.round(vv * (10 ** metric_dicti_traj[metric_name_use]) * (10 ** mul_metric), rv_metric)
                    if "e+" in str(vv):
                        old_str = vv
                        parts_12 = str(vv).split("e+")
                        main_part = float(parts_12[0])
                        exp_part = int(parts_12[1])
                        vv = str(np.round(main_part, rv_metric)) + " \times 10^{" + str(exp_part) + "}"
                        errs.add((varname, model_name_use, str(val_ws), metric_name_use, old_str, vv))
                    str_pr += " & ($" + str(vv) + "$)"
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
        if "R2" in metric_name_use:
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(max_col[val_ws][0]) + "$", "$\\mathbf{" + str(max_col[val_ws][0]) + "}$") 
                str_pr = str_pr.replace("($" + str(max_col[val_ws][1]) + "$)", "($\\mathbf{" + str(max_col[val_ws][1]) + "}$)") 
        else:
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(min_col[val_ws][0]) + "$", "$\\mathbf{" + str(min_col[val_ws][0]) + "}$")
                str_pr = str_pr.replace("($" + str(min_col[val_ws][1]) + "$)", "($\\mathbf{" + str(min_col[val_ws][1]) + "}$)") 
        newstart = start_of_table.replace("METRICNAME", metric_name_use).replace("VARNAME_", varname.replace(" ", "_") + "_").replace("NRMSE ", "NRMSE (\%) ").replace("R2 ", "$R^{2}$ (\%) ").replace("VARNAME", translate_varname[varname])
        if "R2" not in metric_name_use and "NRMSE" not in metric_name_use and mul_metric != 0:
            newstart = newstart.replace(metric_name_use + " ", metric_name_use + " ($\\times 10^{-" + str(mul_metric) + "}$) ")
        if "wt" in metric_name_use:
            newstart = newstart.replace("_wt", "").replace("trajectories", "trajectories and time stamps")
        print(newstart.replace("Euclid ", "The Euclidean distance "))
        print(first_line + " \\\\ \\hline")
        print(str_pr + end_of_table)
        for e in errs:
            total_errs.add(e)

errs2 = set()
errs3 = set()
errs4 = dict()
for e in total_errs:
    errs2.add((e[1], e[2]))
    errs3.add((e[0], e[1], e[2]))
    if (e[0], e[1], e[2], e[3]) not in errs4:
        errs4[(e[0], e[1], e[2], e[3])] = set()
    errs4[(e[0], e[1], e[2], e[3])].add((e[4], e[5]))
#print(errs4)

for model_name_use_val_ws in errs2:
    model_name_use, val_ws = model_name_use_val_ws
    print(model_name_use, val_ws)
    for varname in dicti_all_traj_latest_avg:
        #if (varname, model_name_use, val_ws) not in errs3:
            #continue
        for metric_name_use in dicti_all_traj_latest_avg[varname][model_name_use][val_ws]:
            if (varname, model_name_use, val_ws, metric_name_use) not in errs4:
                continue
            print(varname, metric_name_use, dicti_all_traj_latest_avg[varname][model_name_use][val_ws][metric_name_use], dicti_all_traj_latest_std[varname][model_name_use][val_ws][metric_name_use])
            for v in errs4[(varname, model_name_use, val_ws, metric_name_use)]:
                print(v)
            #print(dicti_all_traj_latest_short[varname][model_name_use][val_ws][metric_name_use])
    for varname in dicti_all_latest_avg:
        #if (varname, model_name_use, val_ws) not in errs3:
            #continue
        for metric_name_use in dicti_all_latest_avg[varname][model_name_use][val_ws]:
            if (varname, model_name_use, val_ws, metric_name_use) not in errs4:
                continue
            print(varname, metric_name_use, dicti_all_latest_avg[varname][model_name_use][val_ws][metric_name_use], dicti_all_latest_std[varname][model_name_use][val_ws][metric_name_use])
            for v in errs4[(varname, model_name_use, val_ws, metric_name_use)]:
                print(v)
            #print(dicti_all_latest_short[varname][model_name_use][val_ws][metric_name_use])