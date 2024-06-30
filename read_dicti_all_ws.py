from utilities import load_object
import numpy as np

dicti_all = load_object("dicti_all_ws")
ord_metric = ["UniTS_longlat_speed_direction"]
metric_dicti = {"NRMSE": 2, "R2": 2, "MAE": 0, "RMSE": 0}
list_ws = [2, 3, 4, 5, 10, 20, 30] 

sf1, sf2 = 5, 5
for metric_name_use in list(metric_dicti.keys()):
    for model_name_use in ord_metric:
        for nf1 in range(sf1):
            for nf2 in range(sf2):
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
                    first_line = metric_name_use + " " + model_name_use + " test " + str(nf1 + 1) + " val " + str(nf2 + 1) + " 10^{" + str(mul_metric) + "} " + str(rv_metric)
                    for varname in dicti_all[nf1][nf2]:
                        for val_ws in list_ws:
                            first_line += " & $" + str(val_ws) + "$s"
                        break
                    for varname in dicti_all[nf1][nf2]:
                        str_pr += varname
                        for val_ws in list_ws:
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
                if "R2" in metric_name_use:
                    for val_ws in list_ws:
                        str_pr = str_pr.replace("$" + str(max_col[val_ws]) + "$", "$\\mathbf{" + str(max_col[val_ws]) + "}$") 
                else:
                    for val_ws in list_ws:
                        str_pr = str_pr.replace("$" + str(min_col[val_ws]) + "$", "$\\mathbf{" + str(min_col[val_ws]) + "}$") 
                #print(first_line + " \\\\ \\hline")
                #print(str_pr)

metrictouse = ["MAE", "R2"]
vartouse = ["direction", "speed", "longitude_no_abs", "latitude_no_abs"]
translate_varname = {"direction": "heading", "speed": "speed", "longitude_no_abs": "$x$ offset", "latitude_no_abs": "$y$ offset", "time": "time intervals"}
translate_num = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
start_of_table = "\\begin{table}[!t]\n\t\\begin{center}\n\t\t\\caption{METRICNAME for the VARNAME estimated on the USENUMBER k-fold testing dataset by the UniTS model using different forecasting times, and k-fold validation datasets.}\n\t\t\\label{tab:k-fold_USENUMBER_test_VARNAME_METRICNAME}\n\t\t\\resizebox{\linewidth}{!}{"
end_of_table = "\t\t\\end{tabular}}\n\t\\end{center}\n\\end{table}\n"
for metric_name_use in metrictouse:
    for varname in vartouse:
        for nf1 in range(sf1):
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
                    for nf2 in range(sf2):
                        str_pr += "\t\t\t Val " + str(nf2 + 1)
                        for val_ws in list_ws: 
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
            if "R2" in metric_name_use:
                for val_ws in list_ws:
                    str_pr = str_pr.replace("$" + str(max_col[val_ws]) + "$", "$\\mathbf{" + str(max_col[val_ws]) + "}$") 
            else:
                for val_ws in list_ws:
                    str_pr = str_pr.replace("$" + str(min_col[val_ws]) + "$", "$\\mathbf{" + str(min_col[val_ws]) + "}$")
            newstart = start_of_table.replace("USENUMBER", translate_num[nf1 + 1]).replace("METRICNAME", metric_name_use).replace("VARNAME_", varname + "_").replace("NRMSE ", "NRMSE (\%) ").replace("R2 ", "$R^{2}$ (\%) ").replace("VARNAME ", translate_varname[varname] + " ")
            if "R2" not in metric_name_use and "NRMSE" not in metric_name_use and mul_metric != 0:
                newstart = newstart.replace(metric_name_use + " ", metric_name_use + " ($\\times 10^{-" + str(mul_metric) + "}$) ")
            #print(newstart)
            #print(first_line + " \\\\ \\hline")
            #print(str_pr + end_of_table)
            
metrictouse = ["MAE", "R2"]
vartouse = ["direction", "speed", "longitude_no_abs", "latitude_no_abs"]
translate_varname = {"direction": "heading", "speed": "speed", "longitude_no_abs": "$x$ offset", "latitude_no_abs": "$y$ offset", "time": "time intervals"}
translate_num = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
start_of_table = "\\begin{table}[!t]\n\t\\begin{center}\n\t\t\\caption{METRICNAME for the VARNAME estimated on the k-fold testing datasets by the UniTS model using different forecasting times, and k-fold validation datasets.}\n\t\t\\label{tab:k-fold_test_VARNAME_METRICNAME}\n\t\t\\resizebox{\linewidth}{!}{"
end_of_table = "\t\t\\end{tabular}}\n\t\\end{center}\n\\end{table}\n"
for metric_name_use in metrictouse:
    for varname in vartouse:
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
                        str_pr += "\t\t\t Test " + str(nf1 + 1) + " Val " + str(nf2 + 1)
                        for val_ws in list_ws: 
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
        if "R2" in metric_name_use:
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(max_col[val_ws]) + "$", "$\\mathbf{" + str(max_col[val_ws]) + "}$") 
        else:
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(min_col[val_ws]) + "$", "$\\mathbf{" + str(min_col[val_ws]) + "}$")
        newstart = start_of_table.replace("METRICNAME", metric_name_use).replace("VARNAME_", varname + "_").replace("NRMSE ", "NRMSE (\%) ").replace("R2 ", "$R^{2}$ (\%) ").replace("VARNAME ", translate_varname[varname] + " ")
        if "R2" not in metric_name_use and "NRMSE" not in metric_name_use and mul_metric != 0:
            newstart = newstart.replace(metric_name_use + " ", metric_name_use + " ($\\times 10^{-" + str(mul_metric) + "}$) ")
        #print(newstart)
        #print(first_line + " \\\\ \\hline")
        #print(str_pr + end_of_table)

metrictouse = ["MAE", "R2"]
vartouse = ["direction", "speed", "longitude_no_abs", "latitude_no_abs"]
translate_varname = {"direction": "heading", "speed": "speed", "longitude_no_abs": "$x$ offset", "latitude_no_abs": "$y$ offset", "time": "time intervals"}
translate_num = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
start_of_table = "\\begin{table}[!t]\n\t\\begin{center}\n\t\t\\caption{The average METRICNAME across k-fold validation datasets, with standard deviation in brackets, for the VARNAME estimated on the k-fold testing datasets by the UniTS model using different forecasting times.}\n\t\t\\label{tab:avg_k-fold_test_VARNAME_METRICNAME}\n\t\t\\resizebox{\linewidth}{!}{"
end_of_table = "\t\t\\end{tabular}}\n\t\\end{center}\n\\end{table}\n"
for metric_name_use in metrictouse:
    for varname in vartouse:
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
            max_col_std = dict()
            for val_ws in list_ws:
                max_col[val_ws] = -1000000
                max_col_std[val_ws] = -1000000
            min_col = dict()
            min_col_std = dict()
            for val_ws in list_ws:
                min_col[val_ws] = 1000000
                min_col_std[val_ws] = 1000000
            repl_dict = dict()
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
                vvavg_dict = dict()
                for nf1 in range(sf1):
                    vvavg_dict[nf1] = dict()
                    for val_ws in list_ws:
                        vvavg_dict[nf1][val_ws] = []
                        for nf2 in range(sf2):
                            vvavg_dict[nf1][val_ws].append(dicti_all[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use])
                for nf1 in range(sf1):
                    str_pr += "\t\t\t\\multirow{2}{*}{Test " + str(nf1 + 1) + "}"
                    for val_ws in list_ws: 
                        vv = np.mean(vvavg_dict[nf1][val_ws])
                        vv_std_original = np.std(vvavg_dict[nf1][val_ws])
                        vv = np.round(vv * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                        mul_metric_second = 0
                        vv_std = np.round(vv_std_original * (10 ** metric_dicti[metric_name_use]) * (10 ** (mul_metric + mul_metric_second)), 2)
                        while "0." in str(vv_std):
                            mul_metric_second += 1
                            vv_std = np.round(vv_std_original * (10 ** metric_dicti[metric_name_use]) * (10 ** (mul_metric + mul_metric_second)), 2)
                        str_pr += " & $" + str(vv) + "$"
                        if vv in set_values[val_ws]:
                            duplicate_val = True
                        if vv in set_values_all:
                            duplicate_val_all = True
                        if "$0." in str_pr:
                            too_small = True
                        set_values[val_ws].add(vv)
                        set_values_all.add(vv)
                        if mul_metric_second == 0:
                            repl_dict_key = str(vv) + " (" + str(vv_std) + ")"
                            repl_dict[repl_dict_key] = "(" + str(vv_std) + ")"
                        else:
                            repl_dict_key = str(vv) + " (" + str(vv_std) + "\\times 10^{-" + str(mul_metric_second) + "})"
                            repl_dict[repl_dict_key] = "(" + str(vv_std) + "\\times 10^{-" + str(mul_metric_second) + "})"
                        if vv > max_col[val_ws]:
                            max_col[val_ws] = vv
                            max_col_std[val_ws] = repl_dict_key
                        if vv < min_col[val_ws]:
                            min_col[val_ws] = vv
                            min_col_std[val_ws] = repl_dict_key
                    str_pr += " \\\\ \n"
                    str_pr += "\t\t\t"
                    for val_ws in list_ws:
                        vv = np.mean(vvavg_dict[nf1][val_ws])
                        vv_std_original = np.std(vvavg_dict[nf1][val_ws])
                        vv = np.round(vv * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                        mul_metric_second = 0
                        vv_std = np.round(vv_std_original * (10 ** metric_dicti[metric_name_use]) * (10 ** (mul_metric + mul_metric_second)), 2)
                        while "0." in str(vv_std):
                            mul_metric_second += 1
                            vv_std = np.round(vv_std_original * (10 ** metric_dicti[metric_name_use]) * (10 ** (mul_metric + mul_metric_second)), 2)
                        if mul_metric_second == 0:
                            str_pr += " & $" + str(vv) + " (" + str(vv_std) + ")$"
                        else:
                            str_pr += " & $" + str(vv) + " (" + str(vv_std) + "\\times 10^{-" + str(mul_metric_second) + "})$"
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
                str_pr = str_pr.replace("$" + str(max_col[val_ws]) + "$", "$\\mathbf{" + str(max_col[val_ws]) + "}$").replace(max_col_std[val_ws], "\\mathbf{" + repl_dict[max_col_std[val_ws]] + "}") 
        else:
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(min_col[val_ws]) + "$", "$\\mathbf{" + str(min_col[val_ws]) + "}$").replace(min_col_std[val_ws], "\\mathbf{" + repl_dict[min_col_std[val_ws]] + "}") 
        for r1 in repl_dict:
            str_pr = str_pr.replace(r1, repl_dict[r1])
        newstart = start_of_table.replace("METRICNAME", metric_name_use).replace("VARNAME_", varname + "_").replace("NRMSE ", "NRMSE (\%) ").replace("R2 ", "$R^{2}$ (\%) ").replace("VARNAME ", translate_varname[varname] + " ")
        if "R2" not in metric_name_use and "NRMSE" not in metric_name_use and mul_metric != 0:
            newstart = newstart.replace(metric_name_use + " ", metric_name_use + " ($\\times 10^{-" + str(mul_metric) + "}$) ")
        #print(newstart)
        #print(first_line + " \\\\ \\hline")
        #print(str_pr + end_of_table)

start_of_table = "\\begin{table}[!t]\n\t\\begin{center}\n\t\t\\caption{The average METRICNAME across k-fold validation datasets, with standard deviation in brackets, for the variables estimated on the k-fold testing datasets by the UniTS model using different forecasting times.}\n\t\t\\label{tab:avg_k-fold_test_all_METRICNAME}\n\t\t\\resizebox{\linewidth}{!}{"
end_of_table = "\t\t\\end{tabular}}\n\t\\end{center}\n\\end{table}\n"
for metric_name_use in metrictouse:
    newstart = start_of_table.replace("METRICNAME", metric_name_use).replace("NRMSE ", "NRMSE (\%) ").replace("R2 ", "$R^{2}$ (\%) ")
    if "R2" not in metric_name_use and "NRMSE" not in metric_name_use and mul_metric != 0:
        newstart = newstart.replace(metric_name_use + " ", metric_name_use + " ($\\times 10^{-" + str(mul_metric) + "}$) ")
    print(newstart)
    print(first_line + " \\\\ \\hline")
    for varname in vartouse:
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
                vvavg_dict = dict()
                for val_ws in list_ws:
                    vvavg_dict[val_ws] = []
                    for nf2 in range(sf2):
                        vvavg_dict[val_ws].append(dicti_all[nf1][nf2][varname][model_name_use][str(val_ws)][metric_name_use])
                if mul_metric != 0:
                    str_pr += "\t\t\t\\multirow{2}{*}{"
                else:
                    str_pr += "\t\t\t\\multirow{3}{*}{"
                str_pr += translate_varname[varname].capitalize() + "}"
                for val_ws in list_ws: 
                    vv = np.mean(vvavg_dict[val_ws])
                    vv_std_original = np.std(vvavg_dict[val_ws])
                    vv = np.round(vv * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                    mul_metric_second = 0
                    vv_std = np.round(vv_std_original * (10 ** metric_dicti[metric_name_use]) * (10 ** (mul_metric + mul_metric_second)), 2)
                    while "0." in str(vv_std):
                        mul_metric_second += 1
                        vv_std = np.round(vv_std_original * (10 ** metric_dicti[metric_name_use]) * (10 ** (mul_metric + mul_metric_second)), 2)
                    str_pr += " & $" + str(vv) + "$"
                    if vv in set_values[val_ws]:
                        duplicate_val = True
                    if vv in set_values_all:
                        duplicate_val_all = True
                    if "$0." in str_pr:
                        too_small = True
                    set_values[val_ws].add(vv)
                    set_values_all.add(vv)
                str_pr += " \\\\ \n"
                str_pr += "\t\t\t"
                for val_ws in list_ws:
                    vv = np.mean(vvavg_dict[val_ws])
                    vv_std_original = np.std(vvavg_dict[val_ws])
                    vv = np.round(vv * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                    mul_metric_second = 0
                    vv_std = np.round(vv_std_original * (10 ** metric_dicti[metric_name_use]) * (10 ** (mul_metric + mul_metric_second)), 2)
                    while "0." in str(vv_std):
                        mul_metric_second += 1
                        vv_std = np.round(vv_std_original * (10 ** metric_dicti[metric_name_use]) * (10 ** (mul_metric + mul_metric_second)), 2)
                    if mul_metric_second == 0:
                        str_pr += " & $(" + str(vv_std) + ")$"
                    else:
                        str_pr += " & $(" + str(vv_std) + "$"
                str_pr += " \\\\ \n"
                str_pr += "\t\t\t"
                if mul_metric != 0:
                    str_pr += " ($\\times 10^{-" + str(mul_metric) + "}$)"
                for val_ws in list_ws:
                    vv = np.mean(vvavg_dict[val_ws])
                    vv_std_original = np.std(vvavg_dict[val_ws])
                    vv = np.round(vv * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                    mul_metric_second = 0
                    vv_std = np.round(vv_std_original * (10 ** metric_dicti[metric_name_use]) * (10 ** (mul_metric + mul_metric_second)), 2)
                    while "0." in str(vv_std):
                        mul_metric_second += 1
                        vv_std = np.round(vv_std_original * (10 ** metric_dicti[metric_name_use]) * (10 ** (mul_metric + mul_metric_second)), 2)
                    if mul_metric_second != 0:
                        str_pr += " & $\\times 10^{-" + str(mul_metric_second) + "})$"
                    else:
                        str_pr += " & "
                str_pr += " \\\\ \\hline"
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
        print(str_pr)
    print(end_of_table)