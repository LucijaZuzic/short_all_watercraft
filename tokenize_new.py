import pandas as pd
import os
import numpy as np
from pytorch_utilities import get_XY 
from utilities import load_object
import numpy as np

def my_token(xv, yv, name_file):
    
    strpr = "x>y\n"

    file_processed = open(name_file, "w")
    file_processed.write(strpr)
    file_processed.close()

    dicti_vals = {"x": [], "y": []}
    for ix1 in range(len(xv)): 

        v1 = str(xv[ix1]).strip()
        v2 = str(yv[ix1]).strip()

        while "  " in v1:
            v1 = v1.replace("  ", " ")

        while "  " in v2:
            v2 = v2.replace("  ", " ")

        dicti_vals["x"].append(v1.replace("[", "").replace("]", "").replace(".", "a").replace(",", "a"))
        dicti_vals["y"].append(v2.replace("[", "").replace("]", "").replace(".", "a").replace(",", "a"))

    df_new = pd.DataFrame(dicti_vals)

    df_new.to_csv(name_file, index = False, sep = ">") 

ws_range = [2, 3, 4, 5, 10, 20, 30]

sf1, sf2 = 5, 5
for nf1 in range(sf1):
    for nf2 in range(sf2):
        resave = True
        if resave:
            for filename in os.listdir("actual_train/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):

                varname = filename.replace("actual_train_", "")
                
                if "time" in varname:
                    continue

                file_object_train = load_object("actual_train/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_train_" + varname)
                file_object_val = load_object("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_val_" + varname)
                file_object_test = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_" + varname)

                for ws_use in ws_range:

                    if os.path.isfile("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_train_" + str(ws_use) + ".csv"):
                        continue
                    
                    x_train_all = []
                    y_train_all = []

                    for k in file_object_train:

                        x_train_part, y_train_part = get_XY(file_object_train[k], ws_use, 1, ws_use)
                        
                        for ix in range(len(x_train_part)):
                            x_train_all.append(x_train_part[ix]) 
                            y_train_all.append(y_train_part[ix])

                    x_train_all = np.array(x_train_all)
                    y_train_all = np.array(y_train_all)

                    if not os.path.isdir("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname):
                        os.makedirs("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname)

                    my_token(x_train_all, y_train_all, "tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_train_" + str(ws_use) + ".csv")

                    print(nf1 + 1, nf2 + 1, varname, np.shape(x_train_all))

                for ws_use in ws_range:

                    if os.path.isfile("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_train_short_" + str(ws_use) + ".csv"):
                        continue

                    x_train_all_short = []
                    y_train_all_short = []

                    for k in file_object_train:

                        x_train_part, y_train_part = get_XY(file_object_train[k], ws_use, ws_use, ws_use)
                        
                        for ix in range(len(x_train_part)):
                            x_train_all_short.append(x_train_part[ix]) 
                            y_train_all_short.append(y_train_part[ix])

                    x_train_all_short = np.array(x_train_all_short)
                    y_train_all_short = np.array(y_train_all_short)
                    
                    if not os.path.isdir("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname):
                        os.makedirs("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname)

                    my_token(x_train_all_short, y_train_all_short, "tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_train_short_" + str(ws_use) + ".csv")
                    
                    print(nf1 + 1, nf2 + 1, varname, np.shape(x_train_all_short))
                    
                for ws_use in ws_range:

                    if os.path.isfile("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_test_" + str(ws_use) + ".csv"):
                        continue

                    x_test_all = []
                    y_test_all = []

                    for k in file_object_test:

                        x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, 1, ws_use)
                        
                        for ix in range(len(x_test_part)):
                            x_test_all.append(x_test_part[ix]) 
                            y_test_all.append(y_test_part[ix])

                    x_test_all = np.array(x_test_all)
                    y_test_all = np.array(y_test_all)
                    
                    if not os.path.isdir("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname):
                        os.makedirs("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname)

                    my_token(x_test_all, y_test_all, "tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_test_" + str(ws_use) + ".csv")

                    print(nf1 + 1, nf2 + 1, varname, np.shape(x_test_all))
                    
                for ws_use in ws_range:

                    if os.path.isfile("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_test_short_" + str(ws_use) + ".csv"):
                        continue

                    x_test_all_short = []
                    y_test_all_short = []

                    for k in file_object_test:

                        x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, ws_use, ws_use)
                        
                        for ix in range(len(x_test_part)):
                            x_test_all_short.append(x_test_part[ix]) 
                            y_test_all_short.append(y_test_part[ix])

                    x_test_all_short = np.array(x_test_all_short)
                    y_test_all_short = np.array(y_test_all_short)
                    
                    if not os.path.isdir("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname):
                        os.makedirs("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname)

                    my_token(x_test_all_short, y_test_all_short, "tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_test_short_" + str(ws_use) + ".csv")

                    print(nf1 + 1, nf2 + 1, varname, np.shape(x_test_all_short))

                for ws_use in ws_range:

                    if os.path.isfile("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv"):
                        continue

                    x_val_all = []
                    y_val_all = []

                    for k in file_object_val:

                        x_val_part, y_val_part = get_XY(file_object_val[k], ws_use, 1, ws_use)
                        
                        for ix in range(len(x_val_part)):
                            x_val_all.append(x_val_part[ix]) 
                            y_val_all.append(y_val_part[ix])

                    x_val_all = np.array(x_val_all)
                    y_val_all = np.array(y_val_all)

                    if not os.path.isdir("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname):
                        os.makedirs("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname)

                    my_token(x_val_all, y_val_all, "tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv")

                    print(nf1 + 1, nf2 + 1, varname, np.shape(x_val_all))
                    
                for ws_use in ws_range:

                    if os.path.isfile("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_val_short_" + str(ws_use) + ".csv"):
                        continue

                    x_val_all_short = []
                    y_val_all_short = []

                    for k in file_object_val:

                        x_val_part, y_val_part = get_XY(file_object_val[k], ws_use, ws_use, ws_use)
                        
                        for ix in range(len(x_val_part)):
                            x_val_all_short.append(x_val_part[ix]) 
                            y_val_all_short.append(y_val_part[ix])

                    x_val_all_short = np.array(x_val_all_short)
                    y_val_all_short = np.array(y_val_all_short)

                    if not os.path.isdir("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname):
                        os.makedirs("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname)

                    my_token(x_val_all_short, y_val_all_short, "tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_val_short_" + str(ws_use) + ".csv")

                    print(nf1 + 1, nf2 + 1, varname, np.shape(x_val_all_short))