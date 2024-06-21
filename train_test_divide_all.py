from utilities import load_object, save_object
import os
from sklearn.model_selection import train_test_split, StratifiedKFold

all_subdirs = os.listdir() 
num_occurences_of_direction = dict()
num_occurences_of_direction_diff = dict()
num_occurences_of_direction_in_next_step = dict()
num_occurences_of_direction_in_next_next_step = dict()

all_good_rides = dict()
train_all = 0
val_all = 0
test_all = 0

X_files = []
Y_files = []
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
        
    for some_file in all_files:  
        if subdir_name + "/cleaned_csv/" + some_file not in bad_rides_filenames and subdir_name + "/cleaned_csv/" + some_file not in gap_rides_filenames:
            X_files.append(some_file)
            Y_files.append(subdir_name)

nf1 = 5
nf2 = 5
skf1 = StratifiedKFold(n_splits = nf1, random_state = 42, shuffle = True)
skf2 = StratifiedKFold(n_splits = nf2, random_state = 42, shuffle = True)
X_train_val, X_test, Y_train_val, Y_test = dict(), dict(), dict(), dict()
X_train, X_val, Y_train, Y_val = dict(), dict(), dict(), dict()
for i, (train_val_index, test_index) in enumerate(skf1.split(X_files, Y_files)):
    X_train_val[i], X_test[i], Y_train_val[i], Y_test[i] = [], [], [], []
    X_train[i], X_val[i], Y_train[i], Y_val[i] = dict(), dict(), dict(), dict()
    for train_val_ix in train_val_index:
        X_train_val[i].append(X_files[train_val_ix])
        Y_train_val[i].append(Y_files[train_val_ix])
    for test_ix in test_index:
        X_test[i].append(X_files[test_ix])
        Y_test[i].append(Y_files[test_ix])
    for j, (train_index, val_index) in enumerate(skf2.split(X_train_val[i], Y_train_val[i])):
        X_train[i][j], X_val[i][j], Y_train[i][j], Y_val[i][j] = [], [], [], []
        for train_ix in train_index:
            X_train[i][j].append(X_train_val[i][train_ix])
            Y_train[i][j].append(Y_train_val[i][train_ix])
        for val_ix in val_index:
            X_val[i][j].append(X_train_val[i][val_ix])
            Y_val[i][j].append(Y_train_val[i][val_ix])

for i in range(nf1):
    for j in range(nf2):
        train_all, val_all, test_all = 0, 0, 0
        for subdir_name in all_subdirs:
            if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
                continue
            X_train_subset, X_val_subset, X_test_subset = [], [], []
            for ix_train in range(len(X_train[i][j])):
                if Y_train[i][j][ix_train] == subdir_name:
                    X_train_subset.append(X_train[i][j][ix_train])
            save_object(subdir_name + "/train_rides_" + str(i + 1) + "_" + str(j + 1), X_train_subset)
            for ix_val in range(len(X_val[i][j])):
                if Y_val[i][j][ix_val] == subdir_name:
                    X_val_subset.append(X_val[i][j][ix_val])
            save_object(subdir_name + "/val_rides_" + str(i + 1) + "_" + str(j + 1), X_val_subset)
            for ix_test in range(len(X_test[i])):
                if Y_test[i][ix_test] == subdir_name:
                    X_test_subset.append(X_test[i][ix_test])
            save_object(subdir_name + "/test_rides_" + str(i + 1), X_test_subset)
            print(subdir_name, len(X_train_subset), len(X_val_subset), len(X_train_subset) + len(X_val_subset), len(X_test_subset), len(X_train_subset) + len(X_val_subset) + len(X_test_subset))
            train_all += len(X_train_subset)
            val_all += len(X_val_subset)
            test_all += len(X_test_subset)
        print(train_all, val_all, train_all + val_all, test_all, train_all + val_all + test_all)
        print(train_all / (train_all + val_all + test_all), val_all / (train_all + val_all + test_all), test_all / (train_all + val_all + test_all))