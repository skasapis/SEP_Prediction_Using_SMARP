import csv
import os
import sys
from datetime import datetime, timedelta
from SMARPMatching_functions import dataset_split, evaluation, normalization_v3, angular_distance, simple_1d_prediction
from SMARPMatching_functions import ar_area, coordinate_transformation, flare_intensity_translation
from statistics_functions import svm_data_scatter_v2, data_histograms, all_data_histograms, d1_histograms_v3, d2_histograms_v3, t_test, cumulative_metrics
from csv import writer
from csv import reader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn import svm



# All data contain: USFLUX,MEANGBZ,LATDTMAX,LATDTMIN,LONDTMAX,LONDTMIN,location,class
# Open and process SEP matched TARP nums list
with open("sep_data_v5.txt", "rb") as pd:   # Unpickling
    positive_data = pickle.load(pd)
    positive_data_processed = []
    for dl in positive_data:
        flux = float(dl[0])
        gradient = float(dl[1])
        area = ar_area(float(dl[2]),float(dl[3]),float(dl[4]),float(dl[5]))
        angular_dist_ar = angular_distance( ((float(dl[2])+float(dl[3]))/2,(float(dl[4])+float(dl[5]))/2) ) # angular distance of the center of the ar
        angular_dist_flare = angular_distance( coordinate_transformation(dl[6]) ) # angular distance of the flare location
        intensity = flare_intensity_translation(dl[7]) # convert intensities to numbers
        r_value = float(dl[8])
        positive_data_processed.append([flux,gradient,angular_dist_ar,angular_dist_flare,intensity,area,r_value])
    print('SEP Data Processed: {} datapoints'.format(len(positive_data_processed)))
    positive_data = positive_data_processed

# R_value - PICK THE LAST NON-ZERO R_VALUE INSTEAD
j = 0
new_positive_data = []
for i in positive_data:
    if i[6] == float(0):
        j += 1
    else:
        new_positive_data.append(i)
print(j/len(positive_data))
positive_data = new_positive_data

# Open and process quiet TARP nums list
with open("negative_sep_data_v3.txt", "rb") as apd:   # Unpickling
    quiet_data = pickle.load(apd)
    quiet_data_processed = []
    for dl in quiet_data:
        flux = float(dl[0])
        gradient = float(dl[1])
        angular_dist_ar = angular_distance( ((float(dl[2])+float(dl[3]))/2,(float(dl[4])+float(dl[5]))/2) ) # angular distance of the center of the ar
        angular_dist_flare = -1 # data not assigned to flares
        intensity = -1 # data not assigned to flares
        quiet_data_processed.append([flux,gradient,angular_dist_ar,angular_dist_flare,intensity])
    quiet_data = quiet_data_processed
    print('Quiet Data Processed')

# Open and process quiet TARP nums list
with open("active_peak_data_v5.txt", "rb") as apd:   # Unpickling
    active_data = pickle.load(apd)
    active_data_processed = []
    for dl in active_data:
        if '' not in dl: # get rid of all data that are missing a parameter
            flux = float(dl[0])
            gradient = float(dl[1])
            area = ar_area(float(dl[2]),float(dl[3]),float(dl[4]),float(dl[5]))
            angular_dist_ar = angular_distance( ((float(dl[2])+float(dl[3]))/2,(float(dl[4])+float(dl[5]))/2) ) # angular distance of the center of the ar
            coords = list(dl[6].split(",")) #split the string and get coordinates
            angular_dist_flare = angular_distance( (float(coords[1][0:-1]),float(coords[0][1:])) ) # GOES DATASET HAS LAT AND LON INVERTED!!! angular distance of the flare location
            intensity = flare_intensity_translation(dl[7]) # convert intensities to numbers
            r_value = float(dl[8])
            active_data_processed.append([flux,gradient,angular_dist_ar,angular_dist_flare,intensity,area,r_value])
    print('Active AR Data Processed: {} datapoints'.format(len(active_data_processed)))
    active_data = active_data_processed

# R_value
j = 0
new_active_data = []
for i in active_data:
    if i[6] == float(0):
        j += 1
    else:
        new_active_data.append(i)
print(j/len(active_data))
active_data = new_active_data

print(len(active_data))

#sys.exit()

# Type of normalization
norm_dict = ['Log','Norm','Norm','Norm','Log','Norm','Norm'] #Choose between: '','Log','BoxCox','Norm' and 0:Flux, 1:Grad, 2:ARDist, 3:FlareDist, 4:Intensity, 5: AR Area
positive_data, active_data = normalization_v3(positive_data, active_data, norm_dict)

# 1-D Histograms
#d1_histograms_v3(positive_data, active_data, quiet_data)

# 2-D Histograms
#d2_histograms_v3(positive_data, active_data, quiet_data)

# T-Tests
#t_test(positive_data, active_data, quiet_data)

#sys.exit()

# Do multiple runs
runs_number = 100 # Number of different runs

for run in range(runs_number):

    # Randomly select some of all the active data
    select_active_peak_data = active_data[np.random.randint(0,active_data.shape[0],len(positive_data))] 

    # Data normalization
    select_positive_data = positive_data

    # Split Datasets
    positive_train, positive_val = dataset_split(select_positive_data) # Positive Dataset
    # The points are selected randomly but from TARPs that produced flares
    active_peak_train, active_peak_val = dataset_split(select_active_peak_data) # Active Peak Dataset
    # Then points are selected from the TARPs that produced flares and the values correspond to the peak flare values.
    
    # Plot randomly selected data that will be used for this current SVM run
    #svm_data_scatter_v2(positive_train,positive_val,active_peak_train,active_peak_val)

    ################### Combine Datasets ########################
    X_train = np.concatenate((positive_train, active_peak_train), axis=0); X_train = X_train[:,[0,3]]
    X_val = np.concatenate((positive_val, active_peak_val), axis=0); X_val = X_val[:,[0,3]]

    Y_train = np.concatenate((np.ones(len(positive_train), dtype = int), np.zeros(len(active_peak_train), dtype = int)))
    Y_val = np.concatenate((np.ones(len(positive_val), dtype = int), np.zeros(len(active_peak_val), dtype = int)))

    # Simple 1D Prediction
    #simple_1d_prediction(X_train, Y_train, X_val, Y_val)

    ################### Create Linear Regression Model ####################
    
    C = 0.05
    threshold = 0.5
    a = 0.5
    clf = LogisticRegression(C = C).fit(X_train, Y_train) # C & tolerance
    #clf = Lasso().fit(X_train, Y_train)
    clf = Ridge(alpha = a).fit(X_train, Y_train) #alpha & tolerance
    output_pred = clf.predict(X_val)
    
    for i in range(len(output_pred)):
        if output_pred[i] >= threshold:
            output_pred[i] = 1
        else:
            output_pred[i] = 0
    
    #print('Accuracy for run {} is: {}'.format(run,sum(output_pred)/len(output_pred)))
    
    # Accumulate Data
    if run == 0:
        cumulative_true = Y_val.astype(int)
        cumulative_pred = output_pred
    else:
        cumulative_true = np.vstack((cumulative_true, Y_val.astype(int)))
        cumulative_pred = np.vstack((cumulative_pred, output_pred))

if runs_number > 1:
    ACC, HSS, TSS = cumulative_metrics(cumulative_pred, cumulative_true, runs_number)
    print('ACC:',ACC)
    print('HSS:',HSS)
    print('TSS:',TSS)



    



    
