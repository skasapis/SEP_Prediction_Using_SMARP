import csv
import os
import sys
from datetime import datetime, timedelta
from SMARPMatching_functions import dataset_split, evaluation, normalization_v3, angular_distance
from SMARPMatching_functions import ar_area, coordinate_transformation, flare_intensity_translation
from statistics_functions import svm_data_scatter_v2, data_histograms, all_data_histograms, d1_histograms_v3, d2_histograms_v3, t_test, cumulative_metrics
from csv import writer
from csv import reader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import svm, datasets
import xlsxwriter



# All data contain: USFLUX,MEANGBZ,LATDTMAX,LATDTMIN,LONDTMAX,LONDTMIN,location,class
# Open and process SEP matched TARP nums list
with open("sep_data_v4.txt", "rb") as pd:   # Unpickling
    positive_data = pickle.load(pd)
    positive_data_processed = []
    for dl in positive_data:
        flux = float(dl[0])
        gradient = float(dl[1])
        area = ar_area(float(dl[2]),float(dl[3]),float(dl[4]),float(dl[5]))
        angular_dist_ar = angular_distance( ((float(dl[2])+float(dl[3]))/2,(float(dl[4])+float(dl[5]))/2) ) # angular distance of the center of the ar
        angular_dist_flare = angular_distance( coordinate_transformation(dl[6]) ) # angular distance of the flare location
        intensity = flare_intensity_translation(dl[7]) # convert intensities to numbers
        positive_data_processed.append([flux,gradient,angular_dist_ar,angular_dist_flare,intensity,area])
    print('SEP Data Processed')
    positive_data = positive_data_processed

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
with open("active_peak_data_v4.txt", "rb") as apd:   # Unpickling
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
            active_data_processed.append([flux,gradient,angular_dist_ar,angular_dist_flare,intensity,area])
    print('Active AR Data Processed')
    active_data = active_data_processed

# Type of normalization
norm_dict = ['Log','Norm','Norm','Norm','Log','Norm'] #Choose between: '','Log','BoxCox','Norm' and 0:Flux, 1:Grad, 2:ARDist, 3:FlareDist, 4:Intensity, 5: AR Area
positive_data, active_data = normalization_v3(positive_data, active_data, norm_dict)

predictors_dict = {'0':'Flux', '1':'Grad', '2':'ARDist', '3':'FlareDist', '4':'Intensity', '5':'AR Area'}

# Do multiple runs
runs_number = 100 # Number of different runs
deg = 3 # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
kernel = 'poly' #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
Cs = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2] #, 5, 7.5, 10, 15, 20]
Gs = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2] #, 5, 7.5, 10, 15, 20]
Ps = [[2,5]]#[[0,2],[0,3],[3,5],[0,3,5],[0,2,5],[0,1,2,5],[0,5],[4,0],[4,1],[4,2],[4,3],[4,5],[4,0,3],[4,0,5],[4,0,3,5]] 


for predictors in Ps: 

    file_name = kernel
    for i in predictors:
        file_name = file_name + '_' + predictors_dict[str(i)]
    print(file_name)
    
    table = [[0]+Cs] #initiate table
    c = 0

    for C in Cs:
        c += 1
        g = 0
        print(c)

        run_acc = [C]

        for gamma in Gs:
            g += 1
            print(g)

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
                X_train = np.concatenate((positive_train, active_peak_train), axis=0); X_train = X_train[:,predictors]
                X_val = np.concatenate((positive_val, active_peak_val), axis=0); X_val = X_val[:,predictors]

                Y_train = np.concatenate((np.ones(len(positive_train), dtype = int), np.zeros(len(active_peak_train), dtype = int)))
                Y_val = np.concatenate((np.ones(len(positive_val), dtype = int), np.zeros(len(active_peak_val), dtype = int)))

                #print(X_train,X_val)
                #print(np.shape(X_train),np.shape(X_val))

                ################### Create the Support Vector Machine model ####################
                
                # Create SVM instance and fit out data
                model = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree = deg).fit(X_train, Y_train)

                print('Accuracy for run {} is: {}'.format(run,model.score(X_val,Y_val)))
                output_pred = model.predict(X_val)

                # Accumulate Data
                if run == 0:
                    cumulative_true = Y_val.astype(int)
                    cumulative_pred = output_pred
                else:
                    cumulative_true = np.vstack((cumulative_true, Y_val.astype(int)))
                    cumulative_pred = np.vstack((cumulative_pred, output_pred))

            if runs_number > 1:
                cumulative_accuracy = cumulative_metrics(cumulative_pred, cumulative_true, runs_number)

            run_acc.append(cumulative_accuracy)
        table.append(run_acc)

    with xlsxwriter.Workbook('SEP_results/SVM/'+file_name+'.xlsx') as workbook: #with xlsxwriter.Workbook('SEP_results/SVM/'+file_name+str(deg)+'.xlsx') as workbook:
        worksheet = workbook.add_worksheet()
        
        for row_num, data in enumerate(table):
            worksheet.write_row(row_num, 0, data)

