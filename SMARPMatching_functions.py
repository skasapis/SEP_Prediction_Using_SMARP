
# Functions for SMARP Matching
from datetime import datetime, timedelta
import sys
import csv
import os
import random
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import Normalizer
import seaborn
import numpy as np



# Function that gets the necessary information for each AR 
def get_ar_info(csv_smarp):

    i = 0
    for row in csv_smarp: # Iterate through active region rows

        # In first iteration
        if i == 0: 
            noaa_ar_number = int(row['NOAA_AR']) # Get NOAA number from the first iteration
            tarp_num = int(row['TARPNUM']) # Get TARP number from the first iteration
            smarp_start_t = row['T_REC'] # Active Region Starting Time               
        
        smarp_end_t = row['T_REC'] # Ending time on last iteration  (How about DATE_OBS and T_OBS?)

        # Error Messages
        if int(row['TARPNUM']) != tarp_num: # NOAA and TARP numbers should not be changing in a file
            sys.exit('Error in TARP {}: Change in TARP number'.format(row['TARPNUM']))
        noaa_ar_number = int(row['NOAA_AR'])
        if int(row['NOAA_AR']) != noaa_ar_number: # Print errors if they do
            sys.exit('Error in TARP {}: Change in NOAA AR number'.format(row['TARPNUM']))
        tarp_num = int(row['TARPNUM'])

        i += 1

    # Convert to datetime objects
    smarp_start_t = datetime.strptime(smarp_start_t[:-4], '%Y.%m.%d_%H:%M:%S')
    smarp_end_t = datetime.strptime(smarp_end_t[:-4], '%Y.%m.%d_%H:%M:%S')

    return smarp_start_t, smarp_end_t, noaa_ar_number, tarp_num



# Function to check whether we get an areal match 
def check_ar_boundaries(flare_location, file_name):

    if flare_location != '(0, 0)':

        # Open TARP file again cause it had problem iterating csv_smarp
        with open('smarp-header/'+file_name, mode='r') as SMARP_file:  
            csv_smarp = csv.DictReader(SMARP_file)

            # Initialization
            flare_location = eval(flare_location)
            lon = float(flare_location[1]); lat = float(flare_location[0])
            lon_bool = False; lat_bool = False; match = False 

            # Loop through all rows
            for row in csv_smarp:
                if (float(row['LATDTMIN']) < lat) and (float(row['LATDTMAX']) > lat):
                    lat_bool = True
                if (float(row['LONDTMIN']) < lon) and (float(row['LONDTMAX']) > lon):
                    lon_bool = True
                match = lat_bool and lon_bool
                if match == True:
                    break

        return match



# Function to find which row of the TARP flare peak belongs to
def row_closer_to_peak(flare_peak_time, file_name):

    # Open TARP file again cause it had problem iterating csv_smarp
    with open('smarp-header/'+file_name, mode='r') as SMARP_file:  
        csv_smarp = csv.DictReader(SMARP_file)
        smallest_difference = 9999999 #seconds (115 days - too much)
        closest_row = 0

        # loops through TARP rows
        row_num = 2
        for row in csv_smarp:
            row_time = datetime.strptime(row['T_REC'][:-4], '%Y.%m.%d_%H:%M:%S') # the recorded time of each row
            difference = abs(flare_peak_time - row_time) # timedifference between peak flare time and row
            #difference_pure = flare_peak_time - row_time
            if difference.total_seconds() < smallest_difference:
                smallest_difference = difference.total_seconds() # update the smallest time difference
                #smallest_difference_pure = difference_pure.total_seconds()
                closest_row = row_num #save the row number
                data = (row['USFLUX'],row['MEANGBZ'],row['LATDTMAX'],row['LATDTMIN'],row['LONDTMAX'],row['LONDTMIN'])
            row_num += 1

    return closest_row, row_num, smallest_difference, data



# Coordinate transformation
def coordinate_transformation(nsew_coords):

    if len(nsew_coords) < 6:
        pos_neg_coords = '(0, 0)'
    elif (nsew_coords[0] == 'N' or nsew_coords[0] == 'S') and (nsew_coords[3] == 'E' or nsew_coords[3] == 'W'):
        # Latitude Coordinate Sign
        if nsew_coords[0] == 'N':
            lat_sign = 1
        else:
            lat_sign = -1
        # Londitude Coordinate Sign
        if nsew_coords[3] == 'W':
            lon_sign = 1
        else:
            lon_sign = -1
        # Transform
        pos_neg_coords = (lat_sign*int(nsew_coords[1:3]),lon_sign*int(nsew_coords[4:6]))
        #pos_neg_coords = str(pos_neg_coords)
    else:
        pos_neg_coords = (0, 0)
    return pos_neg_coords



# Function to Split Dataset to test and val
def dataset_split(dataset):
    #random.shuffle(dataset) 
    dataset_size = len(dataset)
    train_size = int(0.9*dataset_size)
    # Randomly select some of all the negative data
    l = [i for i in range(len(dataset))]
    random.shuffle(l)
    l_train = l[0:train_size]
    l_val = l[train_size:]
    train = dataset[l_train]
    val = dataset[l_val]
    return train, val



# Data normalization for logistic_regression.py
def log_reg_normalization(dataset1, dataset2, dataset3, dataset4):
    # Combine dataset to normalize all together
    dataset_full = dataset1 + dataset2 + dataset3 + dataset4
    # Normalize
    normalized_dataset = sklearn.preprocessing.normalize(dataset_full,axis=0)
    # Split dataset again
    normalized_dataset1 = normalized_dataset[0:len(dataset1)]
    normalized_dataset2 = normalized_dataset[len(dataset1):(len(dataset1)+len(dataset2))]
    normalized_dataset3 = normalized_dataset[(len(dataset1)+len(dataset2)):(len(dataset1)+len(dataset2)+len(dataset3))]
    normalized_dataset4 = normalized_dataset[(len(dataset1)+len(dataset2)+len(dataset3)):]
    return normalized_dataset1, normalized_dataset2, normalized_dataset3, normalized_dataset4



# Function for evaluation of prediction
def evaluation(local_labels,output):
    tn, fp, fn, tp = confusion_matrix(local_labels, output).ravel()
    data = [[tp,fp],[fn,tn]]
    conf_fig = plt.figure(1, figsize=(3, 2))
    plt.title("Confusion Matrix"); labels = ['Positive','Negative']
    seaborn.set(color_codes=True); seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu")
    #ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    conf_fig.set_figwidth(6) 
    conf_fig.set_figheight(6)
    #plt.savefig("plots/Confusion{}.png".format(run)) 
    plt.show()
    #print('Other Metrics')
    #print('F1:   {}, {}, {}  (Macro,Micro,Weighted)'.format(round(f1_score(local_labels, output, average='macro'),2), round(f1_score(local_labels, output, average='micro'),2), round(f1_score(local_labels, output, average='weighted'),2)))
    #print('Loss: {}          (Logarithmic)'.format(round(log_loss(local_labels, output,5))))
    #print('AUC:  {}          (ROC)'.format(round(roc_auc_score(local_labels, output,2))))



# Function to calculate the angular distance between the active region and the magnetic footpoint of earth
def angular_distance(coords): 
    
    latdt = coords[0]
    londt = coords[1]
    
    # Convert in radiants before I use in sin and cosine (1 Degree = 0.01745329 Radian)
    theta1 = latdt*0.01745329 # theta_1, phi_1 is the latitude and longitude of the active region
    phi1 = londt*0.01745329
    theta2 = 0*0.01745329 # theta_2, phi_2 is the latitude and longitude of the magnetic foot point of earth (for now constant)
    phi2 = 45*0.01745329

    # angular distance
    dist = np.arccos(np.sin(theta1)*np.sin(theta2) + np.cos(theta1)*np.cos(theta2)*np.cos(phi1-phi2)) #distance is in radiants - I can convert back to degrees
    
    # Add sign if east or west of W45
    add_sign = True
    if add_sign: 
        if londt < 45:
            dist = -dist

    return dist



# Translate the flare class to a number 
def flare_intensity_translation(flare_class):
    
    class_dict = {"A":pow(10,-8),"B":pow(10,-7),"C":pow(10,-6),"M":pow(10,-5),"X":pow(10,-4)}
    if len(flare_class) == 1:
        code = float(class_dict[flare_class[0]])
    else:
        code = float(class_dict[flare_class[0]])*float(flare_class[1:])

    return code



# Check whether flare is in positive
def not_in_positive(peak_time,goes_class):

    switch = True

    peak_time = datetime.strptime(peak_time[:-4], '%Y-%m-%dT%H:%M:%S') # Convert peak time of flare in datetime object
    
    # Open the flares file
    with open('noaa_SEP_labeled_only.csv', mode='r') as flares_file: 
        flares_reader = csv.DictReader(flares_file)

        for flare in flares_reader:
            flare_peak_time = datetime.strptime(flare['flare_peak_time'], '%Y-%m-%d %H:%M:%S') # Convert peak time of flare in datetime object
            if (peak_time == flare_peak_time) and (goes_class[0] == flare['flare_class'][0]):
                switch = False
    
    return switch



# Data normalization for logistic_regression.py
def normalization_v3(dataset1, dataset2, norm_types):

    # Combine dataset and bring to numpy
    dataset_full = np.array(dataset1 + dataset2)
    
    for predictor in range(np.shape(dataset_full)[1]):
        if norm_types[predictor] == 'Norm': # - mean / std
            normalized_predictor = (dataset_full[:,predictor]-np.mean(dataset_full[:,predictor]))/np.std(dataset_full[:,predictor])
            # L-2 Normalization
            #Data_normalizer = Normalizer(norm='l2').fit(dataset_full[:,predictor])
            #normalized_predictor = Data_normalizer.transform(dataset_full[:,predictor])
            for i in range(len(dataset_full)): dataset_full[i,predictor] = normalized_predictor[i]
        elif norm_types[predictor] == 'Log':
            for i in range(len(dataset_full)): dataset_full[i,predictor] = np.log10(dataset_full[i,predictor])
        elif norm_types[predictor] == 'BoxCox':
            later = 1

    # Split dataset again
    normalized_dataset1 = dataset_full[0:len(dataset1),:]
    normalized_dataset2 = dataset_full[len(dataset1):,:]
    
    return normalized_dataset1, normalized_dataset2



def ar_area(latmax,latmin,lonmax,lonmin):
    area = abs(latmin-latmax)*abs(lonmin-lonmax)
    return area



def simple_1d_prediction(X_train, Y_train, X_val, Y_val):

    idx = np.argsort(X_train)

    # Training
    top = 0
    for i in range(len(Y_train)):
        potential_Y = np.ones(len(Y_train),dtype=np.int8)
        for y in range(i): potential_Y[y] = 0
        if (sum(potential_Y==Y_train[idx])/len(Y_train) > top):
            top = sum(potential_Y==Y_train[idx])/len(Y_train)
            threshold_idx = i

    # Compute Threshold
    threshold = (X_train[threshold_idx]+X_train[threshold_idx])/2

    # Validation
    idx_val = np.argsort(X_val)
    prediction = (X_val[idx_val] > threshold)*1
    accuracy = sum(prediction == Y_val[idx_val])/len(Y_val)

    print('Run {} Accuracy: {}'.format(run,accuracy))



# Function to find which row of the TARP flare peak belongs to
def row_before_and_closer_to_peak(flare_peak_time, file_name):

    # Open TARP file again cause it had problem iterating csv_smarp
    with open('smarp-header/'+file_name, mode='r') as SMARP_file:  
        csv_smarp = csv.DictReader(SMARP_file)
        smallest_difference = 9999999 #seconds (115 days - too much)
        closest_row = 0

        # loops through TARP rows
        row_num = 2
        flux_series = []; lon_series = []
        for row in csv_smarp:
            row_time = datetime.strptime(row['T_REC'][:-4], '%Y.%m.%d_%H:%M:%S') # the recorded time of each row
            difference = flare_peak_time - row_time # timedifference between peak flare time and row
            if (difference.total_seconds() < smallest_difference) and (difference.total_seconds() > 0):
                smallest_difference = difference.total_seconds() # update the smallest time difference
                closest_row = row_num #save the row number
                data = (row['USFLUX'],row['MEANGBZ'],row['LATDTMAX'],row['LATDTMIN'],row['LONDTMAX'],row['LONDTMIN'])
                #if float(row['R_VALUE']) != float(0):
                Rvalue = row['R_VALUE']
            row_num += 1
            flux_series.append(row['USFLUX']); lon_series.append(row['LONDTMIN'])

    #if Rvalue is None:
        #Rvalue = 0

    return closest_row, row_num, smallest_difference, data, flux_series, lon_series, Rvalue