# Create SEP datasets

#from SMARP_SEP_Prediction.statistics_functions import graph_SMARP_sequences_positive
import csv
import os
import sys
from datetime import datetime, timedelta
from SMARPMatching_functions import row_before_and_closer_to_peak, angular_distance, not_in_positive
from statistics_functions import graph_SMARP_sequences_positive, graph_SMARP_sequences_negative
from csv import writer
from csv import reader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import statistics
import seaborn as sns

################## Positive ##################
create_positive = True

if create_positive == True: 
    
    print('Create Positive Data')

    # Open the flares file
    with open('noaa_SEP_labeled_only.csv', mode='r') as matched_sep_flares_file: 
        matched_sep_flares_reader = csv.DictReader(matched_sep_flares_file)

        # Save flux and GBZ of the peak of each flare 
        positive_data = []
        time_differences = []
        time_series_max_entries = []
        flux_all = []; lon_all = []; last_row_time = []; sep_start_time_data = []

        # Iterate through matched sep flares
        flare_row = 2; 
        for matched_sep_flare in matched_sep_flares_reader:
            #print(flare_row)
            peak_time = datetime.strptime(matched_sep_flare['flare_peak_time'], '%Y-%m-%d %H:%M:%S') # flare peak time
            sep_start_time = datetime.strptime(matched_sep_flare['start_time'], '%Y-%m-%d %H:%M:%S') # sep start time
            has_data = False 

            # Open TARP file again cause it had problem iterating csv_smarp
            with open('smarp-header/'+'TARP{}_ATTRS.csv'.format(matched_sep_flare['TARP_num']), mode='r') as SMARP_file:  
                csv_smarp = csv.DictReader(SMARP_file)
                smallest_difference = 35000; smallest_difference_sep = 35000 #seconds
                closest_row = 0; zero_row = 0
                flux_seq = []; lon_seq = []

                # loops through TARP rows
                row_num = 2
                for row in csv_smarp:
                    row_time = datetime.strptime(row['T_REC'][:-4], '%Y.%m.%d_%H:%M:%S') # the recorded time of each row
                    difference = peak_time - row_time # timedifference between peak flare time and row
                    difference_sep = sep_start_time - row_time
                    flux_seq.append(float(row['USFLUX'])); lon_seq.append(float(row['LONDTMIN']))

                    if (has_data == False) and (float(row['USFLUX']) == float(0)): #which row has no data before the flare peak time
                        zero_row = row_num

                    if abs(difference_sep.total_seconds()) < smallest_difference_sep: # ONLY FOR GRAPHING (to get the SEP start time essentially)
                        smallest_difference_sep = difference_sep.total_seconds() # update the smallest time difference
                        usflux_sep = row['USFLUX'] # USFLUX
                        londtmin_sep = row['LONDTMIN'] # USFLUX
                        #print(row_num)

                    if (difference.total_seconds() < smallest_difference) and (difference.total_seconds() > 600) and (float(row['USFLUX']) != float(0)):
                        smallest_difference = difference.total_seconds() # update the smallest time difference
                        closest_row = row_num #save the row number
                        usflux = row['USFLUX'] # USFLUX
                        meangbz = row['MEANGBZ'] # MEANGBZ
                        latdmax = row['LATDTMAX'] # LATDTMAX
                        latdmin = row['LATDTMIN'] # LATDTMIN
                        londtmax = row['LONDTMAX'] # LONDTMAX
                        londtmin = row['LONDTMIN'] # LONDTMIN
                        if float(row['R_VALUE']) != float(0):
                            rvalue = row['R_VALUE'] #R_VALUE
                        has_data = True
                    row_num += 1
                
                if has_data == True:
                    data = [usflux,meangbz,latdmax,latdmin,londtmax,londtmin,matched_sep_flare['flare_location'],matched_sep_flare['flare_class'],rvalue]
                    positive_data.append(data) #USFLUX,MEANGBZ,LATDTMAX,LATDTMIN,LONDTMAX,LONDTMIN,goes_location,goes_class
                    time_differences.append(smallest_difference)
                    time_series_max_entries.append(closest_row-zero_row)
                    flux_all.append(flux_seq); lon_all.append(lon_seq); last_row_time.append(row_time); sep_start_time_data.append([londtmin_sep,usflux_sep])

            flare_row += 1

    #graph_SMARP_sequences_positive(flux_all,lon_all,last_row_time,positive_data,sep_start_time_data) # Graphing of the entire time series

    time_difference_hist = False
    if time_difference_hist == True:
        sns.set(style="ticks")
        time_differences = np.array(time_differences)/60
        time_differences = time_differences[time_differences < 112]
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.05, .85)})
        sns.boxplot(time_differences, ax=ax_box, color = 'teal')
        sns.distplot(time_differences, ax=ax_hist, bins = 10, kde = False,
                    fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, 
                    color='teal', vertical=False, norm_hist=True, axlabel=None, label=None)
        ax_box.set(yticks=[])
        ax_hist.set_xlim([10, 100])
        ax_hist.set_ylim([0, 0.019])
        ax_hist.set_ylabel('Positive   Negative')
        ax_hist.set_xlabel(r'Time Difference $Δt$')
        #sns.despine(ax=ax_hist)
        sns.despine(ax=ax_box,left=True, bottom=False, offset=5)
        plt.show()
        print('Positive Mean time: {} and Std: {}'.format(np.mean(time_differences),np.std(time_differences)))

    # Save Positive Data
    with open("sep_data_v5.txt", "wb") as fp:   #Pickling (v2 refers to the data having the distance too)
        pickle.dump(positive_data, fp)


#################### Quiet #######################
create_quiet = False

if create_quiet == True: 

    print('Create Negative Data')

    # Save random flux and GBZ of each 
    negative_data = [] # from active regions that are not matched with flares in the GOES flares list

    # Open Negative TARP nums list
    with open("negative_tarp_nums.txt", "rb") as fp:   # Unpickling
        tarp_num_list = pickle.load(fp)

    # Loop through negative tarps
    for negative_tarp in tarp_num_list:
        # Open TARP file again cause it had problem iterating csv_smarp
        with open('smarp-header/'+'TARP{}_ATTRS.csv'.format(negative_tarp), mode='r') as SMARP_file:  
            csv_smarp = list(csv.DictReader(SMARP_file))
            k = 0
            for row in csv_smarp:
                if (k == random.randint(1, len(list(csv_smarp)))) and (float(row['USFLUX']) != float(0)):
                    #print('in here')
                    usflux = row['USFLUX'] # USFLUX
                    meangbz = row['MEANGBZ'] # MEANGBZ
                    latdmax = row['LATDTMAX'] # MEANGBZ
                    latdmin = row['LATDTMIN'] # USFLUX
                    londtmax = row['LONDTMAX'] # MEANGBZ
                    londtmin = row['LONDTMIN'] # USFLUX
                k += 1
        
        data = [usflux,meangbz,latdmax,latdmin,londtmax,londtmin,'(0,0)','N/A']
        negative_data.append(data) #USFLUX,MEANGBZ,LATDTMAX,LATDTMIN,LONDTMAX,LONDTMIN,goes_location,goes_class
        print(data)

    # Save Positive Data
    with open("negative_sep_data_v3.txt", "wb") as fp:   #Pickling
        pickle.dump(negative_data, fp)


#################### Active #######################

# Here I only choose one value from each TARP file - and it is random. For the Active Peak I choose all the peak values for every corresponding flare
# This means that for every smarp we could have multiple values

create_active = True

if create_active == True: 
    
    # Open the flares file
    with open('Labeled_GOES_SMARP.csv', mode='r') as flares_file: 
        flares_reader = csv.DictReader(flares_file)

        active_data = []
        previous_tarp = 1

        for flare in flares_reader:

            if (flare['tarp_labels'] != '0') and (flare['tarp_labels'].find('!') < 0) and (len(flare['tarp_labels'])<10): 

                if (int(flare['tarp_labels'][2:len(flare['tarp_labels'])-2]) != previous_tarp):

                    tarp_num = '%06d'%(int('000000')+int(flare['tarp_labels'][2:len(flare['tarp_labels'])-2]))
                    #print(tarp_num)

                    # Open TARP file again cause it had problem iterating csv_smarp
                    with open('smarp-header/'+'TARP{}_ATTRS.csv'.format(tarp_num), mode='r') as SMARP_file:  
                        csv_smarp = list(csv.DictReader(SMARP_file))
                        k = 0
                        inside = False
                        for row in csv_smarp:
                            if (k == random.randint(1, len(list(csv_smarp)))) and (float(row['USFLUX']) != float(0)):
                                #print('in here')
                                inside = True
                                usflux = row['USFLUX'] # USFLUX
                                meangbz = row['MEANGBZ'] # MEANGBZ
                                latdmax = row['LATDTMAX'] # MEANGBZ
                                latdmin = row['LATDTMIN'] # USFLUX
                                londtmax = row['LONDTMAX'] # MEANGBZ
                                londtmin = row['LONDTMIN'] # USFLUX
                            k += 1

                    if inside == True: 
                        active_data.append([usflux,meangbz,latdmax,latdmin,londtmax,londtmin,flare['goes_location'],flare['goes_class']]) #USFLUX,MEANGBZ,LATDTMAX,LATDTMIN,LONDTMAX,LONDTMIN,goes_location,goes_class
                    previous_tarp = int(flare['tarp_labels'][2:len(flare['tarp_labels'])-2])

    # Save Active Data
    with open("active_data.txt", "wb") as fp:   #Pickling
        pickle.dump(active_data, fp)



#################### Negative #######################

create_peak_active = True

if create_peak_active == True: 
    
    time_differences_neg = []

    # Open the flares file
    with open('Labeled_GOES_SMARP.csv', mode='r') as flares_file: 
        flares_reader = csv.DictReader(flares_file)

        active_peak_data = []; flux_all = []; lon_all = []; flare_date = []; flare_points = []
        
        for flare in flares_reader:

            if (flare['tarp_labels'] != '0') and (flare['tarp_labels'].find('!') < 0) and (len(flare['tarp_labels'])<10) and not_in_positive(flare['peak_time'],flare['goes_class']): 

                tarp_num = '%06d'%(int('000000')+int(flare['tarp_labels'][2:len(flare['tarp_labels'])-2]))

                flare_peak_time = datetime.strptime(flare['peak_time'][:-4], '%Y-%m-%dT%H:%M:%S') # Convert peak time of flare in datetime object
                file_name = 'TARP{}_ATTRS.csv'.format(tarp_num)
                closest_row, total_rows, smallest_difference, data, flux_series, lon_series, rvalue = row_before_and_closer_to_peak(flare_peak_time, file_name)
                if (smallest_difference < 6000) and (smallest_difference > 600):
                    time_differences_neg.append(smallest_difference)
                    data = list(data)
                    
                    data.append(flare['goes_location']); data.append(flare['goes_class']); data.append(rvalue)
                    active_peak_data.append(data) #USFLUX,MEANGBZ,LATDTMAX,LATDTMIN,LONDTMAX,LONDTMIN,goes_location,goes_class
                    flux_all.append(flux_series); lon_all.append(lon_series); flare_date.append(flare_peak_time); flare_points.append([data[5],data[0]])

    #graph_SMARP_sequences_negative(flux_all,lon_all,flare_date,flare_points) # Graphing of the entire time series

    time_differences_neg = np.array(time_differences_neg)/60

    print('Negative Mean time: {} and Std: {}'.format(np.mean(time_differences_neg),np.std(time_differences_neg)))

    time_difference_hist = False
    if time_difference_hist == True:
        sns.set(style="ticks")
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.05, .85)})
        sns.boxplot(time_differences_neg, ax=ax_box, color = '#DC143C')
        sns.distplot(time_differences_neg, ax=ax_hist, bins = 10, hist=True, kde=False, 
                    fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, 
                    color='#DC143C', vertical=False, norm_hist=False, axlabel=None, label=None)
        ax_box.set(yticks=[])
        ax_hist.set_xlim([10, 110])
        #ax_hist.set_ylim([0, 0.019])
        #sns.despine(ax=ax_hist)
        sns.despine(ax=ax_box,left=True, bottom=False, offset=5)
        plt.show()


    # Save Active Data
    with open("active_peak_data_v5.txt", "wb") as fp:   #Pickling (v2 refers to the data having the distance too)
        pickle.dump(active_peak_data, fp) #USFLUX,MEANGBZ,dist1,dist2