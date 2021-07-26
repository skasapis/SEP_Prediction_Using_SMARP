## SMARP Matching
import csv
import os
import sys
from datetime import datetime, timedelta
from SMARPMatching_functions import get_ar_info, check_ar_boundaries, row_closer_to_peak
from csv import writer
from csv import reader
import pickle

# Iterate through SMARP active regions (TARP files)
TARP_directory = '/Users/spyroskasapis/Desktop/UofMichigan/CLASPResearch/smarp-header'
ar_count = 1

matched_flare_list = ['matched_flare_row_num']
corresponding_tarp_list = ['corresponding_tarp_num']
negative_ars = []
active_peak_data = []

# Iterate through files in smarp-header
for file_name in sorted(os.listdir(TARP_directory)): 
    negative_ar = True

    # Open each file
    with open('smarp-header/'+file_name, mode='r') as SMARP_file:  
        csv_smarp = csv.DictReader(SMARP_file)
        smarp_start_t, smarp_end_t, noaa_ar_number, tarp_num = get_ar_info(csv_smarp) # get active region starting and ending times
        print(round(tarp_num/13670,3))
        #print('The flares that correspond to TARP{} ({}) are:'.format(tarp_num,ar_count))

        # Open the flares file too
        with open('GOES_SMARP_original.csv', mode='r') as flares_file: 
            flares_reader = csv.DictReader(flares_file)

            # Iterate through GOES flares
            flare_row = 2; 
            for flare in flares_reader:
                flare_peak_time = datetime.strptime(flare['peak_time'][:-4], '%Y-%m-%dT%H:%M:%S') # Convert peak time of flare in datetime object
                matched = False

                # Check whether flare is within active region timeframe
                if (flare_peak_time > smarp_start_t) and (flare_peak_time < smarp_end_t): 

                    # If it is, does it have a matching NOAA AR number with SMARP?
                    if (int(flare['noaa_active_region']) == noaa_ar_number) and (noaa_ar_number != 0): # Only if the ar NOAA number is non zero
                        negative_ar = False # If there is at least one flare match, active region is positive
                        matched = True # It is a match!
                        #matched_flare_list.append(flare_row)
                        #print('Flare #{} with intensity {}'.format(flare_row,flare['goes_class']))
                        #if check_ar_boundaries(flare['goes_location'], file_name) == False: # Active region and flare coordinates check
                            #corresponding_tarp_list.append(str(tarp_num)+'!') #print('#### We got an area problem at flare {} and TARP {}! ####'.format(flare_row,)) # There should all be matching, if not error!
                        #else:
                            #corresponding_tarp_list.append(str(tarp_num))

                    # Or does it maybe have a matching region? USE THIS AS A SANITY CHECK FOR THE ONES THAT HAVE A NOAA NUMBER!
                    elif (int(flare['noaa_active_region']) == 0) and (flare['goes_location'] != '(0, 0)'): 
                        negative_ar = False # If there is at least one flare match, active region is positive
                        matched = True # It is a match!
                        #print('Potentially flare #{} area match'.format(flare_row))
                        #if check_ar_boundaries(flare['goes_location'], file_name): #check if location if within boundaries
                            #print('Flare #{} with intensity {} (area match)'.format(flare_row,flare['goes_class']))
                            #matched_flare_list.append(str(flare_row))
                            #corresponding_tarp_list.append(str(tarp_num))
                
                    # If flare and active region are matched using either way
                    if matched: 
                        closest_row, total_rows, smallest_difference, data = row_closer_to_peak(flare_peak_time, file_name) # find the closest row in the TARP time series
                        print('Closest Row: {}/{}'.format(closest_row, total_rows))
                        active_peak_data.append(data)

                if flare_peak_time > smarp_end_t: 
                    break

                flare_row += 1

    # If no flare is within the active region time range, throw it in the negative dataset
    #if negative_ar:
        #print('Negative: {}'.format(tarp_num))
        #negative_ars.append('%06d'%(int('000000')+tarp_num))

    #if ar_count > 200:
        #break
    ar_count += 1

#print(negative_ars)

################## Negative file ########################

#with open("negative_tarp_nums.txt", "wb") as fp:   #Pickling
    #pickle.dump(negative_ars, fp)

################## Active Peak Data ########################

with open("active_peak_data_v2.txt", "wb") as fp:   #Pickling (v2 refers to the data having the distance too)
    pickle.dump(active_peak_data, fp)




################## Xiantong's file ######################

Xiantongs_file = False 

if Xiantongs_file == True:
    # Open the input_file in read mode and output_file in write mode
    with open('GOES_SMARP_original.csv', 'r') as read_obj, \
            open('Labeled_GOES_SMARP.csv', 'w', newline='') as write_obj:
        # Create a csv.reader and writer objects from the input and output file object
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)

        # Read each row of the input csv file as list
        row_num = 1
        for row in csv_reader:
            if row_num == 1:
                text = 'tarp_labels'
            elif row_num not in matched_flare_list:
                text = '0'
            else:
                tarp_matches = []
                for i in range(len(matched_flare_list)):
                    if matched_flare_list[i] == row_num:
                        tarp_matches.append(corresponding_tarp_list[i])
                text = str(tarp_matches)

            # Append the default text in the row / list
            row.append(text)
            # Add the updated row / list to the output file
            csv_writer.writerow(row)
            row_num += 1

#for i in range(len(matched_flare_list)):
    #print(matched_flare_list[i],corresponding_tarp_list[i])



