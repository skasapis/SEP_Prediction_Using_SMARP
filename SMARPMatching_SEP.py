
# Match the labeled flares with the SEPs
import csv
import os
import sys
from datetime import datetime, timedelta
from SMARPMatching_functions import get_ar_info, coordinate_transformation, check_ar_boundaries, row_closer_to_peak
from csv import writer
from csv import reader

# Iterate through SMARP active regions (TARP files)
TARP_directory = '/Users/spyroskasapis/Desktop/UofMichigan/CLASPResearch/smarp-header'

SEP_flare_correspondance = [('flare_row','TARP_num')]

match_num = 1
# Iterate through files in smarp-header
for file_name in sorted(os.listdir(TARP_directory)): 

    # Open each file
    with open('smarp-header/'+file_name, mode='r') as SMARP_file:  
        csv_smarp = csv.DictReader(SMARP_file)
        smarp_start_t, smarp_end_t, noaa_ar_number, tarp_num = get_ar_info(csv_smarp) # get active region starting and ending times

        # Open the flares file too
        with open('noaa_SEP_1996_6_5-2011_4_11_26Feb2020.csv', mode='r') as flares_file: 
            sep_flares_reader = csv.DictReader(flares_file)

            # Iterate through SEP flares
            flare_row = 2; 
            for sep_flare in sep_flares_reader:

                # Does it have a matching NOAA AR number with SMARP?
                if sep_flare['flare_region'] != '':
                    if (int(sep_flare['flare_region']) == noaa_ar_number) and (noaa_ar_number != 0): # Only if the ar NOAA number is non zero
                        #print('{}. Flare in row {} is matched with TARP #{}'.format(match_num,flare_row,tarp_num))
                        SEP_flare_correspondance.append((flare_row,tarp_num))
                        match_num += 1

                flare_row += 1

######################### NOTES ####################
# What we have done: 
# Discart: a) Those that have no noaa_ar b) No match with TARP c) Those that are matched but their time is out of TARP (limb)

print(SEP_flare_correspondance)

# No correspondance
print('### No correspondance part ###')

# Open the flares file too
with open('noaa_SEP_1996_6_5-2011_4_11_26Feb2020.csv', mode='r') as flares_file: 
    sep_flares_reader = csv.DictReader(flares_file)

    # Iterate through SEP flares
    flare_row = 2; 
    for sep_flare in sep_flares_reader:
        
        # Check if this flare is matched
        is_flare_matched = False
        for match in SEP_flare_correspondance:
            if flare_row in match:
                is_flare_matched = True
                break

        # Do time check
        if is_flare_matched == False:
            print('This flare was not matched: {}'.format(flare_row))

            flare_time = datetime.strptime(sep_flare['flare_peak_time'], '%Y-%m-%d %H:%M:%S') # Convert to datetime 1997-11-04 05:58:00
            # Iterate through SMARP active regions (TARP files)
            TARP_directory = '/Users/spyroskasapis/Desktop/UofMichigan/CLASPResearch/smarp-header' # Folder path
            for file_name in sorted(os.listdir(TARP_directory)): # Iterate through files in smarp-header
                with open('smarp-header/'+file_name, mode='r') as SMARP_file:  # Open each file
                    csv_smarp = csv.DictReader(SMARP_file)
                    smarp_start_t, smarp_end_t, noaa_ar_number, tarp_num = get_ar_info(csv_smarp) # get active region starting and ending times
                    
                    # Check whether flare peak is within smarp
                    if (flare_time > smarp_start_t) and (flare_time < smarp_end_t):
                        sep_coord = coordinate_transformation(sep_flare['flare_location']) # coordinate tranformation
                        #print(sep_coord)
                        print('Coordinate Match with TARP {} (Time)'.format(tarp_num))
                        # if it is, check whether the coordinates agree
                        if check_ar_boundaries(sep_coord, file_name):
                            print('Coordinate Match with TARP {} (Time + Coordinates)'.format(tarp_num))
                        
        flare_row += 1
    


sys.exit()
################# SHOW CLOSEST ROW ###############

# Open the flares file too
with open('noaa_SEP_labeled.csv', mode='r') as matched_sep_flares_file: 
    matched_sep_flares_reader = csv.DictReader(matched_sep_flares_file)

    # Iterate through matched sep flares
    flare_row = 2; 
    for matched_sep_flare in matched_sep_flares_reader:
        peak_time = datetime.strptime(matched_sep_flare['flare_peak_time'], '%Y-%m-%d %H:%M:%S') # peak time

        # find the row closer to the peak time (how much data before flare)
        closest_row, total_row, smallest_difference = row_closer_to_peak(peak_time, 'TARP{}_ATTRS.csv'.format(matched_sep_flare['TARP_num']))
        print('For flare {} the data points before the flare are: {}/{} - {}'.format(flare_row,closest_row,total_row,smallest_difference/60))
        print()
        flare_row += 1




# Open SEP file
#with open('SEPxFlares.csv', mode='r') as SEP_file:  
    #sep_csv = csv.DictReader(SEP_file)

    #matches = []

    # Loop through SEPs 
    #for sep in sep_csv:

        #sep_flare_time = datetime.strptime(sep['flare_peak_time'], '%Y-%m-%d %H:%M:%S') # Datetime conversion

        # Open flare file
        #with open('Labeled_GOES_SMARP.csv', mode='r') as flare_list:  
            #flare_csv = csv.DictReader(flare_list)

            # Initializations
            #smallest_difference = 9999999 #seconds (115 days - too much)
            #closest_row = 0
            #flare_row = 2

            # Loop through flares
            #for flare in flare_csv:
                #flare_time = datetime.strptime(flare['peak_time'][:-4], '%Y-%m-%dT%H:%M:%S') # the recorded time of each row
                #difference = abs(sep_flare_time - flare_time) # timedifference between peak flare time and row
                #if difference.total_seconds() < smallest_difference:
                    #smallest_difference = difference.total_seconds() # update the smallest time difference
                    #closest_row = flare_row #save the row number
                    #matched_flare = flare
                #flare_row += 1

        #matches.append(closest_row)
        #print(sep['flare_peak_time'],sep['flare_class'],sep['flare_location'],sep['flare_region'])
        #print(matched_flare['peak_time'],matched_flare['goes_class'],matched_flare['goes_location'],matched_flare['noaa_active_region'])
        #print(matched_flare['tarp_labels'])
        #print()




