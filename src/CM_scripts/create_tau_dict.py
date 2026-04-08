#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 11:18:01 2026

@author: cmazzio
"""

# Run this script after running data through Abu's Pytau package to upload
# spike trains and tau values into one dictionary 

import sys
import pickle
import easygui
import os
import csv
import numpy as np
from scipy import stats
# Get dataframe for all data
import pandas as pd
from scipy.ndimage import gaussian_filter1d
dframe_path = '/media/cmazzio/large_data/Change_point_models/model_database.csv'
dframe = pd.read_csv(dframe_path)

# %% Function definition


def int_input(prompt):
    # This function asks a user for an integer input
    int_loop = 1
    while int_loop == 1:
        response = input(prompt)
        try:
            int_val = int(response)
            int_loop = 0
        except:
            print("\tERROR: Incorrect data entry, please input an integer.")

    return int_val


def bool_input(prompt):
    # This function asks a user for an integer input
    bool_loop = 1
    while bool_loop == 1:
        response = input(prompt)
        if (response.lower() != 'y')*(response.lower() != 'n'):
            print("\tERROR: Incorrect data entry, only give Y/y/N/n.")
        else:
            bool_val = response.lower()
            bool_loop = 0

    return bool_val

# %% Load all files to be analyzed

desired_states = 4  # which number of states dataset to use

# Prompt user for the number of datasets needed in the analysis
num_files = int_input(
    "How many recording day files do you need to import for this analysis (integer value)? ")
if num_files > 1:
    print("Multiple file import selected.")
else:
    print("Single file import selected.")


# Pull all cp data into a dictionary
# Includes: data_dir, basename, tau values for every cp, 
    # spike trains for every trial, number of states in cp model, session name
nf_i = 0
tau_data_dict = dict()
for nf in range(num_files):
    # Directory selection
    print("Please select the folder where the data # " + str(nf+1) + " is stored.")
    data_dir = easygui.diropenbox(
        title='Please select the folder where data is stored.')
    # Import individual trial changepoint data
    name_bool = dframe['data.data_dir'].isin([data_dir])
    wanted_frame = dframe[name_bool]
    state_bool = wanted_frame['model.states'] == desired_states
    wanted_frame = wanted_frame[state_bool]
    name_list = wanted_frame['data.basename']
    taste_list = list(wanted_frame['data.taste_num'])
    taste_name_list = list(wanted_frame['exp.exp_name'])
    taste_save_path_list = list(wanted_frame['exp.save_path'])
    print("There are " + str(len(taste_list)) + " tastes available.")
    print("Taste Indices: ")
    print(taste_list)
    print("Taste Names: ")
    print(taste_name_list)
    num_taste_keep = int_input("\nHow many tastes do you want to analyze?")
    for t_i in np.arange(num_taste_keep):
        print("Select taste " + str(t_i))
        for tl_i in range(len(taste_list)):
            print_statement = "\nIndex: " + str(tl_i) + "\n\tTaste num: " + str(taste_list[tl_i]) + \
                "\n\tTaste name: " + taste_name_list[tl_i] + \
                "\n\tTaste save path: " + taste_save_path_list[tl_i]
            print(print_statement)
        taste_ind = int_input(
            "\nWhich taste do you want (given index above)? ")
        taste_name = str(np.array(taste_name_list)[taste_ind])
        pkl_path = list(wanted_frame['exp.save_path'])[taste_ind]
        data_name = list(name_list)[taste_ind]
        # Import changepoints for each delivery
        scaled_mode_tau = np.load(
            pkl_path+'_scaled_mode_tau.npy').squeeze()  # num trials x num cp
        # Import spikes following each taste delivery
        # num trials x num neur x time (pre-taste + post-taste length)
        spike_train = np.load(pkl_path+'_raw_spikes.npy').squeeze()
        # Store changepoint and spike data in dictionary
        tau_data_dict[nf_i] = dict()
        tau_data_dict[nf_i]['data_dir'] = data_dir
        print("Give a more colloquial name to the dataset.")
        given_name = input("How would you rename " +
                           data_name + " taste " + taste_name + "? ")
        if given_name[:2] == '\n':
            given_name = given_name[2:]
        tau_data_dict[nf_i]['true_name'] = data_name
        tau_data_dict[nf_i]['given_name'] = given_name
        tau_data_dict[nf_i]['states'] = desired_states
        tau_data_dict[nf_i]['scaled_mode_tau'] = scaled_mode_tau
        tau_data_dict[nf_i]['spike_train'] = spike_train
        # Import associated gapes
        print("Now import associated first gapes data with this dataset.")
        nf_i += 1

# Analysis Storage Directory
print('Please select a directory to save all results from this set of analyses.')
results_dir = easygui.diropenbox(title='Please select the storage folder.')

# Save dictionary
dict_save_dir = os.path.join(results_dir, 'tau_dict.pkl')
f = open(dict_save_dir, "wb")
pickle.dump(tau_data_dict, f)
# with open(dict_save_dir, "rb") as pickle_file:
#    tau_data_dict = pickle.load(pickle_file)