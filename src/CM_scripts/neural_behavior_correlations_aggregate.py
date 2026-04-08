#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:46:40 2026

@author: cmazzio
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 14:55:21 2026

@author: cmazzio
"""

# Runs two types of correlations to neural activity:
    # 1) palatability ranks: manually entered ranks in .info file from BlechClust
    # 2) behavior scores: percent of time spent engaging in MTMs, gapes,
        # no movement (as determined by Blech EMG classifier) in 2s following 
            # taste delivery

# Plots frequency of MTMs, gapes, and no movement as determined by Blech EMG
    # classifier across trials for each session


# Creates 1000x shuffles of palatability ranks and runs control shuffle 
    # correlations for palatability ranks and behavior scores 
    
# Plots palatability rank vs. behavior score correlations on same axes for each session
# Plots shuffled data and real data correlations on same axes for each session

# All plots save to GC_behavior_correlation folder in BlechGapes Analysis folder
    # inside each data directory 

# Final aggregate plot of palatability vs. behavior score correlation plot made 
    # for all datasets included in data directory, saved to large_data/GC_behavior_correlation_aggregate_session_plots

#%%IMPORTS

import numpy as np
import easygui
import pickle
import os
from collections import OrderedDict
from scipy.stats import spearmanr
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import sem 
import math 
from tqdm import tqdm
import pandas as pd
import seaborn as sns


#%% Define functions 

def normalize_taste(name):
    name = name.strip()
    return taste_map.get(name, name.lower())


#%% Load data and create arrays for correlations

# Text file should have list of data directories that you want to process at once
txt_file = "/media/cmazzio/large_data/dataset_list/0.6M_LiCl_CTATest1_dataset_list"
with open(txt_file, "r") as f:
    data_dirs = [line.strip() for line in f if line.strip()]


# Import EMG classifier data 

# EMG data for all sessions is saved inside one dataframe 
emg_data_dir = '/media/cmazzio/large_data/EMG_classifier_data'
filename = '/media/cmazzio/large_data/EMG_classifier_data/christina_all_datasets.pkl'

# Create full pathto EMG data for all sessions
file_path = os.path.join(emg_data_dir, filename)

# Load EMG DataFrame
df = pd.read_pickle(file_path)

# Set taste map for normalizing taste names across files

taste_map= {
            'NaCl' : 'nacl',
             'QHCl' : 'highqhcl',
             'highQHCl' : 'highqhcl',
             'highqhcl' : 'highqhcl',
             'lowQHCl': 'lowqhcl',
             'lowqhcl': 'lowqhcl',
             'nacl': 'nacl',
             'saccharin' : 'saccharin',
             'water' : 'water',
             'sac' : 'saccharin',
             'h2o' : 'water',
             'H2O' : 'water',
             'qhcl' : 'highqhcl'
             }  

# Initialize dictionaries
spike_trains_dict = {} # contains cps, spike trains, tastes, and results_dir for each taste, session
behavior_score_dict = {} # contains behavior fractions for each behavior type in each session

# Loop through all file directories and extract data from tau_dict.pkl
for nf, data_dir in enumerate(data_dirs):
    print(f"\nProcessing session {nf+1}: {data_dir}")

    pkl_path = os.path.join(data_dir, 'BlechGapes_analysis', 'tau_dict.pkl')
    
    # Initialize embedded dictionary for each session
    session_name = os.path.basename(data_dir)
    spike_trains_dict[session_name] = {}
    
    # Define results directory for this session and save in dictionary
    session_results_dir = os.path.join(data_dir, "BlechGapes_analysis", "GC_behavior_correlation")
    os.makedirs(session_results_dir, exist_ok=True)
    
    # Initialize taste_pal_dict for this session:
    taste_pal_dict = {}
    
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)    
        taste_names = []
        taste_list = []
        # Extract spike train for this session
        for taste in range(len(data)):
            this_spike_train = data[taste]['spike_train']
            this_taste = data[taste]['given_name'].split("_")[-1]
            
            # Make sure taste name matches taste map
            normalized_name = taste_map.get(this_taste, this_taste.lower())
            print('extracting spike train for ' + normalized_name)
            taste_names.append(normalized_name)
            
            
            this_taste_cps = data[taste]['scaled_mode_tau']
            n_trials = np.shape(this_spike_train)[0]
            
            spike_trains_dict[session_name][normalized_name] = {
            "spike_train": this_spike_train,
            "n_trials": n_trials,
            "cps": this_taste_cps,
            "results_dir" : session_results_dir
            }
            
        # Keep lists for any additional uses
        taste_names.append(normalized_name)
        taste_list.append(this_spike_train) 
        
        # Read in taste list and palatability rankings according to digin order
        info_file = glob.glob(os.path.join(data_dir, "*.info"))
        if info_file:
            # Open the first .info file found
            with open(info_file[0], "r") as file:
                info_dict = json.load(file)
            digin_tastes = info_dict["taste_params"]["tastes"]
            pal_ranks = info_dict["taste_params"]["pal_rankings"]
        else:
            print("No .info files found in specified directory")
            continue 
        
        # Make sure taste names from .info file match the taste map
        for taste_name, pal in zip(digin_tastes, pal_ranks):
            normalized_name = normalize_taste(taste_name)
            taste_pal_dict[normalized_name] = pal
            # Add palatability rank to spike_train_dict for each session, taste
            if normalized_name in spike_trains_dict[session_name]:
                spike_trains_dict[session_name][normalized_name]['pal_rank'] = pal
                
    # Get the order of keys from taste_pal_dict using enumerate
    order = {key: index for index, key in enumerate(taste_pal_dict.keys())}
    # Match order of tastes in taste_pal_dict and spike_trains_dict for this session
    ordered_spike_trains_dict = dict(sorted(spike_trains_dict[session_name].items(), key=lambda item: order[item[0]])) 
    taste_list_neural = list(ordered_spike_trains_dict.keys())
      
    
    
    ###########################################################################
    ## Creating Behavior Vectors 
    ###########################################################################
    # Create vector for each behavior for each session and save in dictionary
    string_divide = session_name.split('_')
    this_basename = '_'.join(string_divide[:2])
    this_session_df = df[df['basename'] == this_basename]
    
    # Print taste names in dataframe before aligning to neural taste names
    df_taste_names = this_session_df["taste_name"].unique()
    print("EMG classifier taste names:", df_taste_names)
    
    # Normalize taste names in the dataframe
    this_session_df['taste_name'] = (
    this_session_df['taste_name']
    .str.strip()               # remove whitespace
    .map(lambda x: taste_map.get(x, x.lower()))
    )
    
    # Print taste names in dataframe after aligning to neural taste names
    taste_list_behavior_clean = this_session_df["taste_name"].unique()
    print("EMG classifier taste names after aligning with neural data:", taste_list_behavior_clean)
    
    
    ###########################################################################
    ## Creating Trial Map for Concatenation of Tastes
    ###########################################################################
    
    official_taste_order = taste_list_behavior_clean.copy()

    global_trial_map = {}
    current_trial = 0

    for taste in official_taste_order:
        taste_trials = (
            this_session_df.loc[this_session_df['taste_name'] == taste, 'trial']
            .unique()
        )
        taste_trials = sorted(taste_trials)

        for t in taste_trials:
            global_trial_map[(taste, t)] = current_trial
            current_trial += 1
    
    # Set behavior detection window
    start_window= 2000
    end_window = 4000
    window_duration = end_window - start_window
    
    # Initialize dictionary with session names and behavior score arrays
    behavior_score_dict[session_name] = {}
    
    
    # Intialize vector of zeros to eventually fill in behavior score for each trial
    # One score per behavior type, per trial
    num_total_trials = len(global_trial_map)
    num_behaviors = len(this_session_df['cluster_num'].unique())
    behavior_array = np.zeros(shape = (num_total_trials, num_behaviors))

    
    # Create a mapping from cluster number to column index
    cluster_list = sorted(this_session_df['cluster_num'].unique())
    cluster_map = {cluster: i for i, cluster in enumerate(cluster_list)}
    num_clusters = len(cluster_list)
    
    
    for _, row in this_session_df.iterrows():
        taste = row['taste_name']
        local_trial = row['trial']
        cluster = row['cluster_num']
        start_time, end_time = row['segment_bounds']

        # Skip trials not in the global map (extra safety)
        key = (taste, local_trial)
        if key not in global_trial_map:
            continue

        # Compute overlap with analysis window
        overlap_start = max(start_time, start_window)
        overlap_end = min(end_time, end_window)

        # Only add duration if there is overlap
        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start

            global_idx = global_trial_map[key]
            cluster_idx = cluster_map[cluster]

            behavior_array[global_idx, cluster_idx] += overlap_duration
    behavior_fraction = behavior_array / window_duration
    behavior_score_dict[session_name]["behavior_array"] = behavior_fraction
    behavior_score_dict[session_name]["global_trial_map"] = global_trial_map

###########################################################################
## Creating Palatability Vectors 
###########################################################################
pal_rank_dict = {}  # contains manually designated palatability ranks from .info file

for session in spike_trains_dict.keys():
    # Global trial map for this session
    this_trial_map = behavior_score_dict[session]['global_trial_map']
    num_trials = len(this_trial_map)
    
    # Initialize vector
    pal_vector = np.zeros(shape = num_trials)


    # Loop through all trials in global map
    for (taste_name, local_trial), global_idx in this_trial_map.items():

        # Assign palatability rank
        pal_rank = spike_trains_dict[session][taste_name]['pal_rank']
    
        pal_vector[global_idx] = pal_rank

    # Save vector in dictionary
    pal_rank_dict[session] = pal_vector


##############################################################################
## Save Dictionaries in results folder designated per session
##############################################################################


for session in spike_trains_dict.keys():
    
    # Get results folder from one of the tastes (all tastes share the same folder)
    any_taste = list(spike_trains_dict[session].keys())[0]
    results_dir = spike_trains_dict[session][any_taste]["results_dir"]

    # Make sure the directory exists
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------
    # 1️⃣ Save palatability vector for this session
    # -------------------------------
    with open(os.path.join(results_dir, f"{session}_pal_rank_dict.pkl"), "wb") as f:
        pickle.dump(pal_rank_dict[session], f)

    # -------------------------------
    # 2️⃣ Save spike trains dict for this session
    # -------------------------------
    with open(os.path.join(results_dir, f"{session}_spike_trains_dict.pkl"), "wb") as f:
        pickle.dump(spike_trains_dict[session], f)

    # -------------------------------
    # 3️⃣ Save behavior score dict for this session
    # -------------------------------
    with open(os.path.join(results_dir, f"{session}_behavior_score_dict.pkl"), "wb") as f:
        pickle.dump(behavior_score_dict[session], f)

    print(f"Saved session '{session}' data to {results_dir}")


#%% Plot frequency of behaviors across session

# Map cluster numbers to behavior names
behavior_name_map = {0: "No Movement", 1: "Gapes", 2: "MTMs"}

for session_ind, session in enumerate(behavior_score_dict.keys()):
    this_session_trials = np.shape(behavior_score_dict[session]["behavior_array"])[0]
    this_session_num_tastes = len(spike_trains_dict[session])
    this_session_taste_list = list(spike_trains_dict[session].keys())
    print(this_session_taste_list)
    
    fig, axes = plt.subplots(
        this_session_num_tastes, 1,
        figsize=(10, 4 * this_session_num_tastes),
        sharey=True
        )

    # Ensure axes is iterable when only one taste
    if this_session_num_tastes == 1:
        axes = [axes]

    for ax, taste in zip(axes, this_session_taste_list):

        # Get local trials for this taste
        local_trials = sorted([
            trial for (t, trial) in behavior_score_dict[session]["global_trial_map"].keys()
            if t == taste
        ])
        print(local_trials)

        # Map to global indices
        global_indices = [
            behavior_score_dict[session]["global_trial_map"][(taste, trial)]
            for trial in local_trials
            ]

        # Plot one line per behavior cluster
        for cluster, cluster_idx in cluster_map.items():
            ax.plot(
                local_trials,
                behavior_score_dict[session]["behavior_array"][global_indices, cluster_idx],
                marker='o',
                label=behavior_name_map.get(cluster, f'Cluster {cluster}')  # uses name
            )
        
        basename = '_'.join(session.split('_')[:2])
        ax.set_title(f'Taste: {taste} | Session: {basename}', fontsize=14)
        ax.set_xlabel('Local Trial Number')
        ax.set_ylabel('Behavior Frequency')
        ax.legend(fontsize=10)
    plt.tight_layout()
    
    # -------------------------------
    # Save figure in the session's results folder
    # -------------------------------
    any_taste = list(spike_trains_dict[session].keys())[0]
    results_dir = spike_trains_dict[session][any_taste]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    fig_path = os.path.join(results_dir, f"{session}_behavior_plot.png")
    fig.savefig(fig_path, dpi=300)
    print(f"Saved behavior plot for session '{session}' to {fig_path}")
    
    plt.close(fig)  # Close figure to free memory
    

#%% Find binned & z-scored spike rate across each trial for each taste 
pre_stim = 1500
post_stim = 4000
window_size = 250
step_size = 25
# Calcuate the number of windows possible in 1500-4000 ms with current step and window length parameters
num_window = int((post_stim - pre_stim - window_size )/ step_size)


i_fr_dict = {}
concat_z_fr_dict = {}
for session_ind, session in enumerate(spike_trains_dict.keys()): 
    i_fr_dict[session] = {}
    concat_z_fr_dict[session] = {}
    for taste_ind, taste in enumerate(spike_trains_dict[session]):
        this_taste_fr_array = spike_trains_dict[session][taste]['spike_train']
        num_trial, num_neurs, _ = np.shape(this_taste_fr_array)
        i_fr_dict[session][taste] = {}
        
        # Initialize array to store binned FR for each neuron, each trial
        this_taste_i_fr_array = np.zeros((num_trial, num_neurs,num_window))
        # For each neuron, on each trial, find FR within the designated windows
        for trial in range(num_trial):
            for neur in range(num_neurs):
                # Set start and end time of first window
                window_bounds = [int(pre_stim), int(pre_stim + window_size)]
                for window in range(num_window):
                    window_sum = np.sum(this_taste_fr_array[trial,neur,window_bounds[0]:window_bounds[1]])
                    # Divide sum of spikes in each window by window size to get FR in bin
                    i_fr_window = window_sum / window_size
                    # Update window start and end time for next iteration
                    window_bounds[0] += step_size
                    window_bounds[1] += step_size
                    # Add this window's FR to a new array that is trial x neur x window number
                    this_taste_i_fr_array[trial, neur, window] = i_fr_window
                    i_fr_dict[session][taste] = this_taste_i_fr_array
                    
    # Concatenate i_fr arrays along trial axis for each session
    all_trials_i_fr_array = np.concatenate(list(i_fr_dict[session].values()), axis=0)
    
    # Calculate mean FR and standard deviation for each neuron across all trials 
    # Compute mean and std per neuron across trials AND bins
    mean_per_neuron = all_trials_i_fr_array.mean(axis=(0, 2), keepdims=True)
    std_per_neuron = all_trials_i_fr_array.std(axis=(0, 2), keepdims=True)

    # Avoid divide-by-zero
    std_per_neuron[std_per_neuron == 0] = 1

    # Z-score each neuron FR across all taste trials
    all_trials_z_array = (all_trials_i_fr_array - mean_per_neuron) / std_per_neuron
    
    concat_z_fr_dict[session] = all_trials_z_array

   
#%% Run Spearman correlation between z-scored firing rates and: 
# 1) palatability ranks
# 2) behavior scores

# Correlation of activity to palatability rankings from .info file
r2_pal_dict = {}
r2_mean_pal_dict = {}

# Correlate z-scored FR for each neuron in each window with palatability ranks
for session, z_array in concat_z_fr_dict.items():
    n_trials, n_neurons, n_windows = z_array.shape
    r2_pal_dict[session] = {}
    
    # We'll store rho^2 values (coefficient of determination)
    this_session_corr_array = np.zeros((n_neurons, n_windows))
    
    for neur in range(n_neurons):
        for window in range(n_windows):
            this_neur_bin_vector = z_array[:, neur, window]  # length == n_trials
            this_pal_vector = pal_rank_dict[session]         # length == n_trials
            
            # Compute Spearman correlation
            rho, pval = spearmanr(this_neur_bin_vector, this_pal_vector)
            
            # Store rho squared (R^2) for that neuron and time bin
            this_session_corr_array[neur, window] = rho**2
    
    # Save session correlation array in dictionary
    r2_pal_dict[session] = this_session_corr_array
    
    # Calculate mean correlation across neurons at each window
    r2_mean_pal_corr = np.nanmean(this_session_corr_array, axis=0)
    r2_mean_pal_dict[session] = r2_mean_pal_corr
    


# Correlation of activity to behavior scores based on 
# duration of behavior within 2000ms following delivery 

r2_behavior_dict = {}
r2_mean_behavior_dict = {}

for session, z_array in concat_z_fr_dict.items():

    this_session_behavior_array = behavior_score_dict[session]['behavior_array']
    n_clusters = this_session_behavior_array.shape[1]

    n_trials, n_neurons, n_windows = z_array.shape

    # clusters × neurons × windows
    this_session_corr_array = np.zeros((n_clusters, n_neurons, n_windows))

    for cluster in range(n_clusters):

        behavior_vector = this_session_behavior_array[:, cluster]

        for neur in range(n_neurons):
            for window in range(n_windows):

                fr_vector = z_array[:, neur, window]

                rho, pval = spearmanr(fr_vector, behavior_vector)

                this_session_corr_array[cluster, neur, window] = rho**2

    # Store full matrix
    r2_behavior_dict[session] = this_session_corr_array

    # Mean across neurons (axis=1)
    # result shape: clusters × windows
    r2_mean_behavior_dict[session] = np.nanmean(this_session_corr_array, axis=1)


for session in spike_trains_dict.keys():
    any_taste = list(spike_trains_dict[session].keys())[0]
    results_dir = spike_trains_dict[session][any_taste]['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Save neural firing rates
    with open(os.path.join(results_dir, f'{session}_i_fr_dict.pkl'), 'wb') as f:
        pickle.dump(i_fr_dict[session], f)
        
    with open(os.path.join(results_dir, f'{session}_concat_z_fr.pkl'), 'wb') as f:
        pickle.dump(concat_z_fr_dict[session], f)
        
    # Save correlations
    with open(os.path.join(results_dir, f'{session}_r2_pal.pkl'), 'wb') as f:
        pickle.dump(r2_pal_dict[session], f)
        
    with open(os.path.join(results_dir, f'{session}_r2_mean_pal.pkl'), 'wb') as f:
        pickle.dump(r2_mean_pal_dict[session], f)
        
    with open(os.path.join(results_dir, f'{session}_r2_behavior.pkl'), 'wb') as f:
        pickle.dump(r2_behavior_dict[session], f)
        
    with open(os.path.join(results_dir, f'{session}_r2_mean_behavior.pkl'), 'wb') as f:
        pickle.dump(r2_mean_behavior_dict[session], f)
        
    # Save supporting info
    with open(os.path.join(results_dir, f'{session}_pal_rank.pkl'), 'wb') as f:
        pickle.dump(pal_rank_dict[session], f)
        
    with open(os.path.join(results_dir, f'{session}_behavior_array.pkl'), 'wb') as f:
        pickle.dump(behavior_score_dict[session]['behavior_array'], f)


#%% Create shuffle data and run control correlations

# Palatability ranks shuffle control
# Shuffles trial palatability identity, preserving neural data

n_shuffles = 1000

shuffle_pal_results_dict = {}

for session, z_array in concat_z_fr_dict.items():

    n_trials, n_neurons, n_windows = z_array.shape
    real_pal_vector = pal_rank_dict[session]
    shuffle_pal_results_dict[session] = {}

    # Store shuffle results
    shuffle_mean_array = np.zeros((n_shuffles, n_windows))

    for sh in tqdm(range(n_shuffles), desc=f"{session} shuffles"):

        # 🔹 Shuffle trial labels
        shuffled_pal = np.random.permutation(real_pal_vector)

        shuffle_corr = np.zeros((n_neurons, n_windows))

        for neur in range(n_neurons):
            for window in range(n_windows):

                fr_vector = z_array[:, neur, window]

                rho, _ = spearmanr(fr_vector, shuffled_pal)

                shuffle_corr[neur, window] = rho**2

        # Mean across neurons
        shuffle_mean_array[sh] = np.nanmean(shuffle_corr, axis=0)

    # Compute 95% percentile across shuffles for each cluster × window
    shuffle_95behavior = np.percentile(shuffle_mean_array, 95, axis=0)  # shape: clusters x windows
    
    shuffle_pal_results_dict[session] = {
        "real_mean_r2": r2_mean_pal_dict[session],
        "shuffle_mean_r2": shuffle_mean_array,
        "shuffle_95perc_r2": shuffle_95behavior
    }


# Behavior score shuffle control 
# Shuffles behavior scores across trials, preserving neural data

n_shuffles = 1000

shuffle_behavior_results_dict = {}

for session, z_array in concat_z_fr_dict.items():

    this_behavior_array = behavior_score_dict[session]['behavior_array']
    n_clusters = this_behavior_array.shape[1]

    n_trials, n_neurons, n_windows = z_array.shape

    # store: shuffles × clusters × windows
    shuffle_mean_array = np.zeros((n_shuffles, n_clusters, n_windows))

    for sh in tqdm(range(n_shuffles), desc=f"{session} shuffles"):

        for cluster in range(n_clusters):

            shuffled_behavior = np.random.permutation(
                this_behavior_array[:, cluster]
            )

            shuffle_corr = np.zeros((n_neurons, n_windows))

            for neur in range(n_neurons):
                for window in range(n_windows):

                    fr_vector = z_array[:, neur, window]

                    rho, _ = spearmanr(fr_vector, shuffled_behavior)

                    shuffle_corr[neur, window] = rho**2

            shuffle_mean_array[sh, cluster] = np.nanmean(
                shuffle_corr, axis=0
            )
    
    # Compute 95% percentile across shuffles for each cluster × window
    shuffle_95behavior = np.percentile(shuffle_mean_array, 95, axis=0)  # shape: clusters x windows
    
    
    shuffle_behavior_results_dict[session] = {
    "real_mean_r2": r2_mean_behavior_dict[session],  # shape: clusters x windows
    "shuffle_mean_r2": shuffle_mean_array,   # shape: n_shuffles x clusters x windows
    "shuffle_95perc_r2": shuffle_95behavior
    
    }
   

# Loop through sessions
for session in concat_z_fr_dict.keys():
    
    # Grab any taste to get the results_dir
    any_taste = list(spike_trains_dict[session].keys())[0]
    results_dir = spike_trains_dict[session][any_taste]['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # -----------------------------
    # Palatability shuffle
    # -----------------------------
    pal_shuffle_file = os.path.join(results_dir, f"{session}_pal_shuffle.pkl")
    with open(pal_shuffle_file, "wb") as f:
        pickle.dump(shuffle_pal_results_dict[session], f)
    
    # -----------------------------
    # Behavior shuffle
    # -----------------------------
    behavior_shuffle_file = os.path.join(results_dir, f"{session}_behavior_shuffle.pkl")
    with open(behavior_shuffle_file, "wb") as f:
        pickle.dump(shuffle_behavior_results_dict[session], f)


#%% Plotting average real and shuffle correlations across neurons for each session
   
# Plots real data correlations for palatability rank and each of the behaviors

start_time = -500
time_bin = np.arange(start_time, 2000 - window_size, step_size)

behavior_names = ['No Movement', 'Gapes', 'MTMs']

for session in concat_z_fr_dict.keys():

    # Grab any taste to get results_dir
    any_taste = list(spike_trains_dict[session].keys())[0]
    results_dir = spike_trains_dict[session][any_taste]['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # Palatability R²
    pal_r2 = r2_mean_pal_dict[session]

    # Behavior R² (clusters × timebins)
    behavior_r2_all = r2_mean_behavior_dict[session]

    plt.figure(figsize=(10, 6))

    # Plot palatability
    plt.plot(time_bin, pal_r2,
             label="Palatability",
             color='black',
             lw=3)
   
    # Plot each behavior cluster with proper label
    n_clusters = behavior_r2_all.shape[0]

    for cluster_idx in range(n_clusters):
        cluster_r2 = behavior_r2_all[cluster_idx, :]
        label = behavior_names[cluster_idx] if cluster_idx < len(behavior_names) else f"Cluster {cluster_idx}"
        
        plt.plot(time_bin, cluster_r2,
                 lw=2,
                 alpha=0.85,
                 label=label)

    plt.xlabel("Time (ms)")
    plt.ylabel("Mean R² (neurons averaged)")
    plt.title(f"Session: {session} — Real Correlations")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save
    save_file = os.path.join(results_dir, f"{session}_real_correlations_all_clusters.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()


# Real data and shuffle data on same axes
for session in concat_z_fr_dict.keys():

    # Grab any taste to get results_dir
    any_taste = list(spike_trains_dict[session].keys())[0]
    results_dir = spike_trains_dict[session][any_taste]['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------
    # Palatability
    # -----------------------------
    pal_real = shuffle_pal_results_dict[session]["real_mean_r2"]
    pal_shuf95 = shuffle_pal_results_dict[session]["shuffle_95perc_r2"]

    plt.figure(figsize=(12, 6))
    
    # Plot palatability
    plt.plot(time_bin, pal_real, label="Palatability Real R²", color='blue', lw=2)
    plt.fill_between(time_bin, pal_shuf95, color='blue', alpha=0.3, label="Palatability 95% shuffle")


    # -----------------------------
    # Behavior
    # -----------------------------
    for cluster_idx in range(n_clusters):
        cluster_r2 = shuffle_behavior_results_dict[session]["real_mean_r2"][cluster_idx, :]
        label = behavior_names[cluster_idx] if cluster_idx < len(behavior_names) else f"Cluster {cluster_idx}"
        cluster_95 = shuffle_behavior_results_dict[session]["shuffle_95perc_r2"] [cluster_idx, :] 
        plt.plot(time_bin, cluster_r2,
                 lw=2,
                 alpha=0.85,
                 label=label)
        plt.fill_between(time_bin, cluster_95, color='gray', alpha=0.3, label=f"{label} 95% shuffle")
        
    
    plt.xlabel("Time bin")
    plt.ylabel("Mean R²")
    plt.title(f"Session: {session} — Real vs Shuffle Correlations")
    plt.legend()
    plt.tight_layout()

    # Save figure
    save_file = os.path.join(results_dir, f"{session}_real_vs_shuffle_correlations.png")
    plt.savefig(save_file)
    plt.close()


#%%  Plotting real and shuffle correlations across all sessions in text file

# -----------------------------
# Settings
# -----------------------------
behavior_names = ['No Movement', 'Gapes', 'MTMs']
start_time = -500
time_bin = np.arange(start_time, 2000 - window_size, step_size)

save_dir = "/media/cmazzio/large_data/GC_behavior_correlation_aggregate_session_plots"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# Collect session data
# -----------------------------
all_pal_r2 = []
all_behavior_r2 = []

all_pal_shuf95 = []
all_behavior_shuf95 = []

session_list = list(concat_z_fr_dict.keys())
basename_list = []

for session in session_list:

    # Real data
    all_pal_r2.append(r2_mean_pal_dict[session])                      # (windows)
    all_behavior_r2.append(r2_mean_behavior_dict[session])            # (clusters x windows)

    # Shuffle 95th percentile
    all_pal_shuf95.append(
        shuffle_pal_results_dict[session]["shuffle_95perc_r2"]
    )

    all_behavior_shuf95.append(
        shuffle_behavior_results_dict[session]["shuffle_95perc_r2"]
    )
    
    # Create basename list list for saving
    basename = "_".join(session.split("_")[:2])
    basename_list.append(basename)

# Convert to arrays
all_pal_r2 = np.array(all_pal_r2)                        # (sessions x windows)
all_behavior_r2 = np.array(all_behavior_r2)              # (sessions x clusters x windows)

all_pal_shuf95 = np.array(all_pal_shuf95)
all_behavior_shuf95 = np.array(all_behavior_shuf95)

# -----------------------------
# Average across sessions ONLY
# -----------------------------
mean_pal_r2 = np.nanmean(all_pal_r2, axis=0)
mean_behavior_r2 = np.nanmean(all_behavior_r2, axis=0)

mean_pal_shuf95 = np.nanmean(all_pal_shuf95, axis=0)
mean_behavior_shuf95 = np.nanmean(all_behavior_shuf95, axis=0)

n_clusters = mean_behavior_r2.shape[0]

# ============================================================
# 1️⃣ Aggregate Real Correlations (No Shuffle)
# ============================================================
plt.figure(figsize=(12,6))

# Palatability
plt.plot(time_bin, mean_pal_r2,
         color='black', lw=3,
         label='Palatability')

# Each behavior cluster
for cluster_idx in range(n_clusters):

    label = behavior_names[cluster_idx] \
        if cluster_idx < len(behavior_names) \
        else f"Cluster {cluster_idx}"

    plt.plot(time_bin,
             mean_behavior_r2[cluster_idx],
             lw=2,
             alpha=0.9,
             label=label)

plt.xlabel("Time (ms)")
plt.ylabel("Mean R² (across sessions)")
plt.title("Aggregate Real Correlations")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig(os.path.join(save_dir,
            f"{basename_list}_aggregate_real_correlations_all_clusters.png"),
            dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 2️⃣ Aggregate Real + Shuffle (Cluster-Specific)
# ============================================================
plt.figure(figsize=(12,6))

# Palatability
plt.plot(time_bin, mean_pal_r2,
         color='black', lw=3,
         label='Palatability')

plt.fill_between(time_bin,
                 mean_pal_shuf95,
                 color='black',
                 alpha=0.2)

# Behavior clusters
for cluster_idx in range(n_clusters):

    label = behavior_names[cluster_idx] \
        if cluster_idx < len(behavior_names) \
        else f"Cluster {cluster_idx}"

    plt.plot(time_bin,
             mean_behavior_r2[cluster_idx],
             lw=2,
             label=label)

    plt.fill_between(time_bin,
                     mean_behavior_shuf95[cluster_idx],
                     alpha=0.15)

plt.xlabel("Time (ms)")
plt.ylabel("Mean R²")
plt.title("Aggregate Real vs Shuffle (All Clusters)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig(os.path.join(save_dir,
            f"{basename_list}_aggregate_real_vs_shuffle_all_clusters.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print("Aggregate plots saved to:", save_dir)


