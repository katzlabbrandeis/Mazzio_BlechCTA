"""
Collect all data to analyze changes in pre-stimulus LFP across trials for all tastes
Original pkl files contain data for all channels.
We can pull out data for single channel and save for easier access

# File locations in:
data_dirs_LFP.txt

## From notes.txt:
Location of STFT files:
    Individual animal datadir → LFP_analyses → only pickle file inside 

Code:
    plot_LFP_spectrogram.py: runs STFT from ephys.data and saves pickle files with amplitude array and taste list
"""

import os
from tqdm import tqdm
from pprint import pprint as pp
import pandas as pd
import numpy as np

base_dir = '/home/cmazzio/Desktop/Mazzio_BlechCTA/src/lfp_analysis'
data_dirs_file = os.path.join(base_dir, 'data_dirs_LFP.txt')
# Read data directories from text file
with open(data_dirs_file, 'r') as f:
    data_dirs = [line.strip() for line in f.readlines()]

# Loop through each data directory and get path for each pickle file
pkl_files = []
for data_dir in tqdm(data_dirs):
    files = os.listdir(os.path.join(data_dir, 'LFP_analyses'))
    pkl_file = [f for f in files if f.endswith('.pkl')]
    assert len(pkl_file) == 1, f"Expected exactly one pickle file in {data_dir}/LFP_analyses, found {len(pkl_file)}"
    out_dict = {
        'animal': os.path.basename(data_dir),
        'data_dir': data_dir,
        'pkl_file': os.path.join(data_dir, 'LFP_analyses', pkl_file[0])
    }
    pkl_files.append(out_dict)

pkl_df = pd.DataFrame(pkl_files)

# Load each pickle file and extract data for single channel (e.g., channel 0)
pre_stim_data = []
for this_row in tqdm(pkl_df.iterrows(), total=len(pkl_df)):
    idx, row = this_row
    pkl_path = row['pkl_file']
    # Load pickle file (assuming it contains a dictionary with 'amplitude' and 'taste_list')
    with open(pkl_path, 'rb') as f:
        data = pd.read_pickle(f)
    
    # Streucture of data:
    # dict_keys(['amplitude_array_list', 'freq_vec', 'time_vec', 'amp_array_shape', 'info_dict', 'taste_list'])
    # Individual amplitude array shape: (n_trials, n_channels, n_freqs, n_timepoints)

    # Take median ampplitude across channels
    amp_array_list = [
            np.median(x, axis=1) for x in data['amplitude_array_list']
            ]

    # Remove data because it is a very large array
    # Don't want computer to crash during overwrite of variable
    del data

    # Only keep prestimulus time points
    # Assuming time is 0-5sec with 2 seconds pre-stimulus, 3 seconds post-stimulus, and time_vec is in seconds 
    pre_stim_range = [0.5, 1.5] # Given stim=2, we want to look at 0.5-1.5 seconds to avoid edge effects
    time_vec = data['time_vec']
    pre_stim_inds = np.where((time_vec >= pre_stim_range[0]) & (time_vec <= pre_stim_range[1]))[0]
    # Take median for that time range
    pre_stim_med = [np.median(x[..., pre_stim_inds], axis=-1) for x in amp_array_list]
    pre_stim_data.append(pre_stim_med)
