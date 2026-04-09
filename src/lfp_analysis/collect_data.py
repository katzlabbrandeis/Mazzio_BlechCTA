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
import matplotlib.pyplot as plt
from scipy import stats

base_dir = '/home/cmazzio/Desktop/Mazzio_BlechCTA/src/lfp_analysis'
artifacts_dir = '/home/cmazzio/Desktop/Mazzio_BlechCTA/artifacts/lfp_analysis'
plot_dir = '/home/cmazzio/Desktop/Mazzio_BlechCTA/plots/lfp_analysis'
os.makedirs(artifacts_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
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
lfp_med_out_dir = os.path.join(artifacts_dir, 'pre_stim_data')
os.makedirs(lfp_med_out_dir, exist_ok=True)

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
    # Pul out everything else too
    time_vec = data['time_vec']
    freq_vec = data['freq_vec']
    info_dict = data['info_dict']
    taste_list = data['taste_list']
    amp_array_shape = data['amp_array_shape']

    # Remove data because it is a very large array
    # Don't want computer to crash during overwrite of variable
    del data

    # Only keep prestimulus time points
    # Assuming time is 0-5sec with 2 seconds pre-stimulus, 3 seconds post-stimulus, and time_vec is in seconds 
    pre_stim_range = [0.5, 1.5] # Given stim=2, we want to look at 0.5-1.5 seconds to avoid edge effects
    pre_stim_inds = np.where((time_vec >= pre_stim_range[0]) & (time_vec <= pre_stim_range[1]))[0]
    # Take median for that time range
    pre_stim_med = [np.median(x[..., pre_stim_inds], axis=-1) for x in amp_array_list]
    pre_stim_data.append(pre_stim_med)

    # Also save the pre-stim data for this animal as process is time consuming and we don't want to have to repeat it
    out_path = os.path.join(lfp_med_out_dir, f"{row['animal']}_pre_stim_data.pkl")
    out_dict = {
        'animal': row['animal'],
        'pre_stim_med': pre_stim_med,
        'time_vec': time_vec,
        'freq_vec': freq_vec,
        'info_dict': info_dict,
        'taste_list': taste_list,
        'amp_array_shape': amp_array_shape
    }
    with open(out_path, 'wb') as f:
        pd.to_pickle(out_dict, f)

# Compile everything into a single dataframe for easier access
compiled_data_file_list = os.listdir(lfp_med_out_dir)
compiled_data = [
    pd.read_pickle(os.path.join(lfp_med_out_dir, f)) for f in compiled_data_file_list if f.endswith('.pkl')
]
compiled_df = pd.DataFrame(compiled_data)
# 'pre_stim_med' column contains list of arrays (one for each taste) with shape (n_trials, n_freqs) 
# Make plots

for ind, this_row in tqdm(compiled_df.iterrows(), total=len(compiled_df)):
    pre_stim_med = this_row['pre_stim_med']
    taste_list = this_row['taste_list']
    freq_vec = this_row['freq_vec']

    assert len(pre_stim_med) == len(taste_list), f"Expected pre_stim_med and taste_list to have same length, got {len(pre_stim_med)} and {len(taste_list)}"

    fig, ax = plt.subplots(2, len(taste_list), figsize=(2*len(taste_list), 5),
                           sharex=True, sharey='col')
    for i, taste in enumerate(taste_list):
        im = ax[0,i].pcolormesh(
            freq_vec, 
            np.arange(pre_stim_med[i].shape[0]), 
            pre_stim_med[i], 
            shading='auto'
        )
        ax[0,i].set_title(taste)
        ax[0,i].set_ylabel('Trial')
        ax[0,i].set_xlabel('Frequency (Hz)')
        # Also plot zscored version (across trials for same frequency) of the data
        pre_stim_med_z = stats.zscore(pre_stim_med[i], axis=0)
        im = ax[1,i].pcolormesh(
            freq_vec, 
            np.arange(pre_stim_med_z.shape[0]), 
            pre_stim_med_z, 
            shading='auto'
        )
        ax[1,i].set_title(f"{taste} (z-scored)")
        ax[1,i].set_ylabel('Trial')
        ax[1,i].set_xlabel('Frequency (Hz)')

    fig.suptitle(f"Pre-stimulus LFP Spectrogram for {this_row['animal']}")
    plt.tight_layout()
    plt_path = os.path.join(plot_dir, f"{this_row['animal']}_pre_stim_spectrogram.png")
    plt.savefig(plt_path)
    plt.close()

