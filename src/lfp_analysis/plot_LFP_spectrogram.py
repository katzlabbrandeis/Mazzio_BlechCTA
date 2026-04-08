#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:22:19 2026

@author: cmazzio
"""
# IMPORTS
## Run this script from blech_spyder environment and within blech_clust dir ##

import blech_clust.utils.ephys_data.ephys_data as ephys_data
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import zscore
import os
import pickle

#%% Direct to Data

# Path to your text file
file_path = "/media/cmazzio/large_data/dataset_list/data_dirs_LFP"

# Read all lines and strip whitespace/newlines
with open(file_path, "r") as f:
    data_dirs = [line.strip() for line in f.readlines()]

print(f"Found {len(data_dirs)} directories:")
print(data_dirs)


#%% Function Defintion

def process_session(data_dir):
    basename = os.path.basename(data_dir)
    print(f"\nProcessing session: {basename}")

    # Initialize dat object
    dat = ephys_data.ephys_data(data_dir)
    dat.get_stft()
    dat.get_info_dict()
    info_dict = dat.info_dict

    taste_list = info_dict['taste_params']['tastes']
    amp_array_shape = [x.shape for x in dat.amplitude_array_list]

    # Set save directory
    save_dir = os.path.join(data_dir, 'LFP_analyses_high_freq')
    os.makedirs(save_dir, exist_ok=True)

    # Save variables
    save_path = os.path.join(save_dir, f"{basename}_STFT_data.pkl")
    data_to_save = {
        "amplitude_array_list": dat.amplitude_array_list,
        "freq_vec": dat.freq_vec,
        "time_vec": dat.time_vec,
        "amp_array_shape": amp_array_shape,
        "info_dict": info_dict,
        "taste_list": taste_list
    }
    with open(save_path, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"All variables saved to {save_path}")

    # Plot spectrogram
    num_tastes = len(dat.amplitude_array_list)
    fig, axes = plt.subplots(num_tastes, 1, figsize=(14, 14*num_tastes), sharex=True)
    if num_tastes == 1:
        axes = [axes]

    time_vec_ms = (dat.time_vec * 1000) - 2000

    # Frequency and time ticks
    y_min, y_max = dat.freq_vec.min(), dat.freq_vec.max()
    y_ticks = np.arange(y_min, y_max+1, 20)
    x_min, x_max = -500, 2000
    x_ticks = np.arange(x_min, x_max+1, 250)

    # Compute vmin/vmax for consistent color scale across tastes
    all_data = np.concatenate([zscore(np.median(dat.amplitude_array_list[taste], axis=(0,1)), axis=-1) 
                               for taste in range(num_tastes)], axis=None)
    vmin, vmax = np.min(all_data), np.max(all_data)

    for taste, ax in enumerate(axes):
        stft_dat = dat.amplitude_array_list[taste]
        med_stft = np.median(stft_dat, axis=(0,1))
        num_channels = stft_dat.shape[1]

        time_mask = (time_vec_ms >= x_min) & (time_vec_ms <= x_max)

        pcm = ax.pcolormesh(
            time_vec_ms[time_mask],
            dat.freq_vec,
            zscore(med_stft[:, time_mask], axis=-1),
            shading='auto',
            vmin=vmin, vmax=vmax
        )

        ax.axvline(0, color='red', linewidth=2, linestyle='--')
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_ylabel("")
        taste_name = taste_list[taste]
        ax.set_title(f"{taste_name} | {basename} | Channels: {num_channels}", fontsize=22)

    axes[-1].set_xlabel("Time (ms)", fontsize=22)
    fig.text(0.05, 0.5, "Frequency (Hz)", va='center', rotation='vertical', fontsize=22)
    fig.subplots_adjust(right=0.88, hspace=0.5)

    # Colorbar
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax, label="Z-scored power")
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_size(16)

    # Save figures
    fig_name_svg = os.path.join(save_dir, f'{basename}_STFT_spectrogram.svg')
    fig_name_png = os.path.join(save_dir, f'{basename}_STFT_spectrogram.png')
    fig.savefig(fig_name_svg, dpi=300, bbox_inches='tight')
    fig.savefig(fig_name_png, dpi=300, bbox_inches='tight')
    print(f"Figures saved to: {fig_name_svg}, {fig_name_png}")
    plt.close(fig)  # close figure to save memory



#%% Save STFT data and plot spectrogram for all data dirs

for dir_path in data_dirs:
    process_session(dir_path)

