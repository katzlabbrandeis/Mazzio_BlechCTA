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
import umap
from sklearn.decomposition import PCA
import pingouin as pg

# base_dir = '/home/cmazzio/Desktop/Mazzio_BlechCTA'
base_dir = '/media/bigdata/firing_space_plot/Mazzio_BlechCTA'
# src_dir = '/home/cmazzio/Desktop/Mazzio_BlechCTA/src/lfp_analysis'
# artifacts_dir = '/home/cmazzio/Desktop/Mazzio_BlechCTA/artifacts/lfp_analysis'
# plot_dir = '/home/cmazzio/Desktop/Mazzio_BlechCTA/plots/lfp_analysis'
src_dir = os.path.join(base_dir, 'src', 'lfp_analysis')
artifacts_dir = os.path.join(base_dir, 'artifacts', 'lfp_analysis')
plot_dir = os.path.join(base_dir, 'plots', 'lfp_analysis')
os.makedirs(artifacts_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

lfp_med_out_dir = os.path.join(artifacts_dir, 'pre_stim_data')

############################################################
# Extraction from original data 
# (takes a long time, and uses a lot of memory, so we save the extracted data for easier access in future steps)
############################################################
data_dirs_file = os.path.join(src_dir, 'data_dirs_LFP.txt')
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

############################################################
############################################################

##############################
# Compile everything into a single dataframe for easier access
compiled_data_file_list = os.listdir(lfp_med_out_dir)
compiled_data = [
    pd.read_pickle(os.path.join(lfp_med_out_dir, f)) for f in compiled_data_file_list if f.endswith('.pkl')
]
compiled_df = pd.DataFrame(compiled_data)
# 'pre_stim_med' column contains list of arrays (one for each taste) with shape (n_trials, n_freqs) 

# Extract behavioral changepoints and try to align with LFP data (not sure if this will work but worth a try)
# transition_data_dir = '/media/bigdata/firing_space_plot/emg_analysis/CM_behavior_transitions/data'
# *** Repo was reorganized ***
transition_data_dir = '/media/bigdata/firing_space_plot/Mazzio_BlechCTA/data/multibehavior_transition'
# out_df_file = 'behavior_changepoint_out_df.pkl'
# with open(os.path.join(data_dir, out_df_file), 'wb') as f:
#     pd.to_pickle(out_df, f)
behavior_chp_df = pd.read_pickle(os.path.join(transition_data_dir, 'behavior_changepoint_out_df.pkl'))
# Index(['basename', 'taste', 'taste_trials', 'taste_behavior_array',
# 'zscored_taste_behavior_array', 'tau_samples', 'mode_changepoint'],

# Check if 'basename' in behavior_chp_df matches 'animal' in pre_stim_data 
# set(behavior_chp_df['basename']).intersection(set(compiled_df['animal']))
len(np.setdiff1d(
    behavior_chp_df['basename'].unique(), 
    compiled_df['animal'].unique()
))

# Make plots

this_plot_dir = os.path.join(plot_dir, 'pre_stim_spectrograms')
os.makedirs(this_plot_dir, exist_ok=True)

for ind, this_row in tqdm(compiled_df.iterrows(), total=len(compiled_df)):
    pre_stim_med = this_row['pre_stim_med']
    taste_list = this_row['taste_list']
    freq_vec = this_row['freq_vec']

    assert len(pre_stim_med) == len(taste_list), f"Expected pre_stim_med and taste_list to have same length, got {len(pre_stim_med)} and {len(taste_list)}"

    # Try to find corresponding behavior changepoint data for this animal
    chp_data = behavior_chp_df.query(f'basename == "{this_row["animal"]}"')

    fig, ax = plt.subplots(2, len(taste_list), figsize=(2*len(taste_list), 5),
                           sharex=True, sharey='col')
    for i, taste in enumerate(taste_list):
        # Make taste lower to match with behavior changepoint data
        taste = taste.lower()
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

        if len(chp_data) > 0:
            chp_row = chp_data.query(f'taste == "{taste}"')
            if len(chp_row) > 0:
                mode_chp = chp_row['mode_changepoint'].values[0]
                ax[0,i].axhline(mode_chp, color='red', linestyle='--', label='Behavior Changepoint')
                ax[1,i].axhline(mode_chp, color='red', linestyle='--', label='Behavior Changepoint')

            # Put legend below the figure
            handles, labels = ax[0,i].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center') 

    fig.suptitle(f"Pre-stimulus LFP Spectrogram for {this_row['animal']}")
    plt.tight_layout()
    plt_path = os.path.join(this_plot_dir, f"{this_row['animal']}_pre_stim_spectrogram.png")
    plt.savefig(plt_path)
    plt.close()

##############################
# For matching lfp and behavior data, plot average before and after changepoint for each taste, and see if there are any differences in the spectrograms 

chp_lfp_data = []
for ind, this_row in tqdm(compiled_df.iterrows(), total=len(compiled_df)):
    pre_stim_med = this_row['pre_stim_med']
    taste_list = this_row['taste_list']
    freq_vec = this_row['freq_vec']

    assert len(pre_stim_med) == len(taste_list), f"Expected pre_stim_med and taste_list to have same length, got {len(pre_stim_med)} and {len(taste_list)}"

    # Try to find corresponding behavior changepoint data for this animal
    chp_data = behavior_chp_df.query(f'basename == "{this_row["animal"]}"')
    if len(chp_data) == 0:
        continue

    for i, taste in enumerate(taste_list):
        # Make taste lower to match with behavior changepoint data
        taste = taste.lower()
        chp_row = chp_data.query(f'taste == "{taste}"')
        if len(chp_row) == 0:
            continue
        mode_chp = chp_row['mode_changepoint'].values[0]

        pre_chp_data = pre_stim_med[i][:mode_chp, :]
        post_chp_data = pre_stim_med[i][mode_chp:, :]

        pre_chp_med = np.median(pre_chp_data, axis=0)
        post_chp_med = np.median(post_chp_data, axis=0)

        out_dict = {
            'animal': this_row['animal'],
            'taste': taste,
            'pre_chp_med': pre_chp_med,
            'post_chp_med': post_chp_med,
            'freq_vec': freq_vec
        }
        chp_lfp_data.append(out_dict)

chp_lfp_df = pd.DataFrame(chp_lfp_data)
# Explode pre_chp_med, post_chp_med, and freq_vec into long format for easier plotting
# chp_lfp_long_df = chp_lfp_df.explode(['pre_chp_med', 'post_chp_med', 'freq_vec'])

# Plot pre and post changepoint medians for each taste, stacked
med_diff_data_z_list = []
MAD_diff_data_z_list = []
n_tastes = chp_lfp_df['taste'].nunique()
fig, ax = plt.subplots(n_tastes, 4, figsize=(15, 3*n_tastes),
                       sharex=True) 
taste_grouped = chp_lfp_df.groupby('taste')
for ind, (taste_name, this_df) in enumerate(taste_grouped):
    pre_data = np.stack(this_df['pre_chp_med'].values)
    post_data = np.stack(this_df['post_chp_med'].values)
    
    # zscore across animals for each frequency, but collectively for pre and post data to make them comparable
    combined_data = np.concatenate([pre_data, post_data], axis=0)
    combined_mean = np.nanmean(combined_data, axis=0)
    combined_std = np.nanstd(combined_data, axis=0)
    combined_data_z = (combined_data - combined_mean) / combined_std
    pre_data_z = combined_data_z[:pre_data.shape[0], :]
    post_data_z = combined_data_z[pre_data.shape[0]:, :]

    diff_data_z = post_data_z - pre_data_z
    # Drop any rows with NaN values (in case there are frequencies with NaN values for some animals)
    valid_inds = ~np.isnan(diff_data_z).any(axis=1)
    valid_diff_data_z = diff_data_z[valid_inds, :]
    med_diff_data_z = np.median(valid_diff_data_z, axis=0)
    MAD_diff_data_z = np.median(np.abs(valid_diff_data_z - med_diff_data_z), axis=0)
    med_diff_data_z_list.append(med_diff_data_z)
    MAD_diff_data_z_list.append(MAD_diff_data_z)

    freq_vec = this_df['freq_vec'].values[0]

    im = ax[ind, 0].pcolormesh(
        freq_vec,
        np.arange(pre_data.shape[0]),
        pre_data_z,
        shading='auto'
        )
    ax[ind, 0].set_title(f"{taste_name} - Pre Changepoint")
    ax[ind, 0].set_ylabel('Animal')
    ax[ind, 0].set_xlabel('Frequency (Hz)')
    im = ax[ind, 1].pcolormesh(
        freq_vec,
        np.arange(post_data.shape[0]),
        post_data_z,
        shading='auto'
        )
    ax[ind, 1].set_title(f"{taste_name} - Post Changepoint")
    ax[ind, 1].set_ylabel('Animal')
    ax[ind, 1].set_xlabel('Frequency (Hz)')
    im = ax[ind, 2].pcolormesh(
        freq_vec,
        np.arange(pre_data.shape[0]),
        diff_data_z,
        shading='auto',
        cmap='bwr',
        )
    ax[ind, 2].set_title(f"{taste_name} - Post - Pre Changepoint")
    ax[ind, 2].set_ylabel('Animal')
    ax[ind, 2].set_xlabel('Frequency (Hz)')
    ax[ind, 3].errorbar(
        freq_vec,
        med_diff_data_z,
        yerr=MAD_diff_data_z,
        fmt='-o'
        )
    ax[ind, 3].set_title(f"{taste_name} - Median Post - Pre Changepoint")
    ax[ind, 3].set_ylabel('Difference of Z-scored Power')
    ax[ind, 3].set_xlabel('Frequency (Hz)')
    ax[ind, 3].axhline(0, color='k', linestyle='--', linewidth=1.5)
fig.suptitle("Pre and Post Changepoint Median Spectrograms for Each Taste")
plt.tight_layout()
plt_path = os.path.join(plot_dir, 'pre_post_changepoint_spectrograms.png')
plt.savefig(plt_path, bbox_inches='tight')
plt.close()

# Break down into frequency bands and plot
freq_bands = {
        'delta': [1, 4],
        'theta': [4, 8],
        'alpha': [8, 12],
        'beta': [12, 30],
        'low_gamma': [30, 70],
        'high_gamma': [70, 120]
        }

# Calculate band power for pre and post changepoint for each animal and taste
band_power_data = []
for ind, this_row in tqdm(compiled_df.iterrows(), total=len(compiled_df)):
    pre_stim_med = this_row['pre_stim_med']
    taste_list = this_row['taste_list']
    freq_vec = this_row['freq_vec']

    # Try to find corresponding behavior changepoint data for this animal
    chp_data = behavior_chp_df.query(f'basename == "{this_row["animal"]}"')
    if len(chp_data) == 0:
        continue

    for i, taste in enumerate(taste_list):
        # Make taste lower to match with behavior changepoint data
        taste = taste.lower()
        chp_row = chp_data.query(f'taste == "{taste}"')
        if len(chp_row) == 0:
            continue
        mode_chp = chp_row['mode_changepoint'].values[0]

        pre_chp_data = pre_stim_med[i][:mode_chp, :]
        post_chp_data = pre_stim_med[i][mode_chp:, :]

        # Calculate median power for each frequency band
        for band_name, band_range in freq_bands.items():
            band_inds = np.where((freq_vec >= band_range[0]) & (freq_vec <= band_range[1]))[0]
            if len(band_inds) == 0:
                continue
            
            # Calculate median across frequencies in band, then median across trials
            pre_band_power = np.median(np.median(pre_chp_data[:, band_inds], axis=1))
            post_band_power = np.median(np.median(post_chp_data[:, band_inds], axis=1))

            out_dict = {
                'animal': this_row['animal'],
                'taste': taste,
                'band': band_name,
                'pre_power': pre_band_power,
                'post_power': post_band_power
            }
            band_power_data.append(out_dict)

band_power_df = pd.DataFrame(band_power_data)

# Normalize band power data to mean of pre+post for each band and animal
normalized_band_power_data = []
for idx, row in band_power_df.iterrows():
    mean_power = (row['pre_power'] + row['post_power']) / 2
    norm_pre = row['pre_power'] / mean_power if mean_power > 0 else 0
    norm_post = row['post_power'] / mean_power if mean_power > 0 else 0
    
    normalized_band_power_data.append({
        'animal': row['animal'],
        'taste': row['taste'],
        'band': row['band'],
        'norm_pre_power': norm_pre,
        'norm_post_power': norm_post
    })

norm_band_power_df = pd.DataFrame(normalized_band_power_data)

# Create bar plots with paired scatter plots for each taste
n_tastes = norm_band_power_df['taste'].nunique()
n_bands = len(freq_bands)
fig, axes = plt.subplots(1, n_tastes, figsize=(4*n_tastes, 5), sharey=True)
if n_tastes == 1:
    axes = [axes]

taste_grouped = norm_band_power_df.groupby('taste')
for ax_ind, (taste_name, taste_df) in enumerate(taste_grouped):
    ax = axes[ax_ind]
    
    # Calculate mean and SEM for each band
    band_grouped = taste_df.groupby('band')
    band_names = list(freq_bands.keys())
    x_positions = np.arange(len(band_names))
    bar_width = 0.35
    
    pre_means = []
    pre_sems = []
    post_means = []
    post_sems = []
    p_values = []
    
    for band_name in band_names:
        band_data = taste_df[taste_df['band'] == band_name]
        if len(band_data) > 0:
            pre_means.append(band_data['norm_pre_power'].mean())
            pre_sems.append(band_data['norm_pre_power'].sem())
            post_means.append(band_data['norm_post_power'].mean())
            post_sems.append(band_data['norm_post_power'].sem())
            
            # Perform paired t-test
            if len(band_data) > 1:
                t_stat, p_val = stats.ttest_rel(band_data['norm_pre_power'], band_data['norm_post_power'])
                p_values.append(p_val)
            else:
                p_values.append(np.nan)
        else:
            pre_means.append(0)
            pre_sems.append(0)
            post_means.append(0)
            post_sems.append(0)
            p_values.append(np.nan)
    
    # Plot bars
    ax.bar(x_positions - bar_width/2, pre_means, bar_width, 
           yerr=pre_sems, label='Pre-changepoint', alpha=0.7, capsize=5)
    ax.bar(x_positions + bar_width/2, post_means, bar_width,
           yerr=post_sems, label='Post-changepoint', alpha=0.7, capsize=5)
    
    # Overlay paired scatter plots
    for band_idx, band_name in enumerate(band_names):
        band_data = taste_df[taste_df['band'] == band_name]
        if len(band_data) > 0:
            # Add jitter to x positions for visibility
            jitter = 0.05
            pre_x = np.random.normal(x_positions[band_idx] - bar_width/2, jitter, size=len(band_data))
            post_x = np.random.normal(x_positions[band_idx] + bar_width/2, jitter, size=len(band_data))
            
            # Plot individual animal data points
            for i, (idx, row) in enumerate(band_data.iterrows()):
                ax.plot([pre_x[i], post_x[i]], [row['norm_pre_power'], row['norm_post_power']], 
                       'o-', color='gray', alpha=0.5, markersize=4, linewidth=1)
    
    # Add significance stars above bars
    y_max = max(max(pre_means), max(post_means))
    for band_idx, p_val in enumerate(p_values):
        if not np.isnan(p_val):
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            # Place significance marker above the bars
            y_pos = y_max * 1.1
            ax.text(x_positions[band_idx], y_pos, sig_text, 
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Normalized Power (to mean)')
    ax.set_title(f'{taste_name.capitalize()}')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(band_names, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)  # Reference line at normalized mean

fig.suptitle(f'Normalized Band Power: Pre vs Post Changepoint\n(* p<0.05, ** p<0.01, *** p<0.001, ns=not significant)\n{freq_bands}', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt_path = os.path.join(plot_dir, 'band_power_pre_post_changepoint_paired_normalized.png')
plt.savefig(plt_path, bbox_inches='tight', dpi=300)
plt.close()

# Prepare data for 2-way ANOVA (pre/post x taste)
# Convert normalized band power data to long format for ANOVA
anova_data_list = []
for idx, row in norm_band_power_df.iterrows():
    # Add pre-changepoint data
    anova_data_list.append({
        'animal': row['animal'],
        'taste': row['taste'],
        'band': row['band'],
        'condition': 'pre',
        'power': row['norm_pre_power']
    })
    # Add post-changepoint data
    anova_data_list.append({
        'animal': row['animal'],
        'taste': row['taste'],
        'band': row['band'],
        'condition': 'post',
        'power': row['norm_post_power']
    })

anova_df = pd.DataFrame(anova_data_list)

# Perform 2-way ANOVA for each frequency band
print("\n" + "="*80)
print("2-WAY ANOVA RESULTS: Condition (Pre/Post) x Taste")
print("="*80)

anova_results_list = []
for band_name in freq_bands.keys():
    band_anova_df = anova_df[anova_df['band'] == band_name]
    
    if len(band_anova_df) > 0 and len(band_anova_df['taste'].unique()) > 1:
        # Perform 2-way repeated measures ANOVA
        # Note: Using mixed_anova since we have within-subject factor (condition) and between-subject factor (taste)
        # However, if same animals are in all tastes, we should use rm_anova with both factors
        
        # Check if we have repeated measures (same animals across tastes)
        animals_per_taste = band_anova_df.groupby('taste')['animal'].apply(set)
        all_animals = set.union(*animals_per_taste.values)
        
        # Use rm_anova if we have the same animals across conditions
        try:
            aov = pg.rm_anova(
                data=band_anova_df,
                dv='power',
                within=['condition', 'taste'],
                subject='animal',
                detailed=True
            )
            anova_type = 'Repeated Measures ANOVA'
        except:
            # If rm_anova fails (e.g., unbalanced design), use mixed_anova
            try:
                aov = pg.mixed_anova(
                    data=band_anova_df,
                    dv='power',
                    within='condition',
                    between='taste',
                    subject='animal',
                    correction=True
                )
                anova_type = 'Mixed ANOVA'
            except:
                # If both fail, skip this band
                print(f"\n{band_name.upper()} ({freq_bands[band_name][0]}-{freq_bands[band_name][1]} Hz):")
                print("  Could not perform ANOVA (insufficient data or unbalanced design)")
                continue
        
        print(f"\n{band_name.upper()} ({freq_bands[band_name][0]}-{freq_bands[band_name][1]} Hz) - {anova_type}:")
        print(aov.to_string())
        
        # Store results
        aov['band'] = band_name
        aov['anova_type'] = anova_type
        anova_results_list.append(aov)

if anova_results_list:
    anova_results_df = pd.concat(anova_results_list, ignore_index=True)
    # Save ANOVA results
    anova_results_path = os.path.join(artifacts_dir, 'band_power_anova_results.csv')
    anova_results_df.to_csv(anova_results_path, index=False)
    print(f"\nANOVA results saved to: {anova_results_path}")

print("="*80 + "\n")

# Create bar plots organized by frequency band (subplots) with tastes on x-axis
n_bands = len(freq_bands)
fig, axes = plt.subplots(1, n_bands, figsize=(5*n_bands, 5), sharey=True)
if n_bands == 1:
    axes = [axes]

band_names = list(freq_bands.keys())
for band_idx, band_name in enumerate(band_names):
    ax = axes[band_idx]
    
    # Get data for this band across all tastes
    band_data = norm_band_power_df[norm_band_power_df['band'] == band_name]
    
    if len(band_data) == 0:
        ax.set_visible(False)
        continue
    
    # Get unique tastes and calculate statistics
    tastes = sorted(band_data['taste'].unique())
    x_positions = np.arange(len(tastes))
    bar_width = 0.35
    
    pre_means = []
    pre_sems = []
    post_means = []
    post_sems = []
    p_values = []
    
    for taste in tastes:
        taste_data = band_data[band_data['taste'] == taste]
        if len(taste_data) > 0:
            pre_means.append(taste_data['norm_pre_power'].mean())
            pre_sems.append(taste_data['norm_pre_power'].sem())
            post_means.append(taste_data['norm_post_power'].mean())
            post_sems.append(taste_data['norm_post_power'].sem())
            
            # Perform paired t-test
            if len(taste_data) > 1:
                t_stat, p_val = stats.ttest_rel(taste_data['norm_pre_power'], taste_data['norm_post_power'])
                p_values.append(p_val)
            else:
                p_values.append(np.nan)
        else:
            pre_means.append(0)
            pre_sems.append(0)
            post_means.append(0)
            post_sems.append(0)
            p_values.append(np.nan)
    
    # Plot bars
    ax.bar(x_positions - bar_width/2, pre_means, bar_width, 
           yerr=pre_sems, label='Pre-changepoint', alpha=0.7, capsize=5)
    ax.bar(x_positions + bar_width/2, post_means, bar_width,
           yerr=post_sems, label='Post-changepoint', alpha=0.7, capsize=5)
    
    # Overlay paired scatter plots
    for taste_idx, taste in enumerate(tastes):
        taste_data = band_data[band_data['taste'] == taste]
        if len(taste_data) > 0:
            # Add jitter to x positions for visibility
            jitter = 0.05
            pre_x = np.random.normal(x_positions[taste_idx] - bar_width/2, jitter, size=len(taste_data))
            post_x = np.random.normal(x_positions[taste_idx] + bar_width/2, jitter, size=len(taste_data))
            
            # Plot individual animal data points
            for i, (idx, row) in enumerate(taste_data.iterrows()):
                ax.plot([pre_x[i], post_x[i]], [row['norm_pre_power'], row['norm_post_power']], 
                       'o-', color='gray', alpha=0.5, markersize=4, linewidth=1)
    
    # Add significance stars above bars
    y_max = max(max(pre_means), max(post_means)) if pre_means and post_means else 1
    for taste_idx, p_val in enumerate(p_values):
        if not np.isnan(p_val):
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            # Place significance marker above the bars
            y_pos = y_max * 1.1
            ax.text(x_positions[taste_idx], y_pos, sig_text, 
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Taste')
    ax.set_ylabel('Normalized Power (to mean)')
    band_range = freq_bands[band_name]
    ax.set_title(f'{band_name.capitalize()} ({band_range[0]}-{band_range[1]} Hz)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([t.capitalize() for t in tastes], rotation=45, ha='right')
    if band_idx == 0:
        ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)  # Reference line at normalized mean

fig.suptitle('Normalized Band Power by Taste: Pre vs Post Changepoint\n(* p<0.05, ** p<0.01, *** p<0.001, ns=not significant)', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt_path = os.path.join(plot_dir, 'band_power_by_taste_pre_post_changepoint_paired_normalized.png')
plt.savefig(plt_path, bbox_inches='tight', dpi=300)
plt.close()

##############################
# Plot median difference of z-scored power between pre and post changepoint for each taste, with filled area representing MAD 
fig, ax = plt.subplots(figsize=(5, 5))
for ind, taste_name in enumerate(taste_grouped.groups.keys()):
    freq_vec = chp_lfp_df.query(f'taste == "{taste_name}"')['freq_vec'].values[0]
    med_diff_data_z = med_diff_data_z_list[ind]
    MAD_diff_data_z = MAD_diff_data_z_list[ind]
    ax.plot(freq_vec, med_diff_data_z, label=taste_name, linewidth=2)
    ax.fill_between(freq_vec, med_diff_data_z - MAD_diff_data_z, med_diff_data_z + MAD_diff_data_z, alpha=0.3)
ax.set_title("Median Difference of Z-scored Power Between Pre and Post Changepoint for Each Taste")
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Difference of Z-scored Power')
ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
ax.legend()
plt.tight_layout()
plt_path = os.path.join(plot_dir, 'median_diff_z_power_pre_post_changepoint.png')
plt.savefig(plt_path, bbox_inches='tight')
plt.close()

# Plot umap of pre and post changepoint data for each taste, colored by pre vs post changepoint
embedding_data = []
taste_grouped = chp_lfp_df.groupby('taste')
for taste_name, this_df in taste_grouped:
    pre_data = np.stack(this_df['pre_chp_med'].values)
    post_data = np.stack(this_df['post_chp_med'].values)
    data = np.concatenate([pre_data, post_data], axis=0)
    labels = ['pre'] * pre_data.shape[0] + ['post'] * post_data.shape[0]

    # First do PCA to reduce dimensionality to 90% variance explained, then do UMAP on the PCA components
    pca = PCA(n_components=0.9)
    pca_data = pca.fit_transform(data.T).T

    # If >3 dims, do UMAP to reduce to 2D for visualization
    if pca_data.shape[0] < 3:
        transform_name = 'pca'
        plot_data = pca_data.T
    else:
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(pca_data.T)
        transform_name = 'pca + umap'
        plot_data = embedding
    out_dict = {
        'taste': taste_name,
        'data': plot_data,
        'labels': labels,
        'transform': transform_name
    }
    embedding_data.append(out_dict)
