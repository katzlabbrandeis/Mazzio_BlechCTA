# BlechCTA
Scripts used for Christina's Ph.D. thesis on neural mechanisms underlying learned and non-learned gaping.

## Repository Structure

- `src/` - Source code for analysis modules
  - `CM_scripts/` - Core analysis pipeline scripts
    - `combine_classifier_files.py` - Combines Blech EMG Classifier segment files
    - `create_tau_dict.py` - Combines tau, spike trains, and changepoint data
    - `initialize_dataframe.py` - Adds important metrics to dataframe
    - `extract_emg_from_transition.py` - Calculates behavior frequency per session
    - `extract_emg_from_transition_aggregate.py` - Aggregates behavior frequency across sessions
    - `neural_behavior_correlations_aggregate.py` - Neural-behavior correlation analysis
  - `gape_temporal_difference/` - Temporal analysis of gaping behavior
    - `gapes_temporal_features.py` - Feature extraction for gape analysis
  - `lfp_analysis/` - Local field potential analysis
    - `collect_data.py` - Data collection for LFP analysis
    - `cross_trial_analysis.py` - Cross-trial LFP analysis
    - `plot_LFP_spectrogram.py` - LFP spectrogram visualization
    - `data_dirs_LFP.txt` - Configuration file for LFP data directories
    - `notes.txt` - Analysis notes and documentation
  - `multibehavior_transition/` - Multi-behavior transition analysis
    - `multibehavior_transition.py` - Main transition analysis script
    - `DEPENDENCIES.md` - Dependencies documentation
    - `NOTE.txt` - Analysis notes and documentation
- `data/` - Processed data files
  - `changepoint_gapes/` - Changepoint detection data for gapes
    - `gapes_cp02_df.pkl` - Changepoint 0-2 dataframe
    - `gapes_cp23_df.pkl` - Changepoint 2-3 dataframe
- `artifacts/` - Generated analysis artifacts
  - `changepoint_gapes/` - Sorted features and labels (numpy arrays)
    - `sorted_X.npy` - Sorted feature matrix
    - `sorted_y.npy` - Sorted labels
  - `lfp_analysis/` - LFP analysis artifacts
    - `pre_stim_data/` - Pre-stimulus LFP data for individual sessions
- `plots/` - Generated visualization outputs
  - `changepoint_gapes/` - Gape-related plots
    - `binned_temporal_evolution_features.png` - Binned temporal feature evolution
    - `binned_temporal_evolution_features_cutoff_200ms.png` - Binned temporal evolution with 200ms cutoff
    - `binned_temporal_evolution_individual_features_cutoff_200ms.png` - Individual feature evolution with cutoff
    - `pca_explained_variance_smoothed_binned_features.png` - PCA variance explained
    - `temporal_evolution_features.png` - Temporal feature evolution
    - `temporal_evolution_individual_features_cutoff_200ms.png` - Individual feature temporal evolution
  - `lfp_analysis/` - LFP analysis plots
    - `median_diff_z_power_pre_post_changepoint.png` - Median power difference pre/post changepoint
    - `pre_post_changepoint_spectrograms.png` - Spectrograms pre/post changepoint
    - `taste_averaged_band_power_pre_post_changepoint.png` - Taste-averaged band power
    - `pre_stim_spectrograms/` - Individual session pre-stimulus spectrograms
  - `multibehavior_transition/` - Behavior transition plots
    - `two_test_changepoint_plots/` - Individual session changepoint visualizations

## Order of Operations

Initial set-up:
1. Run each session recording through Pytau, Blech_EMG_Classifier, and BlechClust packages before running this package.
2. Run create_tau_dict.py # combines tau, spike trains, and num cps from cp model into dictionary
3. Run combine_classifier_files.py # puts all Blech EMG Classifier segments files into one dataframe
4. Run initialize_dataframe.py # Adds important metrics to dataframe

Behavior-only analyses:
1. Run extract_emg_from_transition.py # calculates frequency of each behavior across trials, plots, and saves artifacts for each session
2. Run extract_emg_from_transition_aggregate.py # plots frequency of each behavior across trials, averages across all sessions
