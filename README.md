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
    - `gapes_temporal_pymc.py` - PyMC modeling for gape analysis
  - `multibehavior_transition/` - Multi-behavior transition analysis
    - `multibehavior_transition.py` - Main transition analysis script
- `data/` - Processed data files
  - `changepoint_gapes/` - Changepoint detection data for gapes
  - `multibehavior_transition/` - Behavior transition data
- `artifacts/` - Generated analysis artifacts
  - `changepoint_gapes/` - Sorted features and labels (numpy arrays)
- `plots/` - Generated visualization outputs
  - `changepoint_gapes/` - Gape-related plots (PCA, temporal evolution, feature space)
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
