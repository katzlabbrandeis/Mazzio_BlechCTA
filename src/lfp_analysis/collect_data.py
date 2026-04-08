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
