#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 11:23:42 2026

@author: cmazzio
"""

# Uses behavior frequency artifacts saved from extract_emg_from_transition.py 
    # Plots average behavior frequency across trials, across all animals per
        # taste, session type

#%% IMPORTS
import sys
import pickle
import easygui
import os
import csv
from matplotlib import cm
import pylab as plt
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import json
# Get dataframe for all data
import pandas as pd
data_path = '/media/cmazzio/large_data'


#%% LOAD aggregate data for behavior frequency plot for classifier 

artifact_save_path = "/media/cmazzio/large_data/behavior_frequency_classifier_artifacts"

all_animal_artifacts = []  # This will hold all loaded pickle objects

for filename in os.listdir(artifact_save_path):
    file_path = os.path.join(artifact_save_path, filename)
    
    if os.path.isfile(file_path) and filename.endswith(".pkl"):
        try:
            with open(file_path, "rb") as f:
                fin_output = pickle.load(f)
                all_animal_artifacts.append(fin_output)
                print(f"Successfully loaded: {filename}")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")


#%% Standardize all taste names

all_tastes = []

for animal_dict in all_animal_artifacts:
    this_animal_tastes = list(animal_dict["tastes"].keys())
    all_tastes.append(this_animal_tastes)

flat_list = [taste for sublist in all_tastes for taste in sublist]
all_unique_tastes = list(set(flat_list))
     
#for simplicity, QHCl in an older dataset will be categorized as highqhcl to match the newer protocol animals
#if any tastes are spelled differently or use upper case instead of lower case, this will fix that
taste_map= {
            'NaCl' : 'nacl',
             'QHCl' : 'highqhcl',
             'qhcl' : 'highqhcl',
             'highQHCl' : 'highqhcl',
             'highqhcl' : 'highqhcl',
             'lowQHCl': 'lowqhcl',
             'lowqhcl': 'lowqhcl',
             'nacl': 'nacl',
             'saccharin' : 'saccharin',
             'water' : 'water',
             }
all_unique_tastes_clean = [
    taste_map.get(taste, taste_map.get(taste.lower(), taste.lower()))
    for taste in all_unique_tastes
]

#this list is edited for all possible alternate spellings of a taste- it is the 
#final standardization
final_taste_list = list(set(all_unique_tastes_clean))


for animal_dict in all_animal_artifacts:
    this_animal = animal_dict["animal"]
    for old_taste in list(animal_dict["tastes"].keys()):
        
        # Normalize using taste_map (case-robust)
        new_taste = taste_map.get(
            old_taste,
            taste_map.get(old_taste.lower(), old_taste.lower())
        )

        # If the taste is already correct, skip
        if new_taste == old_taste:
            continue

        # Rename key while keeping its content
        animal_dict["tastes"][new_taste] = animal_dict["tastes"].pop(old_taste)
        print(f"{this_animal}: '{old_taste}' ➜ '{new_taste}'")
        



#%%A Average Frequency of Each Behavior Per Taste, Per Session

save_dir = "/media/cmazzio/large_data/behavior_frequency_classifier_artifacts/aggregate_plots"
os.makedirs(save_dir, exist_ok=True)

def get_day_key(exp_day_type, num_of_cta):
    if exp_day_type == "Train" and num_of_cta == 0:
        return "Train Day 1"
    if exp_day_type == "Train" and num_of_cta == 1:
        return "Train Day 2"
    if exp_day_type == "Test" and num_of_cta in (1, 2):
        return "Test Day 1"
    if exp_day_type == "Test" and num_of_cta == 4:
        return "Test Day 2"
    return None

day_order = ["Train Day 1", "Train Day 2", "Test Day 1", "Test Day 2"]

behaviors = [0, 1, 2]
behavior_labels = {0: "No Movement", 1: "Gape", 2: "MTM"}
behavior_colors = {0: '#D3D3D3', 1: '#ff9900', 2: '#4285F4'}

for taste in final_taste_list:

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, day_key in zip(axes, day_order):

        traces_by_behavior = {b: [] for b in behaviors}
        x_plot = None

        for animal_dict in all_animal_artifacts:
            taste_data = animal_dict["tastes"].get(taste)
            if taste_data is None:
                continue

            for basename_data in taste_data["basenames"].values():

                exp_day_type = basename_data["exp_day_type"]
                num_of_cta = basename_data["num_of_cta"]

                if get_day_key(exp_day_type, num_of_cta) != day_key:
                    continue

                movement_type = basename_data["movement_type"]

                # handle extra blank dict level
                if isinstance(movement_type, dict) and len(movement_type) == 1:
                    movement_type = list(movement_type.values())[0]

                for behavior in behaviors:
                    if behavior not in movement_type:
                        continue

                    x = movement_type[behavior]["x"]
                    y = movement_type[behavior]["y"]

                    mask = x >= 2000
                    traces_by_behavior[behavior].append(y[mask])

                    if x_plot is None:
                        x_plot = x[mask]

        any_plotted = False
        for behavior in behaviors:
            if len(traces_by_behavior[behavior]) == 0:
                continue

            data_stack = np.vstack(traces_by_behavior[behavior])
            mean_trace = np.mean(data_stack, axis=0)
            sem_trace = np.std(data_stack, axis=0, ddof=1) / np.sqrt(data_stack.shape[0])

            ax.plot(
                x_plot, mean_trace,
                linewidth=2,
                label=behavior_labels[behavior],
                color=behavior_colors[behavior]
            )

            ax.fill_between(
                x_plot,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                alpha=0.3,
                color=behavior_colors[behavior]
            )

            any_plotted = True

        if not any_plotted:
            ax.set_title(f"{day_key}\n(no data)")
            ax.axis("off")
            continue

        ax.axvline(2000, linestyle="--", color="gray", alpha=0.6)
        ax.set_title(day_key)
        ax.legend()

    fig.suptitle(f"{taste} – Mean Frequency + SEM\n(All Animals, by Behavior)")
    fig.supxlabel("Time (ms)")
    fig.supylabel("Frequency (%)")

    plt.tight_layout()
    
    # ---- SAVE FIGURE ----
    save_path = os.path.join(save_dir, f"{taste}_mean_freq_sem.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    
    plt.show()
