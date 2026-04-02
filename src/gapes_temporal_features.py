import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter

base_dir = '/home/abuzarmahmood/projects/CMazzio_analysis'
# data_dir = '/home/abuzarmahmood/projects/CMazzio_analysis/data/changepoint_gapes'
data_dir = os.path.join(base_dir, 'data', 'changepoint_gapes')
plot_dir = os.path.join(base_dir, 'plots', 'changepoint_gapes')
os.makedirs(plot_dir, exist_ok=True)


file_list = os.listdir(data_dir) 
# Load pkls
dfs = []
for file in file_list:
    if file.endswith('.pkl'):
        df = pd.read_pickle(os.path.join(data_dir, file))
        dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

y = final_df['start_time'].values
X = final_df[[x for x in final_df.columns if 'feat' in x]].values

# Check that features are standardized
# X.mean(axis=0)
# X.std(axis=0)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# recheck
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

# Sort by y and plot
sort_inds = np.argsort(y)
sorted_y = y[sort_inds]
sorted_X = X_scaled[sort_inds]

fig, ax = plt.subplots(figsize=(4, 6))
ax.pcolormesh(
    np.arange(sorted_X.shape[1]),
    sorted_y,
    sorted_X,
    shading='auto',
    cmap='jet'
    )
ax.set_xlabel('Feature Index')
ax.set_ylabel('Start Time')
ax.set_title('Temporal Evolution of Features')
plt.colorbar(label='Feature Value (Standardized)')
# plt.show()
fig.savefig(os.path.join(plot_dir, 'temporal_evolution_features.png'), dpi=300)
plt.close()

# Make lineplots
fig, ax = plt.subplots(X.shape[1], 1, figsize=(10, 20), sharex=True)
for i in range(X.shape[1]):
    ax[i].plot(sorted_y, sorted_X[:, i], '.')
    ax[i].set_ylabel(f'Feature {i}')
ax[-1].set_xlabel('Start Time')
plt.suptitle('Temporal Evolution of Individual Features')
fig.savefig(os.path.join(plot_dir, 'temporal_evolution_individual_features.png'), dpi=300)
plt.close()
# plt.show()

# Take average features by binning y
bin_size = 50
bins = np.arange(y.min(), y.max() + bin_size, bin_size)

binned_X = []
wanted_inds = np.digitize(y, bins) - 1
for i in range(len(bins) - 1):
    inds = np.where(wanted_inds == i)[0]
    if len(inds) > 0:
        binned_X.append(X_scaled[inds].mean(axis=0))

binned_X = np.array(binned_X)

fig, ax = plt.subplots(figsize=(4, 6))
im = ax.pcolormesh(
    np.arange(binned_X.shape[1]),
    bins[:-1],
    binned_X,
    shading='auto',
    cmap='jet'
    )
ax.set_xlabel('Feature Index')
ax.set_ylabel('Binned Start Time')
ax.set_title('Binned Temporal Evolution of Features')
plt.colorbar(im, label='Average Feature Value (Standardized)')
# plt.show()
fig.savefig(os.path.join(plot_dir, 'binned_temporal_evolution_features.png'), dpi=300)
plt.close()

# Make lineplots of binned features
fig, ax = plt.subplots(X.shape[1], 1, figsize=(5, 10), sharex=True)
for i in range(X.shape[1]):
    ax[i].plot(bins[:-1], binned_X[:, i], '.-', label='Actual')
    # Also plot smoothed version of the line
    smoothed = savgol_filter(binned_X[:, i], 5, 2)
    ax[i].plot(bins[:-1], smoothed, 'r-', alpha=0.7, label='Smoothed')
    ax[i].set_ylabel(f'Feature {i}')
ax[-1].set_xlabel('Binned Start Time')
plt.suptitle('Binned Temporal Evolution of Individual Features')
# Put legend at bottom of figure
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -1), ncol=2)
plt.tight_layout()
# plt.show()
fig.savefig(os.path.join(plot_dir, 'binned_temporal_evolution_individual_features.png'), dpi=300)
plt.close()

##############################
# Perform PCA to check correlations
pca = PCA()
pca.fit(X_scaled)
explained_variance = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(5, 5))
# Start plot at 0
ax.plot(
        np.arange(0, len(explained_variance)+1),
        np.concatenate(([0], np.cumsum(explained_variance))),
        marker='o'
        )
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance')
fig.suptitle('PCA Explained Variance')
plt.grid()
# plt.show()
fig.savefig(os.path.join(plot_dir, 'pca_explained_variance.png'), dpi=300)
plt.close()

##############################
# Use a 2 layered network to predict y from X
# The second layer will allow us to extract a relevant feature space that is predictive of y
# Check cross-validation performance to ensure we're not overfitting

mlp = MLPRegressor(hidden_layer_sizes=(3), max_iter=10000, random_state=42, activation='relu', verbose=True)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(mlp, X_scaled, y, cv=5, scoring='r2')

# Extract the relevant feature space from the second layer
mlp.fit(X_scaled, y)
# Get the weights of the second layer
weights = mlp.coefs_[0]
intercepts = mlp.intercepts_[0]
# Project the original features onto the weights to get the new feature space
new_features = X_scaled @ weights + intercepts
# Rectify the new features (since we used ReLU)
new_features = np.maximum(0, new_features)

# Plot the new features in 3D space and color by y
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(new_features[:, 0], new_features[:, 1], new_features[:, 2], c=y, cmap='viridis')
ax.set_xlabel('New Feature 1')
ax.set_ylabel('New Feature 2')
ax.set_zlabel('New Feature 3')
ax.set_title('New Feature Space from MLP')
plt.colorbar(label='Start Time')
plt.show()
