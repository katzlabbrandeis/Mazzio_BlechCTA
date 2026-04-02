import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_dir = '/home/abuzarmahmood/projects/CMazzio_analysis/data/changepoint_gapes'
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

fig, ax = plt.subplots(figsize=(10, 6))
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
plt.show()

# Make lineplots
fig, ax = plt.subplots(X.shape[1], 1, figsize=(10, 20), sharex=True)
for i in range(X.shape[1]):
    ax[i].plot(sorted_y, sorted_X[:, i], '.')
    ax[i].set_ylabel(f'Feature {i}')
ax[-1].set_xlabel('Start Time')
plt.suptitle('Temporal Evolution of Individual Features')
plt.show()

# Take average features by binning y
bin_size = 100
bins = np.arange(y.min(), y.max() + bin_size, bin_size)

binned_X = []
wanted_inds = np.digitize(y, bins) - 1
for i in range(len(bins) - 1):
    inds = np.where(wanted_inds == i)[0]
    if len(inds) > 0:
        binned_X.append(X_scaled[inds].mean(axis=0))

binned_X = np.array(binned_X)

fig, ax = plt.subplots(figsize=(10, 6))
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
plt.show()
