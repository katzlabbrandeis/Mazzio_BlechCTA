"""
Fit a random walk model to the Gapes temporal data using PyMC
- Bin data into intervals
- Use a random walk to model latents
- Use projections from latents to model full-dimensional mean and variance (assume spherical covariance)
- Use t-distribution likelihood to model data for robustness to outliers

REF: https://www.pymc.io/projects/examples/en/latest/time_series/MvGaussianRandomWalk_demo.html
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
import pymc as pm

# from sklearn.neural_network import MLPRegressor
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.signal import savgol_filter

base_dir = '/home/abuzarmahmood/projects/CMazzio_analysis'
data_dir = os.path.join(base_dir, 'data', 'changepoint_gapes')
plot_dir = os.path.join(base_dir, 'plots', 'changepoint_gapes')
artifacts_dir = os.path.join(base_dir, 'artifacts', 'changepoint_gapes')

# # Write out as artifacts
# np.save(os.path.join(artifacts_dir, 'sorted_y.npy'), sorted_y)
# np.save(os.path.join(artifacts_dir, 'sorted_X.npy'), sorted_X)

# Load artifacts
sorted_y = np.load(os.path.join(artifacts_dir, 'sorted_y.npy'))
sorted_X = np.load(os.path.join(artifacts_dir, 'sorted_X.npy'))

# Bin data into intervals
bin_size = 200
bin_edges = np.arange(sorted_y.min(), sorted_y.max() + bin_size, bin_size)
bin_indices = np.digitize(sorted_y, bin_edges) - 1  # Get bin indices for each data point

##############################

# def inference(t, y, sections, n_samples=100):
#     N, D = y.shape
#
#     # Standardize y and t
#     y_scaler = Scaler()
#     t_scaler = Scaler()
#     y = y_scaler.fit_transform(y)
#     t = t_scaler.fit_transform(t)
#     # Create a section index
#     t_section = np.repeat(np.arange(sections), N / sections)
#
#     # Create PyTensor equivalent
#     t_t = pytensor.shared(np.repeat(t, D, axis=1))
#     y_t = pytensor.shared(y)
#     t_section_t = pytensor.shared(t_section)
#
#     coords = {"y_": ["y_0", "y_1", "y_2"], "steps": np.arange(N)}
#     with pm.Model(coords=coords) as model:
#         # Hyperpriors on Cholesky matrices
#         chol_alpha, *_ = pm.LKJCholeskyCov(
#             "chol_cov_alpha", n=D, eta=2, sd_dist=pm.HalfCauchy.dist(2.5), compute_corr=True
#         )
#         chol_beta, *_ = pm.LKJCholeskyCov(
#             "chol_cov_beta", n=D, eta=2, sd_dist=pm.HalfCauchy.dist(2.5), compute_corr=True
#         )
#
#         # Priors on Gaussian random walks
#         alpha = pm.MvGaussianRandomWalk(
#             "alpha", mu=np.zeros(D), chol=chol_alpha, shape=(sections, D)
#         )
#         beta = pm.MvGaussianRandomWalk("beta", mu=np.zeros(D), chol=chol_beta, shape=(sections, D))
#
#         # Deterministic construction of the correlated random walk
#         alpha_r = alpha[t_section_t]
#         beta_r = beta[t_section_t]
#         regression = alpha_r + beta_r * t_t
#
#         # Prior on noise ξ
#         sigma = pm.HalfNormal("sigma", 1.0)
#
#         # Likelihood
#         likelihood = pm.Normal("y", mu=regression, sigma=sigma, observed=y_t, dims=("steps", "y_"))
#
#         # MCMC sampling
#         trace = pm.sample(n_samples, tune=1000, chains=4, target_accept=0.9)
#
#         # Posterior predictive sampling
#         pm.sample_posterior_predictive(trace, extend_inferencedata=True)
#
#     return trace, y_scaler, t_scaler, t_section

# Define model
n_bins = len(bin_edges) - 1
n_features = sorted_X.shape[1]
n_latents = 3  # Number of latent dimensions for the random walk

with pm.Model() as model:
    # Random walk latents
    latents = pm.MvGaussianRandomWalk("latents", mu=np.zeros(n_latents), shape=(n_bins, n_latents))
