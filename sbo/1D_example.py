import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.test_functions import *
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

from custom_ei import CustomExpectedImprovement

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('type', type=str, choices=['EI', 'Custom_EI'])
args = parser.parse_args()

type = args.type
def ackley(x):
    """1D Ackley function."""
    a, b, c = 20, 0.2, 2 * np.pi
    return -a * np.exp(-b * np.sqrt(0.5 * (x**2))) - np.exp(0.5 * np.cos(c * x)) + a + np.exp(1)

def levi(x):
    """1D Levi function N.13."""
    return torch.sin(3 * np.pi * x) ** 2 + (x - 1) ** 2 * (1 + torch.sin(3 * np.pi * x) ** 2)

objective_func = ackley
# Define search space
bounds = torch.tensor([[-5.0], [5.0]])  # 1D space

# Generate initial training data
train_X = torch.tensor([[-4.0], [3.87], [1.12]])
train_Y = objective_func(train_X)
# Optimization loop
num_iters = 100
for i in tqdm(range(num_iters)):
    # Fit GP model
    gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # Define EI acquisition function
    if type =='Custom_EI':
        acqf = CustomExpectedImprovement(gp, best_f=train_Y.min().item(),  maximize=False, beta=1)
    else:
        acqf = ExpectedImprovement(gp, best_f=train_Y.min().item(),  maximize=False)
        
    # Optimize EI using grid search
    grid = torch.linspace(-5, 5, 100).unsqueeze(-1).unsqueeze(1)
    ei_values = acqf(grid)
    next_x = grid[torch.argmax(ei_values)]
    
    # Evaluate new point
    next_y = objective_func(next_x)
    
    # Update training data
    train_X = torch.cat([train_X, next_x], dim=0)
    train_Y = torch.cat([train_Y, next_y], dim=0)
    
    # Plot results
    with torch.no_grad():
        test_X = torch.linspace(-5, 5, 100).unsqueeze(-1)
        posterior = gp.posterior(test_X.unsqueeze(1))
        mean = posterior.mean.squeeze()
        std = posterior.variance.sqrt().squeeze()
        
        fig, ax = plt.subplots(2, figsize=(6, 8))
        # plt.figure(figsize=(8, 5))
        ax[0].plot(test_X.numpy().flatten(), objective_func(test_X.unsqueeze(1)).flatten().numpy(), 'k--', label='True Function')
        ax[0].plot(test_X.numpy().flatten(), mean.numpy(), 'b-', label='GP Mean')
        ax[0].fill_between(test_X.numpy().flatten(),
                         (mean - 2*std).numpy(), (mean + 2*std).numpy(),
                         alpha=0.2, color='blue', label='Confidence Interval')
        ax[0].scatter(train_X.numpy(), train_Y.numpy(), color='red', label='Observations')
        ax[0].scatter(next_x.numpy(), next_y.numpy(), color='green', marker='x', s=100, label='Next Point')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('Function Value')
        ax[0].legend(loc='upper left', bbox_to_anchor=(0., -0.15), ncol=5, fontsize=6.25)
        ax[0].set_title(f'Iteration {i+1} - {type} - {objective_func.__name__}')
        
        # fig, ax2 = plt.subplots(figsize=(8, 3))
        ax[1].plot(grid.flatten(), ei_values.numpy(), 'm-', label='EI')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('EI Values')
        ax[1].set_title('Acquisition Function (EI)')
        ax[1].legend()
        plt.tight_layout()
        
        if os.path.isdir(f'examples/{objective_func.__name__}/{type}/') == False:
            os.makedirs(f'examples/{objective_func.__name__}/{type}/')
        plt.savefig(f'examples/{objective_func.__name__}/{type}/{i+1}_{type}.png', dpi=300)


# Type A plot
        # plt.figure(figsize=(8, 5))
        # plt.plot(test_X.numpy().flatten(), objective_func(test_X.unsqueeze(1)).flatten().numpy(), 'k--', label='True Function')
        # plt.plot(test_X.numpy().flatten(), mean.numpy(), 'b-', label='GP Mean')
        # plt.fill_between(test_X.numpy().flatten(),
        #                  (mean - 2*std).numpy(), (mean + 2*std).numpy(),
        #                  alpha=0.2, color='blue', label='Confidence Interval')
        # plt.scatter(train_X.numpy(), train_Y.numpy(), color='red', label='Observations')
        # plt.scatter(next_x.numpy(), next_y.numpy(), color='green', marker='x', s=100, label='Next Point')
        # plt.plot(grid.flatten(), ei_values.numpy(), 'm-', label='EI')
        # plt.legend()
        # plt.title(f'Iteration {i+1} - {type} - {objective_func.__name__}')

# Type B plot 

# fig, ax = plt.subplots(2, figsize=(6, 8))
#         # plt.figure(figsize=(8, 5))
#         ax[0].plot(test_X.numpy().flatten(), objective_func(test_X.unsqueeze(1)).flatten().numpy(), 'k--', label='True Function')
#         ax[0].plot(test_X.numpy().flatten(), mean.numpy(), 'b-', label='GP Mean')
#         ax[0].fill_between(test_X.numpy().flatten(),
#                          (mean - 2*std).numpy(), (mean + 2*std).numpy(),
#                          alpha=0.2, color='blue', label='Confidence Interval')
#         ax[0].scatter(train_X.numpy(), train_Y.numpy(), color='red', label='Observations')
#         ax[0].scatter(next_x.numpy(), next_y.numpy(), color='green', marker='x', s=100, label='Next Point')
#         ax[0].set_xlabel('x')
#         ax[0].set_ylabel('Function Value')
#         ax[0].legend(loc='upper left', bbox_to_anchor=(0., -0.15), ncol=5, fontsize=6.25)
#         ax[0].set_title(f'Iteration {i+1} - {type} - {objective_func.__name__}')
        
#         # fig, ax2 = plt.subplots(figsize=(8, 3))
#         ax[1].plot(grid.flatten(), ei_values.numpy(), 'm-', label='EI')
#         ax[1].set_xlabel('x')
#         ax[1].set_ylabel('EI Values')
#         ax[1].set_title('Acquisition Function (EI)')
#         ax[1].legend()
#         plt.tight_layout()