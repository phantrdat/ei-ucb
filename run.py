import torch
import os
from botorch.test_functions import *
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement 

from test_functions import *
from custom_ei import CustomExpectedImprovement

from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
from botorch.models.transforms.outcome import Standardize
from torch.quasirandom import SobolEngine
# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# Define list function tests
OBJECTIVE_FUNCTIONS = [
    Ackley,
    # Beale,
    # Branin,
    # Bukin,
    # DixonPrice,
    # DropWave,
    # EggHolder,
    # Griewank,
    # Alpine1,
    # Hartmann,
    # HolderTable,
    # Levy,
    # Michalewicz,
    # Powell,
    # Rastrigin,
    # Rosenbrock,
    # Shekel,
    # SixHumpCamel,
    # StyblinskiTang
    ]

DIMS = {
    Ackley: 50,
    Alpine1: 15,
    DixonPrice: 15,
    Griewank:9,
    Levy: 13,
    Powell: 18,
    Rastrigin: 23,
    Rosenbrock: 24,
    StyblinskiTang: 21
    }

N = 10  # Number of runs per acquisition function
BETA=20
for f in OBJECTIVE_FUNCTIONS:
    dim = 0
    if hasattr(f, "dim"):
        dim = f.dim
        objective_func = f().to(dtype=dtype, device=device)
    elif f.__name__ == 'Hartmann': 
        dim = 6
        objective_func = f().to(dtype=dtype, device=device)
    else:
        # dim = np.random.randint(low=7, high=25)         
        dim = DIMS[f]
        objective_func = f(dim=dim).to(dtype=dtype, device=device)
    
    bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)  # Search space bounds
    # Experiment settings
    
    num_initial_points = dim + 1
    num_iterations = 100 if dim <= 10 else 200
    # Generate initial training data
    sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
    fixed_train_X  = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_initial_points).to(dtype=dtype, device=device)
    fixed_train_Y  = objective_func(fixed_train_X).unsqueeze(-1) # Evaluate function and reshape
    def bayesian_optimization(acq_type):
        best_values_runs = []
        for run in range(N):
            print(objective_func.__class__.__name__, acq_type, run)
            train_X = fixed_train_X.clone()
            train_Y = fixed_train_Y.clone()
            # Track best observed function values
            best_values = [train_Y.min().item()]
            gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            for i in range(num_iterations):
                if acq_type == "EI":
                    acq_func = ExpectedImprovement(model=gp, best_f=train_Y.min(), maximize=False)
                elif acq_type == "LogEI":
                    acq_func = LogExpectedImprovement(model=gp, best_f=train_Y.min(), maximize=False)
                elif acq_type == "EI-Custom":
                    # beta = 10  # Exploration parameter for EI Custom
                    acq_func = CustomExpectedImprovement(model=gp, best_f=train_Y.min(), beta=BETA, maximize=False)
                else:
                    raise ValueError("Invalid acquisition function type")
                # Optimize the acquisition function to find the next query point
                candidate, _ = optimize_acqf(
                    acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=100
                )
                # Evaluate the function at the new point
                new_Y = objective_func(candidate).unsqueeze(-1)
                # Update the dataset
                train_X = torch.cat([train_X, candidate])
                train_Y = torch.cat([train_Y, new_Y])
                # Store best observed value
                best_values.append(train_Y.min().item())
                # print(f"Iteration {i+1}: Best Y = {train_Y.min().item():.4f}")
            best_values_runs.append(best_values)
        return np.array(best_values_runs)
    
        
    
    best_values_ei = bayesian_optimization("EI")
    best_values_logei = bayesian_optimization("LogEI")
    for beta in [1,5,10,15, 20, 25, 30, 35, 40]:
        BETA = beta
        
        best_values_ei_custom = bayesian_optimization("EI-Custom")
        
        # Compute mean and standard deviation
        mean_ei = best_values_ei.mean(axis=0)
        std_ei = best_values_ei.std(axis=0)
        mean_logei = best_values_logei.mean(axis=0)
        std_logei = best_values_logei.std(axis=0)
        mean_ei_custom = best_values_ei_custom.mean(axis=0)
        std_ei_custom = best_values_ei_custom.std(axis=0)
        # Plot results
        iterations = np.arange(len(mean_ei))
        plt.figure(figsize=(8, 5))
        
        plt.plot(iterations, mean_ei, marker="o", linestyle="-", color="b", label="EI")
        plt.fill_between(iterations, mean_ei - std_ei, mean_ei + std_ei, color="b", alpha=0.2)
        
        plt.plot(iterations, mean_logei, marker="x", linestyle="-", color="g", label="LogEI")  # Fixed variable
        plt.fill_between(iterations, mean_logei - std_logei, mean_logei + std_logei, color="g", alpha=0.2)  # Fixed color
        
        
        plt.plot(iterations, mean_ei_custom, marker="s", linestyle="-", color="r", label=f"EI Custom, Beta={BETA}")
        plt.fill_between(iterations, mean_ei_custom - std_ei_custom, mean_ei_custom + std_ei_custom, color="r", alpha=0.2)
        
        plt.xlabel("Iteration")
        plt.ylabel("Best Function Value Found")
        # Extract function name dynamically
        func_name = objective_func.__class__.__name__
        plt.title(f"BO with EI vs. EI-Custom on {func_name}-{dim}D")
        plt.legend()
        plt.grid(True)
        if os.path.isdir(f"results/{func_name}") == False:
            os.makedirs(f"results/{func_name}")
        plt.savefig(f"results/{func_name}/{func_name}-{dim}-beta={BETA}.pdf", dpi=300)  # Save plot as PDF
