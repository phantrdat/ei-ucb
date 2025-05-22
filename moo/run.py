import torch
import os
from botorch.test_functions.multi_objective import *
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import (
    qExpectedHypervolumeImprovement, 
    ExpectedHypervolumeImprovement
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement
)
from botorch.sampling.normal import SobolQMCNormalSampler

from custom_ehvi import CustomEHVI

from botorch.optim import optimize_acqf
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
    BraninCurrin(),
    DH1(dim=5),
    DH2(dim=5),
    DH3(dim=5),
    DH4(dim=5),
    DTLZ1(dim=6, n_objectives=2),
    DTLZ2(dim=6, n_objectives=2),
    DTLZ3(dim=6, n_objectives=2),
    DTLZ4(dim=6, n_objectives=2),
    DTLZ5(dim=6, n_objectives=2),
    DTLZ7(dim=6, n_objectives=2),
    Penicillin(),
    VehicleSafety(),
    ZDT1(dim=5),
    ZDT2(dim=5),
    ZDT3(dim=5),
    CarSideImpact(),
    ]

N = 10  # Number of runs per acquisition function
BETA=20
MC_SAMPLES = 128  # Number of Monte Carlo samples for acquisition function
for f in OBJECTIVE_FUNCTIONS:
    dim = f.dim
    objective_func = f().to(dtype=dtype, device=device)
    
    bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)  # Search space bounds
    # Experiment settings
    
    num_initial_points = dim + 1
    num_iterations = 100 if dim <= 10 else 200
    # Generate initial training data
    sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
    fixed_train_X  = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_initial_points).to(dtype=dtype, device=device)
    fixed_train_Y  = objective_func(fixed_train_X) # Evaluate function
    def bayesian_optimization(acq_type):
        best_values_runs = []
        for run in range(N):
            print(objective_func.__class__.__name__, acq_type, run)
            train_X = fixed_train_X.clone()
            train_Y = fixed_train_Y.clone()
            # Track best observed function values
            best_values = [train_Y.min().item()]
            gp = []
            for _ in range(train_Y.shape[-1]):
                gp.append(
                    SingleTaskGP(
                        train_X, 
                        train_Y, 
                        outcome_transform=Standardize(m=1)
                    )
                )
            gp = ModelListGP(*gp)
            mll = SumMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            for _ in range(num_iterations):
                # Multi-objective acquisition function using hypervolume improvement
                sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
                if acq_type == "EHVI":
                    acq_func = ExpectedHypervolumeImprovement(
                        model=gp,
                        ref_point=f.ref_point.tolist(),
                        partitioning=None,  # Let BoTorch handle partitioning
                        sampler=sampler,
                    )
                elif acq_type == "qEHVI":
                    sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
                    acq_func = qExpectedHypervolumeImprovement(
                        model=gp,
                        ref_point=f.ref_point.tolist(),
                        partitioning=None,  # Let BoTorch handle partitioning
                        sampler=sampler,
                    )
                elif acq_type == "qLogEHVI":
                    acq_func = qLogExpectedHypervolumeImprovement(
                        model=gp,
                        ref_point=f.ref_point.tolist(),
                        partitioning=None,  # Let BoTorch handle partitioning
                        sampler=sampler,
                    )
                elif acq_type == "EHVI-Custom":
                    # beta = 10  # Exploration parameter for EI Custom
                    acq_func = CustomEHVI(model=gp, best_f=train_Y.min(), beta=BETA, maximize=False)
                else:
                    raise ValueError("Invalid acquisition function type")
                # Optimize the acquisition function to find the next query point
                candidate, _ = optimize_acqf(
                    acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=100
                )
                # Evaluate the function at the new point
                new_Y = objective_func(candidate)
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
