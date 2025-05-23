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
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

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
    num_iterations = 5 if dim <= 10 else 10
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
                    acq_func = CustomEHVI(
                        model=gp,
                        ref_point=f.ref_point.tolist(),
                        partitioning=None,  # Let BoTorch handle partitioning
                        sampler=sampler,
                        beta=BETA,
                    )
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
                # Record hypervolume
                bd = DominatedPartitioning(ref_point=f.ref_point, Y=train_Y)
                volume = bd.compute_hypervolume().item()
                best_values.append(volume)
                # print(f"Iteration {i+1}: Best Y = {train_Y.min().item():.4f}")
            best_values_runs.append(best_values)
        return np.array(best_values_runs)
    
        
    best_value_ehvi = bayesian_optimization("EHVI")
    best_values_qehvi = bayesian_optimization("qEHVI")
    best_values_qLogehvi = bayesian_optimization("qLogEHVI")
    for beta in [1,5,10,15, 20, 25, 30, 35, 40]:
        BETA = beta
        
        best_values_ehvi_custom = bayesian_optimization("EHVI-Custom")
        
        # Compute mean and standard deviation
        mean_ehvi = best_value_ehvi.mean(axis=0)
        std_ehvi = best_value_ehvi.std(axis=0)
        mean_qehvi = best_values_qehvi.mean(axis=0)
        std_qehvi = best_values_qehvi.std(axis=0)
        mean_qLogehvi = best_values_qLogehvi.mean(axis=0)
        std_qLogehvi = best_values_qLogehvi.std(axis=0)
        mean_ehvi_custom = best_values_ehvi_custom.mean(axis=0)
        std_ehvi_custom = best_values_ehvi_custom.std(axis=0)
        # Plot results
        iterations = np.arange(len(mean_ehvi))
        plt.figure(figsize=(8, 5))
        
        plt.plot(iterations, mean_ehvi, marker="o", linestyle="-", color="b", label="EHVI")
        plt.fill_between(
            iterations, 
            mean_ehvi - std_ehvi, 
            mean_ehvi + std_ehvi, 
            color="b", alpha=0.2
        )
        
        plt.plot(iterations, mean_qehvi, marker="x", linestyle="-", color="g", label="qEHVI")  # Fixed variable
        plt.fill_between(
            iterations, 
            mean_qehvi - std_qehvi, 
            mean_qehvi + std_qehvi, 
            color="g", alpha=0.2
        )  # Fixed color
        
        plt.plot(
            iterations, mean_qLogehvi, marker="^", linestyle="-", color="m", label="qLogEHVI"
        )
        plt.fill_between(
            iterations, 
            mean_qLogehvi - std_qLogehvi, 
            mean_qLogehvi + std_qLogehvi, 
            color="m", alpha=0.2
        )
        
        plt.plot(
            iterations, 
            mean_ehvi_custom, 
            marker="s", 
            linestyle="-", 
            color="r", 
            label=f"EHVI Custom, Beta={BETA}"
        )
        plt.fill_between(
            iterations, 
            mean_ehvi_custom - std_ehvi_custom, 
            mean_ehvi_custom + std_ehvi_custom, 
            color="r", alpha=0.2
        )
        
        plt.xlabel("Iteration")
        plt.ylabel("Best HV")
        # Extract function name dynamically
        func_name = objective_func.__class__.__name__
        plt.title(f"BO with EHVI vs. EHVI-Custom on {func_name}-{dim}D-{f.num_objectives} objectives")
        plt.legend()
        plt.grid(True)
        if os.path.isdir(f"results/{func_name}") == False:
            os.makedirs(f"results/{func_name}")
        plt.savefig(f"results/{func_name}/{func_name}-{dim}-{f.num_objectives}-beta={BETA}.pdf", dpi=300)  # Save plot as PDF
