import torch
import os
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
from torch.quasirandom import SobolEngine
# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models
from tqdm import tqdm
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import gpytorch
MAX_CHOLESKY_SIZE = float('inf')

from custom_ehvi import CustomEHVI
from constants import *

for f in OBJECTIVE_FUNCTIONS:
    objective_func = f(**FUNCTION_ARGS[f]).to(dtype=dtype, device=device)
    dim = objective_func.dim
    
    bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)  # Search space bounds
    # Experiment settings
    
    num_initial_points = 2*dim + 1
    num_iterations = 5 if dim <= 10 else 10
    # Generate initial training data
    sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
    fixed_train_X  = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_initial_points).to(dtype=dtype, device=device)
    fixed_train_Y  = objective_func(fixed_train_X) # Evaluate function
    def bayesian_optimization(acq_type):
        best_values_runs = []
        ref_point = objective_func.ref_point
        for run in range(N):
            print(f"Run {run+1}")
            print(objective_func.__class__.__name__, acq_type, run+1)
            train_X = fixed_train_X.clone()
            train_Y = fixed_train_Y.clone()
            # Track best observed function values
            best_values = [train_Y.min().item()]
            gp = []
            for obj_idx in range(train_Y.shape[-1]):
                train_y = train_Y[..., obj_idx].unsqueeze(-1)
                train_yvar = torch.full_like(train_y, (1.01**train_y.shape[0]) * (NOISE_SE**2))
                gp.append(
                    SingleTaskGP(
                        train_X, 
                        train_y, 
                        train_yvar,
                        outcome_transform=Standardize(m=1)
                    )
                )
            gp = ModelListGP(*gp)
            mll = SumMarginalLogLikelihood(gp.likelihood, gp)
            with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
                fit_gpytorch_mll(mll)
            gp.eval()
            for _ in tqdm(range(num_iterations)):
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
                with torch.no_grad():
                    pred = gp.posterior(train_X).mean
                partitioning = FastNondominatedPartitioning(
                    ref_point=objective_func.ref_point,
                    Y=pred,
                )
                if acq_type == "EHVI":
                    acq_func = ExpectedHypervolumeImprovement(
                        model=gp,
                        ref_point=ref_point.tolist(),
                        partitioning=partitioning, 
                    )
                elif acq_type == "qEHVI":
                    acq_func = qNoisyExpectedHypervolumeImprovement(
                        model=gp,
                        ref_point=ref_point.tolist(),
                        X_baseline=train_X,
                        prune_baseline=True,
                        sampler=sampler,
                    )
                elif acq_type == "qLogEHVI":
                    acq_func = qLogNoisyExpectedHypervolumeImprovement(
                        model=gp,
                        ref_point=ref_point.tolist(),
                        X_baseline=train_X,
                        prune_baseline=True,
                        sampler=sampler,
                    )
                elif acq_type == "EHVI-Custom":
                    acq_func = CustomEHVI(
                        model=gp,
                        ref_point=ref_point.tolist(),
                        X_baseline=train_X,
                        prune_baseline=True, 
                        sampler=sampler,
                        beta=BETA,
                    )
                else:
                    raise ValueError("Invalid acquisition function type")
                # Optimize the acquisition function to find the next query point
                candidate, _ = optimize_acqf(
                    acq_func, 
                    bounds=bounds, 
                    q=1, 
                    num_restarts=20, 
                    raw_samples=512,
                    options={
                        "batch_limit": 5,
                        "maxiter": 100
                    },
                    sequential=True
                )
                # Evaluate the function at the new point
                new_Y = objective_func(candidate)
                # Update the dataset
                train_X = torch.cat([train_X, candidate])
                train_Y = torch.cat([train_Y, new_Y])
                # Record hypervolume
                bd = DominatedPartitioning(ref_point=ref_point, Y=train_Y)
                volume = bd.compute_hypervolume().item()
                best_values.append(volume)
                # print(f"Iteration {i+1}: Best Y = {train_Y.min().item():.4f}")
            best_values_runs.append(best_values)
        return np.array(best_values_runs)
    
    # Extract function name dynamically
    func_name = objective_func.__class__.__name__
    if not os.path.exists(f"{NUMERICAL_RESULTS_DIR}/{func_name}"):
        os.makedirs(f"{NUMERICAL_RESULTS_DIR}/{func_name}")
    # print(f"Running Bayesian Optimization for {func_name} with {dim} dimensions and {objective_func.num_objectives} objectives")
    # print("Running EHVI")
    # best_value_ehvi = bayesian_optimization("EHVI")
    # np.save(f"{NUMERICAL_RESULTS_DIR}/{func_name}/{func_name}_best_values_ehvi.npy", best_value_ehvi)
    # print("Running qEHVI")
    # best_values_qehvi = bayesian_optimization("qEHVI")
    # np.save(f"{NUMERICAL_RESULTS_DIR}/{func_name}/{func_name}_best_values_logehvi.npy", best_values_qehvi)
    # print("Running qLogEHVI")
    # best_values_qLogehvi = bayesian_optimization("qLogEHVI")
    # np.save(f"{NUMERICAL_RESULTS_DIR}/{func_name}/{func_name}_best_values_qlogehvi.npy", best_values_qLogehvi)

    for beta in [1, 5, 10, 15, 20, 25, 30, 35, 40]:
        BETA = beta
        print(f"Running EHVI-Custom with beta={BETA}")
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
        plt.title(f"BO with EHVI vs. EHVI-Custom on {func_name}-{dim}D-{f.num_objectives}objectives")
        plt.legend()
        plt.grid(True)
        if os.path.isdir(f"{FIG_DIR}/{func_name}") == False:
            os.makedirs(f"{FIG_DIR}/{func_name}")
        plt.savefig(f"{FIG_DIR}/{func_name}/{func_name}-{dim}-{f.num_objectives}-beta={BETA}.pdf", dpi=300)  # Save plot as PDF
