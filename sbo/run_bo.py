import torch
import os
from botorch.test_functions import *
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement 
from botorch.acquisition.logei import qLogExpectedImprovement
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from test_functions import *
from custom_ei import CustomExpectedImprovement
from constants import *
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
from botorch.models.transforms.outcome import Standardize
from torch.quasirandom import SobolEngine
from botorch.sampling import SobolQMCNormalSampler
# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models
import numpy as np
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

# Define list function tests


def single_objective_bayesian_optimization(
        acq_type: str, 
        f: SyntheticTestFunction, 
        beta: Optional[float] = None,
        maximize_mode: bool=False) -> np.ndarray:

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
    # Create directory for numerical results if it doesn't exist
    func_name = objective_func.__class__.__name__ 
    save_path = f"{NUMERICAL_RESULTS_DIR}/{func_name}_{dim}"   
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    results_path = f"{save_path}/{func_name}_{dim}_best_values_{acq_type.lower()}.npy" if beta is None else f"{save_path}/{func_name}_{dim}_best_values_ei_custom_beta={beta}.npy"
    # Define bounds for the optimization
    
    # Experiment settings
    bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)  # Search space bounds
    num_initial_points = dim + 1
    num_iterations = 100 if dim <= 10 else 200
    # Generate initial training data
    sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
    fixed_train_X  = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_initial_points).to(dtype=dtype, device=device)
    fixed_train_Y  = objective_func(fixed_train_X).unsqueeze(-1) # Evaluate function and reshape

    if os.path.exists(results_path):
        # print(f"Results already exist for {func_name} with dim={DIM} and acquisition function {acqf}. Loading...")
        best_values_runs = np.load(results_path)
        if best_values_runs.shape[0] >= EXP_RUNS:
            print(f"Skipping {func_name} with dim={dim} and acquisition function {acq_type} as results already exist.")
            return best_values_runs
        else:  
            best_values_runs = best_values_runs.tolist()
    else:
        best_values_runs = []
    n_done = len(best_values_runs)
    for run in range(n_done, EXP_RUNS):
        print(objective_func.__class__.__name__, acq_type, run)
        train_X = fixed_train_X.clone()
        train_Y = fixed_train_Y.clone()
        # Track best observed function values
        if maximize_mode:
            train_Y = -train_Y  # Flip for maximization

        best_values = [train_Y.max().item() if maximize_mode else train_Y.min().item()]
        
        for i in range(num_iterations):
            gp_model = SingleTaskGP(train_X, train_Y, covar_module=ScaleKernel(RBFKernel()), outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            fit_gpytorch_mll(mll)
            # gp_hyperparams = {
            #     "lengthscale": gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy(),
            #     "outputscale": gp_model.covar_module.outputscale.detach().cpu().numpy(),
            #     "noise": gp_model.likelihood.noise.detach().cpu().numpy()
            # }
            # print(gp_hyperparams)
            if acq_type.lower() == "ei":
                acq_func = ExpectedImprovement(model=gp_model, best_f=train_Y.min(), maximize=False)
            elif acq_type.lower() == "logei":
                acq_func = LogExpectedImprovement(model=gp_model, best_f=train_Y.min(), maximize=False)
            elif acq_type.lower() == "ei-custom":
                # beta = 10  # Exploration parameter for EI Custom
                beta_value = beta if beta is not None else 1.0  # Provide a default float value
                acq_func = CustomExpectedImprovement(model=gp_model, best_f=train_Y.min(), beta=beta_value, maximize=False)
            elif acq_type.lower() == "qlogei":
                sampler = SobolQMCNormalSampler(torch.Size([1024]))
                acq_func = qLogExpectedImprovement(model=gp_model, best_f=train_Y.max(), sampler=sampler)  
            else:
                raise ValueError("Invalid acquisition function type")
            # Optimize the acquisition function to find the next query point
            candidate, _ = optimize_acqf(
                acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=100
            )
            # Evaluate the function at the new point
            new_Y = objective_func(candidate).unsqueeze(-1)
            if maximize_mode:
                new_Y = -new_Y  # flip sign
            # Update the dataset
            train_X = torch.cat([train_X, candidate])
            train_Y = torch.cat([train_Y, new_Y])
            # Store best observed value
       
            best_values.append(train_Y.max().item() if maximize_mode else train_Y.min().item())
            
        best_values_runs.append(best_values) if acq_type.lower() != "qlogei" else best_values_runs.append(-np.array(best_values))
        np.save(results_path, np.array(best_values_runs))
          
    return np.array(best_values_runs)
def main():  
    for f in OBJECTIVE_FUNCTIONS: 
        # Call EI 
        single_objective_bayesian_optimization("EI", f)
        # Call LogEI 
        single_objective_bayesian_optimization("LogEI", f)


        # Call qLogEI 
        best_values_qlogei = single_objective_bayesian_optimization("qLogEI", f, maximize_mode=True)
        best_values_qlogei = -best_values_qlogei  # Flip back to match EI/LogEI results

        # Call EI-Custom
        for beta in [1,5,10,15, 20, 25, 30, 35, 40]:            
            single_objective_bayesian_optimization("EI-Custom", f, beta=beta)


if __name__ == "__main__":
    main()
    print("All experiments completed.")