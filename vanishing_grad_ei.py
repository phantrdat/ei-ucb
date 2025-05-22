import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Branin, Hartmann, Ackley
from botorch.utils.sampling import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# --- Define your objective functions and DIMS as needed ---
OBJECTIVE_FUNCTIONS = [Ackley]
DIMS = {Ackley: 10}
tol = 1e-53
dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Gradient vanishing fraction utility ---
def fraction_vanishing_ei_gradient(
    acqf_func, bounds, n_grid=100, tol=1e-6,
    device=torch.device("cpu"), dtype=torch.double,
    batch_size=1000, seed=0, norm_type=2
):
    """
    Compute the fraction of points where the gradient of the acquisition function vanishes
    using efficient vectorized operations where possible.
    
    This version processes multiple points at once for better efficiency.
    
    Args:
        acqf_func: The acquisition function
        bounds: Tensor of bounds for the input space, shape [2, d]
        n_grid: Number of points to evaluate
        tol: Tolerance for considering a gradient as vanishing
        device: Torch device
        dtype: Torch data type
        batch_size: Batch size for processing points (to avoid memory issues)
        seed: Random seed for reproducibility
        norm_type: Type of norm to use (1, 2, or float('inf'))
        
    Returns:
        Fraction of points with vanishing gradients
    """
    d = bounds.shape[1]
    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    X_raw = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(n_grid).to(device=device, dtype=dtype)
    vanish_count = 0
    
    # Process in batches to avoid memory issues
    num_batches = (n_grid + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_grid)
        
        # Get current batch
        X_batch = X_raw[start_idx:end_idx].detach().clone().requires_grad_(True)
        
        # BoTorch acquisition functions typically expect input shape [batch_size, 1, d]
        # So reshape accordingly
        X_batch_reshaped = X_batch.unsqueeze(1)
        
        # Compute acquisition function values for the entire batch
        with torch.enable_grad():
            ei_vals = acqf_func(X_batch_reshaped)
            
            # We need to compute gradients for each point in the batch separately
            # since we want to check if each individual gradient is vanishing
            for i in range(X_batch.shape[0]):
                # Reset gradients
                if X_batch.grad is not None:
                    X_batch.grad.zero_()
                
                # Compute gradient for this point
                ei_val = ei_vals[i]
                ei_val.backward(retain_graph=True)
                
                # Get gradient for current point
                grad = X_batch.grad[i]
                
                # Calculate the norm based on norm_type
                if norm_type == float('inf'):
                    grad_norm = grad.abs().max().item()
                else:
                    grad_norm = grad.norm(p=norm_type).item()
                    
                if grad_norm <= tol:
                    vanish_count += 1
    
    return vanish_count / n_grid

# --- Main BO loop ---
N = 1

for f in OBJECTIVE_FUNCTIONS:
    dim = getattr(f, "dim", DIMS[f])
    objective_func = f(dim=dim).to(dtype=dtype, device=device) if f != Branin else f().to(dtype=dtype, device=device)
    bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)
    num_initial_points = 5
    num_iterations = 150 if dim <= 10 else 200
    sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
    fixed_train_X = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_initial_points).to(dtype=dtype, device=device)
    fixed_train_Y = objective_func(fixed_train_X).unsqueeze(-1)

    def bayesian_optimization(kernel_type="matern"):
        best_values_runs = []
        frac_vanish_runs = []
        for run in range(N):
            print(objective_func.__class__.__name__, kernel_type, "EI", run)
            train_X = fixed_train_X.clone()
            train_Y = fixed_train_Y.clone()
            best_values = [train_Y.min().item()]
            frac_vanish = []
            vanish_count = 0
            for i in tqdm(range(num_iterations)):
                if kernel_type == "matern":
                    covar_module = MaternKernel(nu=2.5)
                elif kernel_type == "se":
                    covar_module = RBFKernel()
                else:
                    raise ValueError("Unsupported kernel type")

                gp = SingleTaskGP(train_X, train_Y, covar_module=covar_module, outcome_transform=Standardize(m=1))
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)

                acq_func = ExpectedImprovement(model=gp, best_f=train_Y.min(), maximize=False)
                frac = fraction_vanishing_ei_gradient(acq_func, bounds,
                                                      n_grid=10000, tol=tol,
                                                      device=device, dtype=dtype)
                


                # candidate, _ = optimize_acqf(acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=100)

                # clone_candidate = candidate.detach().clone().requires_grad_(True)
                # ei_val_candidate = acq_func(clone_candidate.unsqueeze(1))
                # ei_val_candidate.backward(retain_graph=True)
                # grad = clone_candidate.grad
                # if norm_type == float('inf'):
                #     grad_norm = grad.abs().max().item()
                # else:
                # grad_norm = grad.norm(p=2).item()
                    
                # if grad_norm <= tol:
                #     vanish_count += 1
                frac_vanish.append(frac)

                new_Y = objective_func(candidate).unsqueeze(-1)
                train_X = torch.cat([train_X, candidate])
                train_Y = torch.cat([train_Y, new_Y])
                best_values.append(train_Y.min().item())

            best_values_runs.append(best_values)
            frac_vanish_runs.append(frac_vanish)

        return np.array(best_values_runs), np.array(frac_vanish_runs)

    results = {}
    for kernel_type in ["matern", "se"]:
        best_values, vanish = bayesian_optimization(kernel_type)
        results[kernel_type] = {
            "best_values": best_values,
            "vanish": vanish,
            "mean_best": best_values.mean(axis=0),
            "std_best": best_values.std(axis=0),
            "mean_vanish": vanish.mean(axis=0),
        }

    iterations = np.arange(len(results["matern"]["mean_vanish"]))
    func_name = objective_func.__class__.__name__

    # Plot comparison of best function values
    # plt.figure(figsize=(8, 5))
    # for kernel_type, color in zip(["matern", "se"], ["b", "g"]):
    #     mean = results[kernel_type]["mean_best"]
    #     std = results[kernel_type]["std_best"]
    #     plt.plot(iterations, mean, label=f"EI-{kernel_type}", color=color)
    #     plt.fill_between(iterations, mean - std, mean + std, color=color, alpha=0.2)
    # plt.xlabel("Iteration")
    # plt.ylabel("Best Function Value Found")
    # plt.title(f"BO with EI on {func_name}-{dim}D (Matern vs SE)")
    # plt.legend()
    # plt.grid(True)
    # os.makedirs(f"results/{func_name}", exist_ok=True)
    # plt.savefig(f"results/{func_name}/{func_name}-{dim}-ei-kernel-comp.pdf", dpi=300)
    # plt.close()

    # Plot vanishing gradient comparison
    plt.figure(figsize=(8, 4))
    for kernel_type, color in zip(["matern", "se"], ["b", "g"]):
        plt.plot(iterations, results[kernel_type]["mean_vanish"], label=f"EI-{kernel_type}", color=color)
    plt.xlabel("Iteration")
    plt.ylabel("Fraction with ||grad EI|| < ")
    plt.title(f"Vanishing Gradient Fraction on {func_name}-{dim}D")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{func_name}/{func_name}-{dim}-ei-kernel-comp-vanishfrac.pdf", dpi=300)
    plt.close()