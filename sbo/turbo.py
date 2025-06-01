import os
import math
import warnings
import numpy as np
from dataclasses import dataclass
import torch
from botorch.test_functions import *
from botorch.acquisition import qExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from custom_ei import CustomExpectedImprovement
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from constants import OBJECTIVE_FUNCTIONS, DIMS, NUMERICAL_RESULTS_DIR, EXP_RUNS
from tqdm import tqdm
# ==================== Configuration ====================
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

BATCH_SIZE = 1
N_CANDIDATES = None
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
MAX_CHOLESKY_SIZE = float("inf")







def eval_objective(func, x, bounds):
    return func(unnormalize(x, bounds))


def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    return sobol.draw(n=n_pts).to(dtype=dtype, device=device)


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    best_value: float = -float("inf")
    restart_triggered: bool = False
    success_tolerance: int = 10
    failure_tolerance: int = float("nan")

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max(4.0 / self.batch_size, float(self.dim) / self.batch_size))


def update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    y_max = Y_next.max().item()
    if y_max > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, y_max)
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(state, model, X, Y, batch_size, n_candidates, acqf="ts"):
    assert acqf in ("ts", "ei", "custom_ei"), "Invalid acquisition function specified."
    dim = X.shape[-1]
    x_center = X[Y.argmax(), :].clone()

    # Trust region scaling
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        sobol = SobolEngine(dim, scramble=True)
        pert = tr_lb + (tr_ub - tr_lb) * sobol.draw(n_candidates).to(dtype=dtype, device=device)
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = 1
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, Y.max())
        X_next, _ = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
    elif acqf == "custom_ei":
        ei_custom = CustomExpectedImprovement(model, Y.max(), beta=1.0)
        X_next, _ = optimize_acqf(
            ei_custom,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
    return X_next


# ==================== Main TuRBO Loop ====================
def run_turbo(f: SyntheticTestFunction, acqf: str = "ts"):
    
    
        # --- Set up dimension and objective function ---
        if hasattr(f, "dim"):
            DIM = f.dim
            objective_func = f().to(dtype=dtype, device=device)
        elif f.__name__ == "Hartmann":
            DIM = 6
            objective_func = f().to(dtype=dtype, device=device)
        else:
            DIM = DIMS[f]
            objective_func = f(dim=DIM, negate=True).to(dtype=dtype, device=device)
        BOUNDS = objective_func.bounds
        print(f"Running TuRBO on {f.__name__} with dim={DIM} and acquisition function {acqf}")

        N_INIT = DIM + 1 
        N_CANDIDATES = min(5000, max(2000, 200 * DIM)) if not SMOKE_TEST else 4
        N_ITERATIONS = 100 if DIM <= 10 else 500
        

        # torch.manual_seed(0)
        # check if save file exists
        func_name = objective_func.__class__.__name__
            
        save_path = f"{NUMERICAL_RESULTS_DIR}/{func_name}_{DIM}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        results_path = f"{save_path}/{func_name}_{DIM}_best_values_turbo-{acqf}.npy"
        
        if os.path.exists(results_path):
            # print(f"Results already exist for {func_name} with dim={DIM} and acquisition function {acqf}. Loading...")
            best_values_runs = np.load(results_path)
            if best_values_runs.shape[0] >= EXP_RUNS:
                print(f"Skipping {func_name} with dim={DIM} and acquisition function {acqf} as results already exist.")
                return best_values_runs
            else:  
                best_values_runs = best_values_runs.tolist()
        else:
            best_values_runs = []
        n_done = len(best_values_runs)
        for run in range(n_done, EXP_RUNS):
            X = get_initial_points(DIM, N_INIT)
            Y = torch.tensor([eval_objective(objective_func, x, bounds=BOUNDS) for x in X], dtype=dtype, device=device).unsqueeze(-1)

            state = TurboState(dim=DIM, batch_size=BATCH_SIZE, best_value=Y.max().item())
            best_values_turbo = [-state.best_value]
            print(objective_func.__class__.__name__, "TURBO", run)
            with tqdm(total=N_ITERATIONS, desc=f"Running TuRBO on {f.__name__} with dim={DIM} and acquisition function {acqf}") as pbar:
                
                for it in range(N_ITERATIONS):
                    train_Y = (Y - Y.mean()) / Y.std()

                    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                    covar_module = ScaleKernel(
                        MaternKernel(nu=2.5, ard_num_dims=DIM, lengthscale_constraint=Interval(0.005, 4.0))
                    )
                    model = SingleTaskGP(X, train_Y, covar_module=covar_module, likelihood=likelihood)
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)

                    with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
                        fit_gpytorch_mll(mll)
                        X_next = generate_batch(state, model, X, train_Y, BATCH_SIZE, N_CANDIDATES, acqf=acqf)

                    Y_next = torch.tensor([eval_objective(objective_func, x, BOUNDS) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)

                    state = update_state(state, Y_next)
                    best_values_turbo.append(-state.best_value)

                    X = torch.cat((X, X_next), dim=0)
                    Y = torch.cat((Y, Y_next), dim=0)
                    pbar.set_description(f"Run {run+1}/{EXP_RUNS}, Iteration {it+1}/{N_ITERATIONS}, Best Value: {-state.best_value:.3e}, TR Length: {state.length:.3e}")
                    pbar.update(1)                
            # print(f"{len(X):>4}) Best value: {state.best_value:.3e}, TR length: {state.length:.3e}")
            
            best_values_runs.append(best_values_turbo.copy())
        
            # best_values_runs = np.array(best_values_runs)
            
            # if not os.path.exists(f"{NUMERICAL_RESULTS_DIR}/{func_name}"):
            #     os.makedirs(f"{NUMERICAL_RESULTS_DIR}/{func_name}")
            
            np.save(save_path, np.array(best_values_runs))   
        


if __name__ == "__main__":
    for f in OBJECTIVE_FUNCTIONS:
        run_turbo(f, "ts")
        run_turbo(f, "custom_ei")
        run_turbo(f, "ei")
        # func_name = objective_func.__class__.__name__
            # if not os.path.exists(f"{NUMERICAL_RESULTS_DIR}/{func_name}"):
            #     os.makedirs(f"{NUMERICAL_RESULTS_DIR}/{func_name}")

        # np.save(f"{NUMERICAL_RESULTS_DIR}/{func_name}/{func_name}_best_values_ei.npy", best_values_turbo)
