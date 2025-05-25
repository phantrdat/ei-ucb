from constants import *

from torch.quasirandom import SobolEngine
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

def sample_data(objectives, num_samples):
    """
    Sample random data points from the objectives.
    """
    dim = objectives.dim
    sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
    bounds = objectives.bounds
    X = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_samples)
    Y = objectives(X)
    return X, Y

def compute_hypervolume(objectives, num_samples=10):
    """
    Compute the hypervolume of the Pareto front defined by the objectives.
    """
    X, Y = sample_data(objectives, num_samples)
    # Assuming objectives is a callable that returns the objective values
    bd = DominatedPartitioning(ref_point=objectives.ref_point, Y=Y)
    volume = bd.compute_hypervolume().item()
    return volume

def check_hypervolume():
    """
    Check the hypervolume for each objective function.
    """
    results = {}
    for name, func in OBJECTIVE_FUNCTIONS.items():
        args = FUNCTION_ARGS[func]
        args["negate"] = True
        objectives = func(**args)
        hv = compute_hypervolume(objectives, 2*objectives.dim+1)
        print(f"Hypervolume for {name} with negate=True: {hv:.4f}")
        args["negate"] = False
        objectives = func(**args)
        hv = compute_hypervolume(objectives, 2*objectives.dim+1)
        results[name] = hv
        print(f"Hypervolume for {name} with negate=False: {hv:.4f}")
    return results

if __name__ == "__main__":
    results = check_hypervolume()