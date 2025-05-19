import torch
from botorch.test_functions import SyntheticTestFunction

class Alpine1(SyntheticTestFunction):
    """Alpine 1 test function.
    
    f(x) = sum |x_i * sin(x_i) + 0.1 * x_i|
    
    The function is typically evaluated on [-10, 10]^d.
    The global minimum is at x* = (0, ..., 0) with f(x*) = 0.
    """
    _optimal_value = 0.0
    
    def __init__(self, dim=2, noise_std=None, negate=False, dtype=torch.double):
        self.dim = dim
        bounds = [(-10.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, dtype=dtype)
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(X * torch.sin(X) + 0.1 * X), dim=-1)

class Alpine2(SyntheticTestFunction):
    """Alpine 2 test function.
    
    f(x) = prod sqrt(x_i) * sin(x_i)
    
    The function is typically evaluated on [0, 10]^d.
    The global minimum is at x* = (7.917, ..., 7.917) with f(x*) = 2.808^d.
    """
    def __init__(self, dim=2, noise_std=None, negate=False, dtype=torch.double):
        self.dim = dim
        bounds = [(0.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(7.917 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, dtype=dtype)
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return torch.prod(torch.sqrt(X) * torch.sin(X), dim=-1)