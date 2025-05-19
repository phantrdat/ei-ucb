from botorch.acquisition.analytic import AnalyticAcquisitionFunction, _scaled_improvement, _ei_helper, UpperConfidenceBound
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor
from botorch.acquisition.objective import PosteriorTransform
import torch
from botorch.utils.probability.utils import (
    compute_log_prob_feas_from_bounds,
    log_ndtr as log_Phi,
    log_phi,
    ndtr as Phi,
    phi,
)
def _custom_ei_helper(u: Tensor) -> Tensor:
    """Computes (u^2+1) * Phi(u) + u* phi(u) , where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return u*phi(u) + (u**2+1) * Phi(u)


class CustomExpectedImprovement(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        beta: float | Tensor, 
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ):
        
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize
            
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        ei =  sigma * _ei_helper(u)
        expected_of_i_square = sigma**2 * _custom_ei_helper(u)
        ei_std = torch.sqrt(expected_of_i_square - ei**2)
        return ei + self.beta.sqrt() * ei_std