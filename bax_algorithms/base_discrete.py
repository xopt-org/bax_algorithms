from abc import ABC, abstractmethod
from typing import Callable, Tuple
from pydantic import Field
import torch
from torch import Tensor

from botorch.models.model import Model, ModelList
from xopt.generators.bayesian.bax.algorithms import Algorithm

class BaseDiscreteAlgoFn(ABC):
    @abstractmethod
    def __call__(self, posterior_samples: Tensor, x_grid: Tensor, **algo_kwargs) -> Tuple[Tensor, Tensor]:
        pass

class FunctionWrapper(BaseDiscreteAlgoFn):
    def __init__(self, fn: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]):
        self.fn = fn

    def __call__(self, posterior_samples: Tensor, x_grid: Tensor, **algo_kwargs) -> Tuple[Tensor, Tensor]:
        return self.fn(posterior_samples, x_grid, **algo_kwargs)
    
class DiscreteSubsetAlgorithm(Algorithm, ABC):
    algo_fn: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = Field(None,
                              description="Python function defining a BAX algorithm on a discrete grid")
    grid: Tensor = Field(None,
                         description="n-d grid of discrete points")
    observable_names_ordered: list[str] = Field(["y1"],
        description="keys designating output properties")
    algo_kwargs: dict = Field({},
        description="keyword args for generic subset algorithm")
        
    def get_execution_paths(self, model: Model, bounds: Tensor):
        test_points = self.grid
        
        if isinstance(model, ModelList):
            test_points = test_points.to(model.models[0].train_targets)
        else:
            test_points = test_points.to(model.train_targets)

        # get samples of the model posterior at mesh points
        posterior_samples = self.evaluate_virtual_objective(
            model, test_points, bounds, self.n_samples
        )
        
        # wrap if needed
        if not isinstance(self.algo_fn, BaseDiscreteAlgoFn):
            self.algo_fn = FunctionWrapper(self.algo_fn)

        x_opt, y_opt = self.algo_fn(posterior_samples, test_points, **self.algo_kwargs)

        # get the solution_center and solution_entropy for Turbo
        # note: the entropy calc here drops a constant scaling factor
        solution_center = x_opt.mean(dim=0).numpy()
        solution_entropy = float(torch.log(x_opt.std(dim=0) ** 2).sum())

        # collect secondary results in a dict
        results_dict = {
            "test_points": test_points,
            "posterior_samples": posterior_samples,
            "execution_paths": torch.hstack((x_opt, y_opt)),
            "solution_center": solution_center,
            "solution_entropy": solution_entropy,
        }

        # return execution paths
        return x_opt.unsqueeze(-2), y_opt.unsqueeze(-2), results_dict

    def evaluate_virtual_objective(
        self,
        model: Model,
        x: Tensor,
        bounds: Tensor,
        n_samples: int,
        tkwargs: dict = None,
    ) -> Tensor:
        """
        Evaluate the virtual objective (samples).

        Parameters:
        -----------
        model : Model
            The model to use for evaluating the virtual objective.
        x : Tensor
            The inputs at which to evaluate the virtual objective.
        bounds : Tensor
            The bounds for the optimization.
        n_samples : int
            The number of samples to generate.
        tkwargs : dict, optional
            Additional keyword arguments for the evaluation.

        Returns:
        --------
        Tensor
            The evaluated virtual objective values.
        """
        # get samples of the model posterior at inputs given by x
        with torch.no_grad():
            post = model.posterior(x)
            objective_values = post.rsample(torch.Size([n_samples]))

        return objective_values
