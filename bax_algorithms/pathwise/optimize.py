from abc import ABC, abstractmethod
from typing import List
from collections.abc import Callable
from torch import Tensor
from xopt.pydantic import XoptBaseModel
from pydantic import Field
import torch
import numpy as np
import time
import warnings
from scipy.optimize import NonlinearConstraint, differential_evolution


class VirtualOptimizer(XoptBaseModel, ABC):
    minimize: bool = Field(
        True,
        description="Whether to minimize (True) or maximize (False) the virtual objective",
    )
    verbose: bool = Field(
        False, description="Whether to print diagnostics during optimization."
    )

    @abstractmethod
    def optimize(
        self,
        virtual_objective: Callable,
        sample_functions_list: List[Callable],
        bounds: Tensor,
        n_samples: int,
        optimization_indeces: Tensor = None,
        constraints: Callable | None = None,
    ) -> Tensor:
        """
        Optimizes virtual objective sample functions and returns optimal inputs.

        constraints, when provided, maps full inputs of shape
        (n_samples, n_points, n_dims) to residuals of shape
        (n_samples, n_points, n_constraints). Every residual must be <= 0.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def _wrap_virtual_objective(
        self,
        virtual_objective: Callable,
        sample_functions_list: list[Callable],
        bounds: Tensor,
        n_samples: int,
        optimization_indeces: Tensor,
    ) -> Callable:
        """
        Wraps virtual objective function so inputs/outputs are suitable for optimization method.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def _get_virtual_optimization_bounds(
        self, bounds: Tensor, n_samples: int, optimization_indeces: Tensor
    ) -> Tensor:
        """
        Get bounds for virtual optimization (may not be the same as bounds passed to optimizer).
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _get_target_function(
        self,
        virtual_objective: Callable,
        sample_functions_list: list[Callable],
        bounds: Tensor,
        n_samples: int,
        optimization_indeces: Tensor,
    ) -> Callable:
        """
        Multiply virtual objective by -1 if not in minimization mode (to maximize instead).
        """
        wrapped_virtual_objective = self._wrap_virtual_objective(
            virtual_objective,
            sample_functions_list,
            bounds,
            n_samples,
            optimization_indeces,
        )

        def target_function(x):
            if not self.minimize:
                return -1.0 * wrapped_virtual_objective(x)
            else:
                return wrapped_virtual_objective(x)

        return target_function


class DifferentialEvolution(VirtualOptimizer):
    popsize: int = Field(
        15,
        description="Number of points sampled in each generation of evolutionary algorithm.",
    )
    maxiter: int = Field(
        100, description="Max number of generations in the evolutionary algorithm."
    )
    polish: bool = Field(
        False,
        description="Whether to use gradient-based optimization after evolution to further optimize result.",
    )

    def optimize(
        self,
        virtual_objective: Callable,
        sample_functions_list: List[Callable],
        bounds: Tensor,
        n_samples: int,
        optimization_indeces: Tensor = None,
        constraints: Callable | None = None,
    ) -> Tensor:
        target_function = self._get_target_function(
            virtual_objective,
            sample_functions_list,
            bounds,
            n_samples,
            optimization_indeces,
        )

        de_bounds = self._get_virtual_optimization_bounds(
            bounds, n_samples, optimization_indeces
        )

        def constraint_function(x):
            full_inputs = self._prepare_inputs(
                x, bounds, n_samples, optimization_indeces
            )
            residuals = constraints(full_inputs)
            if residuals.ndim != 3 or residuals.shape[:2] != full_inputs.shape[:2]:
                raise ValueError(
                    "constraints must return (n_samples, n_points, n_constraints)"
                )
            # Keep every sample's constraints separate rather than summing them.
            residuals = residuals.permute(0, 2, 1).reshape(-1, full_inputs.shape[1])
            residuals = residuals.detach().cpu().numpy()
            residuals = np.where(np.isfinite(residuals), residuals, np.inf)
            return residuals[:, 0] if x.ndim == 1 else residuals

        scipy_constraints = (
            (NonlinearConstraint(constraint_function, -np.inf, 0.0),)
            if constraints is not None
            else ()
        )
        if self.verbose:
            start = time.time()
            print("Beginning BAX virtual optimization.")
        res = differential_evolution(
            target_function,
            bounds=de_bounds,
            constraints=scipy_constraints,
            vectorized=True,
            updating="deferred",
            polish=self.polish,
            popsize=self.popsize,
            maxiter=self.maxiter,
            seed=1,
        )
        if self.verbose:
            print("BAX virtual optimization took:", time.time() - start, "seconds.")

        if constraints is not None:
            if not np.isfinite(res.fun) or not np.all(constraint_function(res.x) <= 0):
                raise RuntimeError("No feasible prediction found; do not use result.x")
            if not res.success:
                warnings.warn(
                    f"Feasible result, but optimizer did not converge: {res.message}",
                    stacklevel=2,
                )

        if optimization_indeces is not None:
            ndim = len(optimization_indeces)
        else:
            ndim = bounds.shape[-1]

        best_x = torch.from_numpy(res.x).to(bounds).reshape(n_samples, 1, ndim)

        return best_x

    @staticmethod
    def _prepare_inputs(x, bounds, n_samples, optimization_indeces):
        x = torch.from_numpy(x).to(bounds)
        dim = int(x.shape[0] / n_samples)
        x = x.reshape(n_samples, dim, -1).transpose(1, 2)
        if optimization_indeces is None:
            return x
        full_inputs = bounds.mean(dim=0).repeat(*x.shape[:-1], 1)
        full_inputs[..., optimization_indeces] = x
        return full_inputs

    def _wrap_virtual_objective(
        self,
        virtual_objective: Callable,
        sample_functions_list: List[Callable],
        bounds: Tensor,
        n_samples: int,
        optimization_indeces: Tensor,
    ) -> Callable:
        def wrapped_virtual_objective(x):
            # SciPy can pass no points when the whole population is infeasible.
            if x.ndim == 2 and x.shape[1] == 0:
                return np.empty(0)
            x_p = self._prepare_inputs(x, bounds, n_samples, optimization_indeces)
            res = virtual_objective(sample_functions_list, x_p, bounds)
            res = res.sum(dim=0).detach().cpu().numpy()
            return res.item() if x.ndim == 1 else res

        return wrapped_virtual_objective

    def _get_virtual_optimization_bounds(
        self, bounds: Tensor, n_samples: int, optimization_indeces: Tensor
    ) -> Tensor:
        """
        Get bounds for virtual optimization (may not be the same as bounds passed to optimizer).
        """
        if optimization_indeces is not None:
            return np.tile(
                bounds[..., optimization_indeces].detach().cpu().numpy().T,
                (n_samples, 1),
            )
        else:
            return np.tile(bounds.detach().cpu().numpy().T, (n_samples, 1))
