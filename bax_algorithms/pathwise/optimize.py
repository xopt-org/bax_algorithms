from abc import ABC, abstractmethod
from typing import List
from collections.abc import Callable
from torch import Tensor
from xopt.pydantic import XoptBaseModel
from pydantic import Field
import torch
import numpy as np
import time


class VirtualOptimizer(XoptBaseModel, ABC):
    minimize: bool = Field(True,
        description = "Whether to minimize (True) or maximize (False) the virtual objective")
    verbose: bool = Field(False,
        description="Whether to print diagnostics during optimization.")


    @abstractmethod
    def optimize(self,
                 virtual_objective: Callable,
                 sample_functions_list: List[Callable],
                 bounds: Tensor,
                 n_samples: int,
                 optimization_indeces: Tensor = None,
                ) -> Tensor:
        '''
        Minimizes virtual objective sample functions and returns optimal inputs.
        '''

    @abstractmethod
    def _wrap_virtual_objective(self,
                           virtual_objective: Callable, 
                           sample_functions_list: list[Callable], 
                           bounds: Tensor,
                           n_samples: int,
                           optimization_indeces: Tensor,
                               ) -> Callable:
        '''
        Wraps virtual objective function so inputs/outputs are suitable for optimization method.
        '''

    @abstractmethod
    def _get_virtual_optimization_bounds(self, bounds: Tensor, n_samples: int, optimization_indeces: Tensor) -> Tensor:
        '''
        Get bounds for virtual optimization (may not be the same as bounds passed to optimizer).
        '''

    def _get_target_function(self,
                           virtual_objective: Callable, 
                           sample_functions_list: list[Callable], 
                           bounds: Tensor,
                           n_samples: int,
                           optimization_indeces: Tensor,
                            ) -> Callable:
        '''
        Multiply virtual objective by -1 if not in minimization mode (to maximize instead).
        '''
        wrapped_virtual_objective = self._wrap_virtual_objective(virtual_objective,
                                                           sample_functions_list,
                                                           bounds,
                                                           n_samples,
                                                           optimization_indeces,
                                                                )
        def target_function(x):
            if not self.minimize:
                return -1. * wrapped_virtual_objective(x)
            else:
                return wrapped_virtual_objective(x)

        return target_function

from scipy.optimize import differential_evolution
class DifferentialEvolution(VirtualOptimizer):
    popsize: int = Field(15,
        description="Number of points sampled in each generation of evolutionary algorithm.")
    maxiter: int = Field(100,
        description="Max number of generations in the evolutionary algorithm.")
    polish: bool = Field(False,
        description="Whether to use gradient-based optimization after evolution to further optimize result.")


    def optimize(self,
                 virtual_objective: Callable, 
                 sample_functions_list: List[Callable],
                 bounds: Tensor,
                 n_samples: int,
                 optimization_indeces: Tensor = None,
                ) -> Tensor:

        target_function = self._get_target_function(virtual_objective,
                                                    sample_functions_list,
                                                    bounds,
                                                    n_samples,
                                                    optimization_indeces)

        de_bounds = self._get_virtual_optimization_bounds(bounds, n_samples, optimization_indeces)

        if self.verbose:
            start = time.time()
            print('Beginning sample optimization.')
        res = differential_evolution(target_function, 
                                     bounds=de_bounds, 
                                     vectorized=True, 
                                     polish=self.polish, 
                                     popsize=self.popsize, 
                                     maxiter=self.maxiter, 
                                     seed=1)
        if self.verbose:
            print('Sample optimization took:', time.time()-start, 'seconds.')

        if optimization_indeces is not None:
            ndim = len(optimization_indeces)
        else:
            ndim = bounds.shape[-1]

        best_x = torch.from_numpy(res.x).reshape(n_samples, 1, ndim)

        return best_x
        
    def _wrap_virtual_objective(self,
                               virtual_objective: Callable, 
                               sample_functions_list: List[Callable], 
                               bounds: Tensor,
                               n_samples: int,
                               optimization_indeces: Tensor,
                               ) -> Callable:

        def wrapped_virtual_objective(x):
            x = torch.from_numpy(x)
            # x.shape = num_params x popsize*num_params // num_params = dim*n_samples
            dim = int(x.shape[0]/n_samples)
            x = x.reshape(n_samples,dim,-1)
            x = torch.swapaxes(x,1,2)
            if optimization_indeces is not None:
                x_p = bounds.mean(dim=0).repeat(*x.shape[:-1],1)
                x_p[...,optimization_indeces] = x
            else:
                x_p = x
            res = virtual_objective(sample_functions_list, x_p, bounds)
            res = res.sum(dim=0).detach().numpy()
            return res

        return wrapped_virtual_objective

    def _get_virtual_optimization_bounds(self, bounds: Tensor, n_samples: int, optimization_indeces: Tensor) -> Tensor:
        '''
        Get bounds for virtual optimization (may not be the same as bounds passed to optimizer).
        '''
        if optimization_indeces is not None:
            return np.tile(bounds[...,optimization_indeces].numpy().T, (n_samples,1))
        else:
            return np.tile(bounds.numpy().T, (n_samples,1))