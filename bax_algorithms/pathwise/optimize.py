from abc import ABC, abstractmethod
from collections.abc import Callable
from torch import Tensor
from xopt.pydantic import XoptBaseModel

class VirtualOptimizer(XoptBaseModel, ABC):
    minimize: bool = Field(True,
        description = "Whether to minimize (True) or maximize (False) the virtual objective")

    @abstractmethod
    def optimize(self,
                 virtual_objective: Callable,
                 sample_functions_list: list[Callable],
                 xopt_bounds: Tensor,
                 virtual_optimization_bounds: Tensor) -> Tensor:
        '''
        Minimizes virtual objective sample functions and returns optimal inputs.
        '''

    @abstractmethod
    def wrap_virtual_objective(self,
                           virtual_objective: Callable, 
                           sample_functions_list: list[Callable], 
                           bounds: Tensor) -> Callable:
        '''
        Wraps virtual objective function so inputs/outputs are suitable for optimization method.
        '''

    def get_target_function(self,
                           virtual_objective: Callable, 
                           sample_functions_list: list[Callable], 
                           bounds: Tensor) -> Callable:
        '''
        Multiply virtual objective by -1 if not in minimization mode (to maximize instead).
        '''
        wrapped_virtual_objective = wrap_virtual_objective(virtual_objective,
                                                           sample_functions_list,
                                                           bounds)
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
    maxiter: int = Field(10,
        description="Max number of generations in the evolutionary algorithm.")
    polish: bool = Field(False,
        description="Whether to use gradient-based optimization after evolution to further optimize result.")

    def optimize(self,
                 virtual_objective: Callable, 
                 sample_functions_list: list[Callable],
                 xopt_bounds: Tensor,
                 virtual_optimization_bounds: Tensor) -> Tensor:

        target_function = self.get_target_function(virtual_objective,
                                                    sample_functions_list,
                                                    xopt_bounds)

        res = differential_evolution(target_function, 
                                     bounds=virtual_optimization_bounds.numpy(), 
                                     vectorized=True, 
                                     polish=self.polish, 
                                     popsize=self.popsize, 
                                     maxiter=self.maxiter, 
                                     seed=1)
        
        best_x = torch.from_numpy(res.x)
        
        return best_x
        
    def wrap_virtual_objective(self,
                               virtual_objective: Callable, 
                               sample_functions_list: list[Callable], 
                               bounds: Tensor) -> Callable:

        def wrapped_virtual_objective(x):
            x = torch.from_numpy(x)
            res = virtual_objective(sample_funcs_list, x.T.unsqueeze(0), bounds)
            return res[0,:].numpy()

        return wrapped_virtual_objective