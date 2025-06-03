# to be added to basic algorithms in Xopt

from abc import abstractmethod
from bax_algorithms.pathwise.optimize import VirtualOptimizer, DifferentialEvolution
from botorch.models.model import Model, ModelList
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
from pydantic import Field
from xopt.generators.bayesian.bax.algorithms import Algorithm
from torch import Tensor
import torch
from typing import List
from collections.abc import Callable


class PathwiseOptimization(Algorithm):
    """
    Base algorithm for BAX pathwise function sample minimization.

    Attributes:
    -----------
    name : str
        The name of the algorithm.
    optimizer : VirtualOptimizer
        The optimizer to be used in virtual optimization of the sample functions.

    Methods:
    --------
    execute_algorithm(self, model: Model, bounds: Tensor) -> Tensor
        Run virtual algorithm on pathwise function samples and return
        execution paths.

    get_execution_paths(self, model: Model, bounds: Tensor) -> Tensor
        Get execution paths from virtual optimization result.

    draw_sample_functions_list(self, model: Model) -> List
        Generate callable function samples from GP model.

    get_virtual_optimization_bounds(self, xopt_bounds: Tensor) -> Tensor
        Get the bounds for virtual optimization.
    """

    name = "pathwise_optimization"
    optimizer: VirtualOptimizer = Field(DifferentialEvolution(),
        description = "Optimizer for virtual objective.")
    results: dict = Field(
        default=None,
        description="dictionary containing algorithm results",
    )
    observable_names_ordered: List[str] = Field(
        default=None,
        description="names of observable models used in this algorithm",
    )


    def execute_algorithm(self, sample_functions_list: List[Callable], bounds: Tensor) -> Tensor:
        '''
        Run virtual algorithm on pathwise function samples.
        '''

        optimization_indeces = self._get_optimization_indeces(bounds)

        # optimize sample functions
        best_inputs = self.optimizer.optimize(virtual_objective = self.evaluate_virtual_objective,
                                              sample_functions_list = sample_functions_list,
                                              bounds = bounds,
                                              optimization_indeces = optimization_indeces,
                                              n_samples = self.n_samples)

        return best_inputs

    def get_execution_paths(self, model: Model, bounds: Tensor) -> Tensor:
        '''
        Execute algorithm and get execution paths from optimization result.
        '''
        
        # draw callable sample functions
        sample_functions_list = self.draw_sample_functions_list(model)

        best_inputs = self.execute_algorithm(sample_functions_list, bounds)
        best_objective = self.evaluate_virtual_objective(sample_functions_list, best_inputs, bounds)
        
        self.results = {}
        self.results['best_inputs'] = best_inputs
        self.results['best_objective'] = best_objective
        self.results['sample_functions_list'] = sample_functions_list

        return best_inputs, best_objective, self.results      

    def draw_sample_functions_list(self, model: Model) -> List:
        '''
        Generates a callable function sample object for each observable model
        and stores them in list ordered according to observable_names_ordered.
        '''
        sample_funcs_list = [draw_matheron_paths(m, sample_shape=torch.Size([self.n_samples])) for m in model.models]
        return sample_funcs_list

    def _get_optimization_indeces(self, bounds) -> Tensor:
        '''
        Get indeces specifying parameters for virtual objective optimization.
        '''
        return torch.tensor(range(bounds.shape[1]))

    def evaluate_virtual_observables(
        self,
        model: Model,
        x: Tensor,
        n_samples: int = None,
    ) -> Tensor:
        '''
        Evaluate observable models. model must either be a ModelList (GP) or a list of callable function samples.
        '''
        if isinstance(model, ModelList):
            assert len(x.shape) == 2 # x.shape should equal (n_points, ndim)
            p = model.posterior(x)
            vobs = p.sample(torch.Size([n_samples])) # vobs.shape will be (n_samples, n_points, num_outputs)
        else:
            assert n_samples is None
            assert len(x.shape) in [2,3] # x.shape can be (n_samples, n_points, ndim) for samplewise evaluation
                                         # or (n_points, ndim) for broadcasting to all samples
            vobs_list = [sample_funcs(x) for sample_funcs in model]
            vobs = torch.stack(vobs_list, dim=-1) # vobs.shape will be (n_samples, n_points, num_outputs)
        return vobs