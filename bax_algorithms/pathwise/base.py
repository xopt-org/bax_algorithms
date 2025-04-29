# to be added to basic algorithms in Xopt

from abc import ABC, abstractmethod
from bax_algorithms.pathwise.optimize import VirtualOptimizer, DifferentialEvolution
from botorch.models.model import Model, ModelList

class PathwiseOptimization(Algorithm, ABC):
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

    def execute_algorithm(self, model: Model, bounds: Tensor) -> Tensor:
        '''
        Run virtual algorithm on pathwise function samples.
        '''

        # draw callable sample functions
        self.sample_functions_list = self.draw_sample_functions_list(model)

        virtual_optimization_bounds = self.get_virtual_optimization_bounds(xopt_bounds = bounds)

        # optimize sample functions
        self.best_x = self.optimizer.optimize(virtual_objective = self.evaluate_virtual_objective,
                                              sample_functions_list = self.sample_functions_list,
                                              xopt_bounds = bounds,
                                              virtual_optimization_bounds = virtual_optimization_bounds)

        # get execution paths
        exe_paths = self.get_execution_paths(self.best_x)

        return exe_paths

    @abstractmethod
    def get_execution_paths(self, best_x: Tensor) -> Tensor:
        '''
        Get execution paths from optimization result.
        '''

    @abstractmethod
    def draw_sample_functions_list(self, model: Model) -> List:
        '''
        Generates a callable function sample object for each observable model
        and stores them in list ordered according to observable_names_ordered.
        '''

    @abstractmethod
    def get_virtual_optimization_bounds(self, xopt_bounds: Tensor) -> Tensor:
        '''
        Get bounds for virtual objective optimization.
        '''
