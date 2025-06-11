from xopt.generators.bayesian.bax_generator import BaxGenerator
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from bax_algorithms.pathwise.optimize import VirtualOptimizer
import torch


def get_bax_model_and_bounds(generator: BaxGenerator):
    bax_model_ids = [
        generator.vocs.output_names.index(name)
        for name in generator.algorithm.observable_names_ordered
    ]
    model = generator.train_model()
    bax_model = model.subset_output(bax_model_ids)

    if isinstance(bax_model, SingleTaskGP):
        bax_model = ModelListGP(bax_model)

    return bax_model, generator._get_optimization_bounds()

def get_bax_mean_prediction(generator: BaxGenerator, mean_optimizer: VirtualOptimizer):
    model, bounds = get_bax_model_and_bounds(generator)
    def get_mean_function(model):
        def func(x):
            gp_mean = model.posterior(x).mean
            return gp_mean
        return func
    mean_functions_list = [get_mean_function(m) for m in model.models]

    optimization_indeces = generator.algorithm._get_optimization_indeces(bounds)

    # optimize mean functions
    best_inputs = mean_optimizer.optimize(virtual_objective = generator.algorithm.evaluate_virtual_objective,
                                          sample_functions_list = mean_functions_list,
                                          bounds = bounds,
                                          optimization_indeces = optimization_indeces,
                                          n_samples = 1)

    return best_inputs.squeeze(0)

def tuning_input_tensor_to_dict(generator, x_tuning):
    """
    Converts a single set of tuning parameters to a dictionary for input to Xopt
    
    x_tuning = tensor of shape (1, n_tuning_dims)
    """
    _, bounds = get_bax_model_and_bounds(generator)
    optimization_indeces = generator.algorithm._get_optimization_indeces(bounds)
    tuning_parameter_names = [generator.vocs.variable_names[i]
                              for i in optimization_indeces]
    x_tuning_dict = {}
    for i in range(x_tuning.shape[1]):
        x_tuning_dict[tuning_parameter_names[i]] = float(x_tuning[0,i])
    return x_tuning_dict

def uniform_random_sample_in_bounds(n_samples, bounds):
    ndim = len(bounds.T)

    # uniform sample, rescaled, and shifted to cover the domain
    x_samples = torch.rand(n_samples, ndim) * torch.tensor(
        [bounds_i[1] - bounds_i[0] for bounds_i in bounds.T]
    ) + torch.tensor([bounds_i[0] for bounds_i in bounds.T])

    return x_samples