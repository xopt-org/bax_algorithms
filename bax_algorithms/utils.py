from xopt.base import Xopt
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from bax_algorithms.pathwise.optimize import VirtualOptimizer


def get_bax_model_and_bounds(X: Xopt):
    bax_model_ids = [
        X.generator.vocs.output_names.index(name)
        for name in X.generator.algorithm.observable_names_ordered
    ]
    model = X.generator.train_model()
    bax_model = model.subset_output(bax_model_ids)

    if isinstance(bax_model, SingleTaskGP):
        bax_model = ModelListGP(bax_model)

    return bax_model, X.generator._get_optimization_bounds()

def get_bax_mean_prediction(X: Xopt, mean_optimizer: VirtualOptimizer):
    model, bounds = get_bax_model_and_bounds(X)
    def get_mean_function(model):
        def func(x):
            gp_mean = model.posterior(x).mean
            return gp_mean
        return func
    mean_functions_list = [get_mean_function(m) for m in model.models]

    optimization_indeces = X.generator.algorithm._get_optimization_indeces(bounds)

    # optimize mean functions
    best_inputs = mean_optimizer.optimize(virtual_objective = X.generator.algorithm.evaluate_virtual_objective,
                                          sample_functions_list = mean_functions_list,
                                          bounds = bounds,
                                          optimization_indeces = optimization_indeces,
                                          n_samples = 1)

    return best_inputs.squeeze(0)