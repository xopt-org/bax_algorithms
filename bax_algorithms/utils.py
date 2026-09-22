from collections.abc import Callable
import math

from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.generators.bayesian.objectives import create_constraint_callables
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
    best_inputs = mean_optimizer.optimize(
        virtual_objective=generator.algorithm.evaluate_virtual_objective,
        sample_functions_list=mean_functions_list,
        bounds=bounds,
        optimization_indeces=optimization_indeces,
        n_samples=1,
    )

    return best_inputs.squeeze(0)


def get_constrained_mean_prediction(
    generator: BaxGenerator,
    mean_optimizer: VirtualOptimizer,
    *,
    kappa: float = 0.0,
    constraint_points: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    | None = None,
):
    """Optimize the BAX mean objective subject to all VOCS output constraints.

    Uses Xopt's constraint callables, including output ordering and LESS_THAN /
    GREATER_THAN directions. kappa=0 requires posterior-mean feasibility. Positive
    kappa requires the signed constraint mean plus kappa times its posterior
    standard deviation to be <= 0. This is a pointwise uncertainty margin, not
    a guarantee of physical safety or joint feasibility.

    By default, constraints are evaluated at the full candidate inputs passed to
    the virtual objective. Dimensions not optimized by the algorithm are filled
    with the midpoint of their bounds, as in get_bax_mean_prediction. To enforce
    constraints at other inputs, provide constraint_points(x, bounds), returning
    shape (*x.shape[:-1], n_evaluation_points, x.shape[-1]). All evaluation points
    must satisfy every constraint. This callback can also select a reference
    setting for dimensions not optimized by the algorithm.

    The optimizer must implement the optional constraints callback in
    VirtualOptimizer.optimize (implemented by DifferentialEvolution). Without
    VOCS constraints this delegates to get_bax_mean_prediction unchanged.
    Returns shape (1, n_tuning_dims), in the algorithm's optimization order.
    Like the original helper, this expects bounds covering all model inputs;
    reduced bounds from fixed_features are not supported.
    """
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    if not generator.vocs.constraints:
        return get_bax_mean_prediction(generator, mean_optimizer)

    # Retain the constraint models separately from the objective observables.
    model = generator.train_model()
    bounds = generator._get_optimization_bounds()
    if bounds.shape[-1] != len(generator.model_input_names):
        raise ValueError("Mean prediction requires bounds covering all model inputs")
    output_names = generator.vocs.output_names
    observable_models = [
        model.subset_output([output_names.index(name)])
        for name in generator.algorithm.observable_names_ordered
    ]
    mean_functions = [(lambda x, m=m: m.posterior(x).mean) for m in observable_models]
    constraint_functions = create_constraint_callables(generator.vocs)
    constraint_indices = [
        output_names.index(name) for name in generator.vocs.constraint_names
    ]

    @torch.no_grad()
    def constraints(x):
        points = (
            x.unsqueeze(-2)
            if constraint_points is None
            else constraint_points(x, bounds)
        )
        if (
            points.ndim != x.ndim + 1
            or points.shape[:-2] != x.shape[:-1]
            or points.shape[-1] != x.shape[-1]
            or points.shape[-2] == 0
        ):
            raise ValueError(
                "constraint_points must return (*x.shape[:-1], n_points, n_dims)"
            )
        posterior = model.posterior(points)
        mean = posterior.mean
        residuals = torch.stack([fn(mean) for fn in constraint_functions], dim=-1)
        if kappa:
            std = posterior.variance[..., constraint_indices].clamp_min(0).sqrt()
            residuals = residuals + kappa * std
        residuals = torch.where(torch.isfinite(residuals), residuals, torch.inf)
        return residuals.amax(dim=-2)

    best_inputs = mean_optimizer.optimize(
        virtual_objective=generator.algorithm.evaluate_virtual_objective,
        sample_functions_list=mean_functions,
        bounds=bounds,
        optimization_indeces=generator.algorithm._get_optimization_indeces(bounds),
        n_samples=1,
        constraints=constraints,
    )
    return best_inputs.squeeze(0)


def tuning_input_tensor_to_dict(generator, x_tuning):
    """
    Converts a single set of tuning parameters to a dictionary for input to Xopt

    x_tuning = tensor of shape (1, n_tuning_dims)
    """
    _, bounds = get_bax_model_and_bounds(generator)
    optimization_indeces = generator.algorithm._get_optimization_indeces(bounds)
    tuning_parameter_names = [
        generator.vocs.variable_names[i] for i in optimization_indeces
    ]
    x_tuning_dict = {}
    for i in range(x_tuning.shape[1]):
        x_tuning_dict[tuning_parameter_names[i]] = float(x_tuning[0, i])
    return x_tuning_dict


def uniform_random_sample_in_bounds(n_samples, bounds):
    ndim = len(bounds.T)

    # uniform sample, rescaled, and shifted to cover the domain
    x_samples = torch.rand(n_samples, ndim) * torch.tensor(
        [bounds_i[1] - bounds_i[0] for bounds_i in bounds.T]
    ) + torch.tensor([bounds_i[0] for bounds_i in bounds.T])

    return x_samples
