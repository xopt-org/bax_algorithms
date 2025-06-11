import torch
from matplotlib import pyplot as plt
from botorch.models import ModelListGP, SingleTaskGP
from typing import List
from xopt.generators.bayesian.visualize import (
    _generate_input_mesh,
    _get_reference_point,
)
from xopt.generators.bayesian.bax_generator import BaxGenerator
from bax_algorithms.utils import get_bax_model_and_bounds


def visualize_virtual_measurement_result(
    generator: BaxGenerator,
    variable_names: list[str] = None,
    idx: int = -1,
    reference_point: dict = None,
    n_grid: int = 11,
    n_samples: int = 100,
    kwargs: dict = None,
    result_keys: List[str] = ['objective'],
) -> tuple:
    """
    Displays BAX's virtual measurement results computed from samples drawn
    from the GP model(s) of the observable(s).

    Parameters
    ----------
    generator : Generator
        Bayesian generator object.
    variable_names : List[str]
        The variables with respect to which the GP models are displayed (maximum
        of 2). Defaults to generator.vocs.variable_names.
    idx : int
        Index of the last sample to use. This also selects the point of reference in
        higher dimensions unless an explicit reference_point is given.
    reference_point : dict
        Reference point determining the value of variables in
        generator.vocs.variable_names, but not in variable_names (slice plots in
        higher dimensions). Defaults to last used sample.
    n_grid : int, optional
        Number of grid points per dimension used to display the model predictions.
    n_samples : int, optional
        Number of virtual measurement samples to evaluate for each point in the scan.
    kwargs : dict, optional
        Additional keyword arguments for evaluating the virtual measurement.

    Returns:
    --------
        The matplotlib figure and axes objects.
    """
    vocs, data = generator.vocs, generator.data
    reference_point = _get_reference_point(reference_point, vocs, data, idx)
    # define output and variable names
    if variable_names is None:
        variable_names = vocs.variable_names
    dim_x = len(variable_names)
    if dim_x not in [1, 2]:
        raise ValueError(
            f"Visualization is only supported with respect to 1 or 2 variables, "
            f"not {dim_x}. Provide a compatible list of variable names to create "
            f"slice plots at higher dimensions."
        )

    # validate variable names
    invalid = [name not in getattr(vocs, "variable_names") for name in variable_names]
    if any(invalid):
        invalid_names = [
            variable_names[i] for i in range(len(variable_names)) if invalid[i]
        ]
        raise ValueError(
            f"Variable names {invalid_names} are not in generator.vocs.variable_names."
        )

    # validate reference point keys
    invalid = [
        name not in getattr(vocs, "variable_names") for name in [*reference_point]
    ]
    if any(invalid):
        invalid_names = [
            [*reference_point][i] for i in range(len([*reference_point])) if invalid[i]
        ]
        raise ValueError(
            f"reference_point contains keys {invalid_names}, "
            f"which are not in generator.vocs.variable_names."
        )
    tkwargs = generator.tkwargs
    x = _generate_input_mesh(vocs, variable_names, reference_point, n_grid, tkwargs)

    # get bax observable models and bounds
    bax_model, bounds = get_bax_model_and_bounds(generator)

    # get virtual measurement (sample) results
    kwargs = kwargs if kwargs else {}
    measurement_result = generator.algorithm.perform_virtual_measurement(
        bax_model, x, bounds, tkwargs=tkwargs, n_samples=n_samples, **kwargs
    )

    # create figure and subplots
    figsize = (4 * dim_x, 3 * len(result_keys))

    fig, ax = plt.subplots(
        nrows=len(result_keys), ncols=dim_x, sharex=False, sharey=False, figsize=figsize
    )
    
    for i, key in enumerate(result_keys):
        
        result = measurement_result[key]
    
        # get sample stats
        result_med = result.nanmedian(dim=0)[0].flatten()
        result_upper = torch.nanquantile(result, q=0.975, dim=0).flatten()
        result_lower = torch.nanquantile(result, q=0.025, dim=0).flatten()
        result_std = (result_upper - result_lower) / 4
    
    
    
        if dim_x == 1:
            if len(result_keys) == 1:
                ax_i = ax
            else:
                ax_i = ax[i]
            # 1d line plot
            x_axis = x[:, vocs.variable_names.index(variable_names[0])].squeeze().numpy()
            ax_i.plot(x_axis, result_med, color="C0", label="Median")
            ax_i.fill_between(
                x_axis,
                result_lower,
                result_upper,
                color="C0",
                alpha=0.5,
                label="95% C.I.",
            )
            ax_i.legend()
            ax_i.set_ylabel(key)
            ax_i.set_xlabel(variable_names[0])
        else:
            # 2d heatmaps
            for j in [0, 1]:
                if len(result_keys) == 1:
                    ax_ij = ax[j]
                else:
                    ax_ij = ax[i,j]
                ax_ij.locator_params(axis="both", nbins=5)
                if j == 0:
                    prediction = result_med
                    title = key + " Median"
                    cbar_label = title
                elif j == 1:
                    prediction = result_std
                    title = fr"$\sigma\,$[{key}]"
                    cbar_label = title
    
                pcm = ax_ij.pcolormesh(
                    x[:, vocs.variable_names.index(variable_names[0])]
                    .reshape(n_grid, n_grid)
                    .numpy(),
                    x[:, vocs.variable_names.index(variable_names[1])]
                    .reshape(n_grid, n_grid)
                    .numpy(),
                    prediction.reshape(n_grid, n_grid),
                    rasterized=True,
                )
    
                from mpl_toolkits.axes_grid1 import make_axes_locatable  # lazy import
    
                divider = make_axes_locatable(ax_ij)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(pcm, cax=cax)
                ax_ij.set_title(title)
                ax_ij.set_xlabel(variable_names[0])
                ax_ij.set_ylabel(variable_names[1])
                cbar.set_label(cbar_label)

    fig.tight_layout()
    return fig, ax
