from pydantic import Field, PositiveInt, field_validator, field_serializer
from typing import List, Optional, Any
import ast
import torch
from torch import Tensor

from gpytorch.kernels.rbf_kernel import RBFKernel
from bax_algorithms.pathwise.base import PathwiseOptimization
from bax_algorithms.pathwise.sampling import draw_product_kernel_post_paths
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
from botorch.models.model import Model
from xopt.generators.bayesian.bax.algorithms import (
    Algorithm,
    OptimizationAlgorithmResult,
    VirtualMeasurementResult,
)

import numpy as np
from scipy.optimize import minimize
import importlib.util


def bmag_func(bb, ab, bl, al):
    """
    TODO: replace function with import from slac-tools upon release.
    SOURCE:
    https://github.com/slaclab/lcls-tools/blob/main/lcls_tools/common/data/model_general_calcs.py

    Calculates the BMAG miss match parameter.  bb and ab are the modeled
    beta and alpha functions at a given element and bl and al are the
    reference (most of the time desing) values"""
    return 1 / 2 * (bl / bb + bb / bl + bb * bl * (ab / bb - al / bl) ** 2)


def propagate_twiss(twiss_init: np.ndarray, rmat: np.ndarray):
    """
    TODO: replace function with import from slac-tools upon release.
    SOURCE:
    https://github.com/slaclab/lcls-tools/blob/main/lcls_tools/common/data/model_general_calcs.py

    Propagates twiss parameters downstream given a transport rmat.

    Parameters:
        twiss_init: numpy array shape batchshape x 3 containing the initial twiss params
                    (ordered: beta, alpha, gamma)
        rmat: numpy array shape batchshape x 2 x 2 containing 2x2 transport rmats

    Outputs:
        twiss_final: numpy array shape batchshape x 3 containing the downstream twiss params
                    (ordered: beta, alpha, gamma)
    """
    twiss_transport = twiss_transport_mat_from_rmat(
        rmat
    )  # result shape (batchshape x 3 x 3)

    twiss_final = twiss_transport @ np.expand_dims(twiss_init, axis=-1)
    # result shape (batchshape x 3 x 1)

    return twiss_final.squeeze(-1)


def bdes_to_kmod(e_tot, effective_length, bdes):
    """
    TODO: replace function with import from slac-tools upon release.
    SOURCE:
    https://github.com/slaclab/lcls-tools/blob/main/lcls_tools/common/data/model_general_calcs.py

    Returns K in 1/m^2 given BDES
    Need to privide either particle energy e_tot and quad effective_length
    or element name and tao object"""
    bp = e_tot / 1e9 / 299.792458 * 1e4  # kG m
    return bdes / effective_length / bp  # kG / m / kG m = 1/m^2


def build_quad_rmat(k: np.ndarray, q_len: float, thin_lens: bool = False):
    """
    TODO: replace function with import from slac-tools upon release.
    SOURCE:
    https://github.com/slaclab/lcls-tools/blob/main/lcls_tools/common/data/model_general_calcs.py

    Constructs quad rmat transport matrices for a quadrupole of length q_len with geometric focusing strengths
    given by k.

    Parameters:
        k: numpy array containing geometric focusing strengths
        q_len: float specifying quad length in meters
        thin_lens: boolean specifying whether or not to use thin-lens approximation

    Outputs:
        rmat: numpy array of shape (*k.shape, 2, 2) containing rmats corresponding to the
                various focusing strengths given by k

    source: https://uspas.fnal.gov/materials/11ODU/Lecture6_Transverse_Beam_Optics_1.pdf
    """

    if not thin_lens:
        sqrt_k = np.sqrt(np.abs(k)) + 1.0e-6  # add small value for numerical stability

        c = (
            np.cos(sqrt_k * q_len) * (k > 0)
            + np.cosh(sqrt_k * q_len) * (k < 0)
            + np.ones_like(k) * (k == 0)
        )
        s = (
            np.nan_to_num(1.0 / sqrt_k) * np.sin(sqrt_k * q_len) * (k > 0)
            + np.nan_to_num(1.0 / sqrt_k) * np.sinh(sqrt_k * q_len) * (k < 0)
            + q_len * np.ones_like(k) * (k == 0)
        )
        cp = (
            -sqrt_k * np.sin(sqrt_k * q_len) * (k > 0)
            + sqrt_k * np.sinh(sqrt_k * q_len) * (k < 0)
            + np.zeros_like(k) * (k == 0)
        )
        sp = (
            np.cos(sqrt_k * q_len) * (k > 0)
            + np.cosh(sqrt_k * q_len) * (k < 0)
            + np.ones_like(k) * (k == 0)
        )

    else:
        c, s, cp, sp = (np.ones_like(k), np.zeros_like(k), -k * q_len, np.ones_like(k))

    rmat = np.stack(
        (
            np.stack((c, s), axis=-1),
            np.stack((cp, sp), axis=-1),
        ),
        axis=-2,
    )  # final shape (*k.shape, 2, 2)

    return rmat


def twiss_transport_mat_from_rmat(rmat: np.ndarray):
    """
    TODO: replace function with import from slac-tools upon release.
    SOURCE:
    https://github.com/slaclab/lcls-tools/blob/main/lcls_tools/common/data/model_general_calcs.py

    Converts from 2x2 rmats to 3x3 twiss transport matrices.

    Parameters:
        rmat: numpy array shape batchshape x 2 x 2

    Outputs:
        result: numpy array shape batchshape x 3 x 3
    """
    # Converts from 2x2 rmats to 3x3 twiss transport matrices.
    c, s, cp, sp = rmat[..., 0, 0], rmat[..., 0, 1], rmat[..., 1, 0], rmat[..., 1, 1]
    result = np.stack(
        (
            np.stack((c**2, -2 * c * s, s**2), axis=-1),
            np.stack((-c * cp, c * sp + cp * s, -s * sp), axis=-1),
            np.stack((cp**2, -2 * cp * sp, sp**2), axis=-1),
        ),
        axis=-2,
    )  # result shape (batchshape, 3, 3)
    return result


def compute_emit_bmag(
    beamsize_squared: np.ndarray,
    rmat: np.ndarray,
    twiss_design: np.ndarray = None,
    maxiter: int = None,
):
    """
    TODO: replace function with import from slac-tools upon release.
    SOURCE:
    https://github.com/slaclab/lcls-tools/blob/main/lcls_tools/common/data/emittance.py

    Computes the emittance(s) from a set of beamsize measurements and their corresponding
    transport matrices (rmats).
    Must provide beamsize measurements corresponding to at least 3 unique rmats (e.g. quad scan
    with minimum of 3 steps, or 3-wire scan).
    Uses nonlinear fitting of beam matrix parameters to guarantee physically valid results.


    Parameters
    ----------
    beamsize_squared : numpy.ndarray
        Array of shape (batchshape x n_measurements x 1), representing the mean-square
        beamsize outputs in [mm^2].

    rmat : numpy.ndarray
        Array of shape (n_measurements x 2 x 2) or (batchshape x n_measurements x 2 x 2)
        containing the 2x2 R matrices describing the transport from a common upstream
        point in the beamline to the locations at which each beamsize was observed.

    twiss_design : numpy.ndarray, optional
        Array of shape (batchshape x n_measurements x 2) designating the design (beta, alpha)
        twiss parameters at each measurement location.
        Note that it is also possible to pass an array of shape (batchshape x 1 x 2),
        which will result in broadcasting a single set of design twiss parameters
        to each measurement in the respective batch for the calculation of Bmag
        (useful for quad scans).

    maxiter : int, optional
        Maximum number of iterations to perform in nonlinear fitting (minimization algorithm).

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - 'emittance': numpy.ndarray of shape (batchshape x 1) containing the geometric emittance
          fit results for each scan in mm-mrad.
        - 'bmag': numpy.ndarray of shape (batchshape x n_steps) containing the bmag corresponding
          to each point in each scan.
        - 'beam_matrix': numpy.ndarray of shape (batchshape x 3) containing [sig11, sig12, sig22]
          where sig11, sig12, sig22 are the reconstructed beam matrix parameters at the entrance
          of the measurement quad.
        - 'twiss_at_screen': numpy.ndarray of shape (batchshape x nsteps x 3) containing the
          reconstructed twiss parameters at the measurement screen for each step in each quad scan.

    References
    ----------
    SOURCE PAPER: http://www-library.desy.de/preparch/desy/thesis/desy-thesis-05-014.pdf
    """
    # return variable dictionary
    rv = {}

    # prepare the A matrix from eq. (3.2) & (3.3) of source paper
    r11, r12 = rmat[..., 0, 0], rmat[..., 0, 1]
    amat = np.stack((r11**2, 2.0 * r11 * r12, r12**2), axis=-1)
    # amat result (batchshape x nsteps x 3)

    def beam_matrix_tuple(params):
        """
        converts fit parameters (batchshape x 3), containing [lambda1, lambda2, c],
        to tuple of beam matrix parameters (sig11, sig12, sig22) where each
        element in the tuple is shape batchshape, for stacking.
        """
        return (
            params[..., 0] ** 2,  # lamba1^2 = sig11
            params[..., 0]
            * params[..., 1]
            * params[..., 2],  # lambda1*lambda2*c = sig12
            params[..., 1] ** 2,  # lamba2^2 = sig22
        )

    # check if torch is available to be imported
    torch_spec = importlib.util.find_spec("torch")
    torch_found = torch_spec is not None
    if torch_found:
        # define loss function in torch and use autograd to get its jacobian
        import torch

        amat = torch.from_numpy(amat)
        beamsize_squared = torch.from_numpy(beamsize_squared)

        def loss_torch(params):
            params = torch.reshape(params, [*beamsize_squared.shape[:-2], 3])
            sig = torch.stack(beam_matrix_tuple(params), dim=-1).unsqueeze(-1)
            # sig should now be shape batchshape x 3 x 1 (column vectors)
            total_abs_error = (
                (torch.sqrt(amat @ sig) - torch.sqrt(beamsize_squared)).abs().nansum()
            )
            return total_abs_error

        def loss_jacobian(params):
            return (
                torch.autograd.functional.jacobian(loss_torch, torch.from_numpy(params))
                .detach()
                .numpy()
            )

        def loss(params):
            return loss_torch(torch.from_numpy(params)).detach().numpy()

    else:
        # define loss function in numpy without jacobian
        def loss(params):
            params = np.reshape(params, [*beamsize_squared.shape[:-2], 3])
            sig = np.expand_dims(np.stack(beam_matrix_tuple(params), axis=-1), axis=-1)
            # sig should now be shape batchshape x 3 x 1 (column vectors)
            total_abs_error = np.nansum(
                np.abs(np.sqrt(amat @ sig) - np.sqrt(beamsize_squared))
            )
            return total_abs_error

        loss_jacobian = None

    # for numerical stability
    eps = 1.0e-6

    # get initial guesses for lambda1, lambda2, c, from pseudo-inverse method
    init_beam_matrix = np.linalg.pinv(np.array(amat)) @ np.array(beamsize_squared)
    lambda1 = np.sqrt(init_beam_matrix[..., 0, 0].clip(min=eps))
    lambda2 = np.sqrt(init_beam_matrix[..., 2, 0].clip(min=eps))
    c = (init_beam_matrix[..., 1, 0] / (lambda1 * lambda2)).clip(
        min=-1 + eps, max=1 - eps
    )
    init_params = np.stack((lambda1, lambda2, c), axis=-1).flatten()

    # define bounds (only c parameter is bounded, between -1 and 1)
    bounds = np.tile(
        np.array([[None, None], [None, None], [-1.0 + eps, 1.0 - eps]]),
        (np.prod(beamsize_squared.shape[:-2]), 1),
    )
    if maxiter is not None:
        options = {"maxiter": maxiter}
    else:
        options = None

    # minimize loss
    res = minimize(
        loss,
        init_params,
        jac=loss_jacobian,
        bounds=bounds,
        options=options,
    )

    # get the fit result and reshape to (batchshape x 3)
    fit_params = np.reshape(res.x, [*beamsize_squared.shape[:-2], 3])

    # convert fit params back to beam matrix params
    rv["beam_matrix"] = np.stack(beam_matrix_tuple(fit_params), axis=-1)
    # result shape (batchshape x 3) containing [sig11, sig12, sig22]

    rv["emittance"] = np.sqrt(
        rv["beam_matrix"][..., 0:1] * rv["beam_matrix"][..., 2:3]
        - rv["beam_matrix"][..., 1:2] ** 2
    )
    # result shape (batchshape x 1)

    # get twiss at upstream origin from beam_matrix
    def _twiss_upstream(b_matrix):
        return np.expand_dims(
            np.stack(
                (
                    b_matrix[..., 0],
                    -1 * b_matrix[..., 1],
                    b_matrix[..., 2],
                ),
                axis=-1,
            )
            / rv["emittance"],
            axis=-2,
        )

    # propagate twiss params to screen (expand_dims for broadcasting)
    rv["twiss_at_screen"] = propagate_twiss(_twiss_upstream(rv["beam_matrix"]), rmat)
    # result shape (batchshape x nsteps x 3)
    beta, alpha = (
        rv["twiss_at_screen"][..., 0],
        rv["twiss_at_screen"][..., 1],
    )
    # shapes batchshape x nsteps

    # compute bmag if twiss_design is provided
    if twiss_design is not None:
        beta_design, alpha_design = (
            twiss_design[..., 0],
            twiss_design[..., 1],
        )
        # shape batchshape x nsteps x 1 (multi-device) or batchshape x 1 (quad scan)

        # result batchshape x 3 containing [beta, alpha, gamma]
        rv["bmag"] = bmag_func(
            beta, alpha, beta_design, alpha_design
        )  # result batchshape x nsteps
    else:
        rv["bmag"] = None

    return rv


def compute_emit_bmag_quad_scan(
    k: np.ndarray,
    beamsize_squared: np.ndarray,
    q_len: float,
    rmat: np.ndarray,
    twiss_design: np.ndarray = None,
    thin_lens: bool = False,
    maxiter: int = None,
):
    """
    TODO: replace function with import from slac-tools upon release.
    SOURCE:
    https://github.com/slaclab/lcls-tools/blob/main/lcls_tools/common/measurements/emittance_measurement.py

    Computes the emittance(s) corresponding to a set of quadrupole measurement scans
    using nonlinear fitting of beam matrix parameters to guarantee physically valid results.

    Parameters
    ----------
    k : numpy.ndarray
        Array of shape (n_steps_quad_scan,) or (batchshape x n_steps_quad_scan)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan(s).

    beamsize_squared : numpy.ndarray
        Array of shape (batchshape x n_steps_quad_scan), representing the mean-square
        beamsize outputs in [mm^2] of the emittance scan(s) with inputs given by k.

    q_len : float
        The (longitudinal) quadrupole length or "thickness" in [m].

    rmat : numpy.ndarray
        Array of shape (2x2) or (batchshape x 2 x 2) containing the 2x2 R matrices
        describing the transport from the end of the measurement quad to the observation screen.

    twiss_design : numpy.ndarray, optional
        Array of shape (batchshape x 2) designating the design (beta, alpha)
        twiss parameters at the screen.

    thin_lens : bool, optional
        Specifies whether or not to use thin lens approximation for measurement quad.

    maxiter : int, optional
        Maximum number of iterations to perform in nonlinear fitting (minimization algorithm).

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - 'emittance': numpy.ndarray of shape (batchshape x 1) containing the geometric emittance
          fit results for each scan in mm-mrad.
        - 'bmag': numpy.ndarray of shape (batchshape x n_steps) containing the bmag corresponding
          to each point in each scan.
        - 'beam_matrix': numpy.ndarray of shape (batchshape x 3) containing [sig11, sig12, sig22]
          where sig11, sig12, sig22 are the reconstructed beam matrix parameters at the entrance
          of the measurement quad.
        - 'twiss_at_screen': numpy.ndarray of shape (batchshape x nsteps x 3) containing the
          reconstructed twiss parameters at the measurement screen for each step in each quad scan.
    """
    # calculate and add the measurement quad transport to the rmats
    quad_rmat = build_quad_rmat(
        k, q_len, thin_lens=thin_lens
    )  # result shape (batchshape x nsteps x 2 x 2)
    total_rmat = np.expand_dims(rmat, -3) @ quad_rmat
    # result shape (batchshape x nsteps x 2 x 2)

    # reshape inputs
    beamsize_squared = np.expand_dims(beamsize_squared, -1)
    twiss_design = (
        np.expand_dims(twiss_design, -2) if twiss_design is not None else None
    )

    # compute emittance
    rv = compute_emit_bmag(beamsize_squared, total_rmat, twiss_design, maxiter)

    return rv


class VirtualEmittanceMeasurementResult(VirtualMeasurementResult):
    emittance_x: Optional[Tensor] = Field(
        default=None,
        description="The geometric emittance in the x transverse dimension.",
    )
    emittance_y: Optional[Tensor] = Field(
        default=None,
        description="The geometric emittance in the y transverse dimension.",
    )
    bmag_x: Optional[Tensor] = Field(
        default=None, description="The Bmag in the x transverse dimension."
    )
    bmag_y: Optional[Tensor] = Field(
        default=None, description="The Bmag in the y transverse dimension."
    )


class EmittanceAlgorithm(Algorithm):
    name: str = Field("minimize_emittance", frozen=True)
    x_key: str | None = Field(
        None,
        description="key designating the beamsize squared output in x from evaluate function",
    )

    y_key: str | None = Field(
        None,
        description="key designating the beamsize squared output in y from evaluate function",
    )
    energy: float = Field(1.0, description="Beam energy in [eV]")
    q_len: float = Field(
        0.08, description="the longitudinal thickness of the measurement quadrupole"
    )
    rmat_x: Tensor | None = Field(
        Tensor([[1.0, 1.0], [0.0, 1.0]]),
        description="2x2 Tensor containing downstream rmat for x dimension",
    )
    rmat_y: Tensor | None = Field(
        Tensor([[1.0, 1.0], [0.0, 1.0]]),
        description="2x2 Tensor containing downstream rmat for y dimension",
    )
    twiss0_x: Tensor | None = Field(
        Tensor([1.0, 0.0]),
        description="List length 2 containing design x-twiss: [beta0_x, alpha0_x] (for bmag)",
    )
    twiss0_y: Tensor | None = Field(
        Tensor([1.0, 0.0]),
        description="List length 2 containing design y-twiss: [beta0_y, alpha0_y] (for bmag)",
    )
    meas_dim: int = Field(
        0,
        description="index identifying the measurement quad dimension in the model",
    )
    n_steps_measurement_param: int = Field(
        5, description="number of steps to use in the virtual measurement scans"
    )
    thin_lens: bool = Field(
        False,
        description="Whether to use thin-lens approximation in transport for emittance calc",
    )
    use_bmag: bool = Field(
        True,
        description="Whether to multiply the emit by the bmag to get virtual objective.",
    )
    maxiter_fit: int = Field(
        20, description="Maximum number of iterations in nonlinear emittance fitting."
    )
    crop_scans: bool = Field(
        False,
        description="Whether to retain beamsize values only around the minimum from each scan.",
    )

    @field_validator("rmat_x", "rmat_y", "twiss0_x", "twiss0_y", mode="before")
    @classmethod
    def validate_tensors(cls, v: Any) -> Tensor:
        """Accept tensors, (possibly nested) lists/tuples, or their string
        representations (e.g. "[[1.0, 1.0], [0.0, 1.0]]" or "1.0, 0.0") and
        convert them into a double-precision tensor."""
        if isinstance(v, Tensor):
            return v
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("[") or stripped.startswith("("):
                v = ast.literal_eval(stripped)
            else:
                v = [item for item in stripped.split(",")]
        if isinstance(v, (list, tuple)):
            float_list = cls._to_nested_floats(v)
            return torch.tensor(float_list, dtype=torch.double)
        raise ValueError(f"Cannot convert {v} to a Tensor.")

    @classmethod
    def _to_nested_floats(cls, v: Any) -> Any:
        """Recursively convert a (possibly nested) list/tuple to floats,
        preserving the nesting structure."""
        if isinstance(v, (list, tuple)):
            return [cls._to_nested_floats(item) for item in v]
        return float(v)

    @field_serializer("rmat_x", "rmat_y", "twiss0_x", "twiss0_y")
    def serialize_tensor(self, v: Any) -> Any:
        """Serialize tensor fields to nested lists so they round-trip through
        JSON/YAML (pydantic otherwise dumps unknown types as ``'torch.Tensor'``)."""
        if isinstance(v, Tensor):
            return v.tolist()
        return v

    @property
    def x_idx(self) -> int:
        """
        The index of the x-beamsize model in the BAX observable ModelList passed
        to self.get_execution_paths() by Xopt's BaxGenerator.
        """
        return self.observable_names_ordered.index(self.x_key)

    @property
    def y_idx(self) -> int:
        """
        The index of the y-beamsize model in the BAX observable ModelList passed
        to self.get_execution_paths() by Xopt's BaxGenerator.
        """
        return self.observable_names_ordered.index(self.y_key)

    def perform_virtual_measurement(
        self, model, x, bounds, tkwargs: dict = None, n_samples: int = None
    ):
        """
        inputs:
            model: a botorch ModelListGP
            x: tensor shape (n_points, n_dim) or (n_samples, n_points, n_dim)
                    specifying points in the full-dimensional model space
                    at which to evaluate the objective.
            bounds: tensor shape (2, n_dim) specifying the upper and lower measurement bounds
        returns:
            VirtualEmittanceMeasurementResult
        """
        tuning_idxs = torch.arange(bounds.shape[1])
        tuning_idxs = tuning_idxs[
            tuning_idxs != self.meas_dim
        ]  # remove measurement dim index
        x_tuning = x[..., tuning_idxs]

        # x_tuning must be shape n_tuning_configs x n_tuning_dims
        emit, bmag = self.evaluate_posterior_emittance(
            model,
            x_tuning,
            bounds,
            tkwargs,
            n_samples,
        )

        # store virtual measurement results
        result = {
            "emittance_x": None,
            "emittance_y": None,
            "bmag_x": None,
            "bmag_y": None,
        }
        if self.x_key:
            result["emittance_x"] = emit[..., self.x_idx]
            best_bmag_x = torch.min(bmag[..., self.x_idx], dim=-1, keepdim=True)[0]
            result["bmag_x"] = best_bmag_x
            objective = result["emittance_x"]
            mean_bmag = result["bmag_x"]
        if self.y_key:
            result["emittance_y"] = emit[..., self.y_idx]
            best_bmag_y = torch.min(bmag[..., self.y_idx], dim=-1, keepdim=True)[0]
            result["bmag_y"] = best_bmag_y
            objective = result["emittance_y"]
            mean_bmag = result["bmag_y"]
        if self.x_key and self.y_key:
            objective = (result["emittance_x"] * result["emittance_y"]).sqrt()
            best_bmag_idcs = torch.min(
                (bmag[..., self.x_idx] * bmag[..., self.y_idx]), dim=-1, keepdim=True
            )[1]
            best_bmag_x = torch.gather(bmag[..., self.x_idx], -1, best_bmag_idcs)
            best_bmag_y = torch.gather(bmag[..., self.y_idx], -1, best_bmag_idcs)
            result["bmag_x"] = best_bmag_x
            result["bmag_y"] = best_bmag_y
            mean_bmag = (result["bmag_x"] * result["bmag_y"]).sqrt()
        if self.use_bmag:
            objective *= mean_bmag

        algorithm_result = VirtualEmittanceMeasurementResult(
            objective=objective,
            emittance_x=result["emittance_x"],
            emittance_y=result["emittance_y"],
            bmag_x=result["bmag_x"],
            bmag_y=result["bmag_y"],
        )
        return algorithm_result

    def get_meas_scan_inputs(
        self, x_tuning: Tensor, bounds: Tensor, tkwargs: dict = None
    ):
        """
        A function that generates the inputs for virtual emittance measurement scans at the tuning
        configurations specified by x_tuning.

        Parameters:
            x_tuning: a tensor of shape n_points x n_tuning_dims, where each row specifies a tuning
                        configuration where we want to do an emittance scan.
                        >>batchshape x n_tuning_configs x n_tuning_dims (ex: batchshape = n_samples x n_tuning_configs)
        Returns:
            xs: tensor, shape (n_tuning_configs*n_steps_meas_scan) x d,
                where n_tuning_configs = x_tuning.shape[0],
                n_steps_meas_scan = len(x_meas),
                and d = x_tuning.shape[1] -- the number of tuning parameters
                >>batchshape x n_tuning_configs*n_steps x ndim
        """
        # each row of x_tuning defines a location in the tuning parameter space
        # along which to perform a quad scan and evaluate emit

        # expand the x tensor to represent quad measurement scans
        # at the locations in tuning parameter space specified by X
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}

        x_meas = torch.linspace(
            *bounds.T[self.meas_dim], self.n_steps_measurement_param, **tkwargs
        )

        # prepare column of measurement scans coordinates
        x_meas_expanded = x_meas.reshape(-1, 1).repeat(*x_tuning.shape[:-1], 1)

        # repeat tuning configs as necessary and concat with column from the line above
        # to make xs shape: (n_tuning_configs*n_steps_quad_scan) x d ,
        # where d is the full dimension of the model/posterior space (tuning & meas)
        x_tuning_expanded = torch.repeat_interleave(
            x_tuning, self.n_steps_measurement_param, dim=-2
        )

        x = torch.cat(
            (
                x_tuning_expanded[..., : self.meas_dim],
                x_meas_expanded,
                x_tuning_expanded[..., self.meas_dim :],
            ),
            dim=-1,
        )

        return x

    def evaluate_posterior_emittance(
        self, model, x_tuning, bounds, tkwargs: dict = None, n_samples: int = None
    ):
        """
        inputs:
            x_tuning: tensor shape n_points x (n_dim-1) specifying points in the **tuning** space
                    at which to evaluate the objective.
        returns:
            emit: tensor shape n_points x 1 or 2
            bmag: tensor shape n_points x 1 or 2
        """
        assert len(x_tuning.shape) in [2, 3]
        # x_tuning must be shape (n_tuning_configs, n_tuning_dims) or (n_samples, n_tuning_configs, ndim)
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}

        x = self.get_meas_scan_inputs(
            x_tuning, bounds, tkwargs
        )  # result shape n_tuning_configs*n_steps x ndim
        bss = self.evaluate_virtual_observables(model, x, n_samples)

        # package inputs for emittance calculation

        bss = bss.reshape(
            -1, x_tuning.shape[-2], self.n_steps_measurement_param, bss.shape[-1]
        )
        # bss.shape = (n_samples, x_tuning.shape[-2], self.n_steps_measurement_param, 1 or 2)
        x = x.reshape(
            -1, x_tuning.shape[-2], self.n_steps_measurement_param, x.shape[-1]
        )
        # x.shape = (n_samples, x_tuning.shape[-2], self.n_steps_measurement_param, ndim)
        if len(x_tuning.shape) == 2:
            x = x.repeat(bss.shape[0], 1, 1, 1)

        if self.x_key and not self.y_key:
            k = bdes_to_kmod(self.energy, self.q_len, x[..., self.meas_dim])
            beamsize_squared = bss[
                ..., self.x_idx
            ]  # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_x.to(**tkwargs).repeat(
                *bss.shape[:2], 1, 1
            )  # n_samples x n_tuning x 2 x 2
            twiss0 = self.twiss0_x.repeat(*bss.shape[:2], 1)
        elif self.y_key and not self.x_key:
            k = -1 * bdes_to_kmod(self.energy, self.q_len, x[..., self.meas_dim])
            beamsize_squared = bss[
                ..., self.y_idx
            ]  # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_y.to(**tkwargs).repeat(
                *bss.shape[:2], 1, 1
            )  # n_samples x n_tuning x 2 x 2
            twiss0 = self.twiss0_y.repeat(*bss.shape[:2], 1)
        else:
            k_x = bdes_to_kmod(self.energy, self.q_len, x[..., self.meas_dim])
            k_y = k_x * -1.0  # n_samples x n_tuning x n_steps
            k = torch.cat((k_x, k_y))  # shape (2*n_samples x n_tuning x n_steps)

            beamsize_squared = torch.cat((bss[..., self.x_idx], bss[..., self.y_idx]))

            rmat_x = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2], 1, 1)
            rmat_y = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2], 1, 1)
            rmat = torch.cat((rmat_x, rmat_y))  # shape (2*n_samples x n_tuning x 2 x 2)

            twiss0 = torch.cat(
                (
                    self.twiss0_x.repeat(*bss.shape[:2], 1),
                    self.twiss0_y.repeat(*bss.shape[:2], 1),
                )
            )

        if self.crop_scans:
            beamsize_squared = self._crop_quad_scans(beamsize_squared)

        # compute emittance
        rv = compute_emit_bmag_quad_scan(
            k.numpy(),
            beamsize_squared.detach().numpy(),
            self.q_len,
            rmat.numpy(),
            twiss0.numpy(),
            thin_lens=self.thin_lens,
            maxiter=self.maxiter_fit,
        )

        emit = torch.from_numpy(rv["emittance"])
        bmag = torch.from_numpy(rv["bmag"])
        # emit.shape = (n_samples x n_tuning) or (2*n_samples x n_tuning) if optimizing both x and y
        # bmag.shape = (n_samples x n_tuning x nsteps) or (2*n_samples x n_tuning x nsteps) if optimizing both x and y

        if self.x_key and self.y_key:
            emit = torch.cat(
                (
                    emit[: bss.shape[0]].unsqueeze(-1),
                    emit[bss.shape[0] :].unsqueeze(-1),
                ),
                dim=-1,
            )
            # emit.shape = (n_samples, n_tuning, 1, 2)
            bmag = torch.cat(
                (
                    bmag[: bss.shape[0]].unsqueeze(-1),
                    bmag[bss.shape[0] :].unsqueeze(-1),
                ),
                dim=-1,
            )
            # bmag.shape = (n_samples, n_tuning, n_steps, 2)
        else:
            emit = emit.unsqueeze(-1)
            bmag = bmag.unsqueeze(-1)
        # final shapes: n_samples x n_tuning_configs (?? NEED TO CHECK THIS, don't think it's correct)

        return emit, bmag

    def _crop_quad_scans(
        self, beamsize_squared: Tensor, n_neighbors: int = 2
    ) -> Tensor:
        min_values, min_indices = torch.min(beamsize_squared, dim=-1, keepdim=True)
        row_indices = torch.arange(beamsize_squared.shape[-1]).repeat(
            *beamsize_squared.shape[:-1], 1
        )
        mask_first_min = row_indices == min_indices

        # add each minimum's nearest neighbors to mask
        x = mask_first_min.detach().clone()
        for i in range(n_neighbors):
            # Create shifted copies
            x_left = torch.cat(
                (x[..., 1:], torch.zeros(*x.shape[:-1], 1, dtype=torch.bool)), dim=-1
            )  # Shift left, add False at the end
            x_right = torch.cat(
                (torch.zeros(*x.shape[:-1], 1, dtype=torch.bool), x[..., :-1]), dim=-1
            )  # Shift right, add False at the beginning

            # Perform boolean OR operation
            x = x | x_left | x_right

        beamsize_squared_cropped = beamsize_squared.clone()
        beamsize_squared_cropped[~x] = torch.nan

        return beamsize_squared_cropped


class PathwiseMinimizeEmittance(EmittanceAlgorithm, PathwiseOptimization):
    name: str = Field("pathwise_minimize_emittance", frozen=True)
    n_batch: PositiveInt = Field(
        1,
        description="Number of sample batches to optimize, with each batch containing self.n_samples",
    )

    def execute(self, model: Model, bounds: Tensor) -> Tensor:
        best_tuning_inputs_list = []
        best_objective_list = []
        best_scan_inputs_list = []
        best_scan_outputs_list = []
        for i in range(self.n_batch):
            # draw callable sample functions
            sample_functions_list = self.draw_sample_functions_list(model)

            best_tuning_inputs = self.optimize_samples_funcs_list(
                sample_functions_list, bounds
            )
            best_meas_scan_inputs = self.get_meas_scan_inputs(
                best_tuning_inputs, bounds
            )
            best_meas_scan_outputs = torch.vstack(
                [
                    sample_func(best_meas_scan_inputs)
                    for sample_func in sample_functions_list
                ]
            ).T.unsqueeze(0)
            best_result = self.perform_virtual_measurement(
                sample_functions_list, best_meas_scan_inputs[:, :1, :], bounds
            )
            best_tuning_inputs_list += [best_tuning_inputs]
            best_objective_list += [best_result.objective]
            best_scan_inputs_list += [best_meas_scan_inputs]
            best_scan_outputs_list += [best_meas_scan_outputs]

        input_execution_paths = torch.cat(best_scan_inputs_list)
        output_execution_paths = torch.cat(best_scan_outputs_list)
        best_inputs = torch.cat(best_tuning_inputs_list)
        best_objective = torch.cat(best_objective_list)
        solution_center = best_inputs.mean(dim=0)
        solution_entropy = float(torch.log(best_inputs.std(dim=0) ** 2).sum())

        algorithm_result = OptimizationAlgorithmResult(
            best_inputs=best_inputs.detach(),
            best_objective=best_objective.detach(),
            input_execution_paths=input_execution_paths.detach(),
            output_execution_paths=output_execution_paths.detach(),
            solution_center=solution_center.detach(),
            solution_entropy=solution_entropy,
        )

        return algorithm_result

    def draw_sample_functions_list(self, model):
        sample_funcs_list = []
        for m in model.models:
            if isinstance(model.models[0].covar_module, RBFKernel):
                sample_funcs = draw_matheron_paths(
                    m, sample_shape=torch.Size([self.n_samples])
                )
            else:  # must be polynomial product kernel
                sample_funcs = draw_product_kernel_post_paths(
                    m, n_samples=self.n_samples
                )
            sample_funcs_list += [sample_funcs]
        return sample_funcs_list

    def _get_optimization_indeces(self, bounds) -> Tensor:
        """
        Get indeces specifying parameters for virtual objective optimization.
        """
        idcs = torch.tensor(range(bounds.shape[1]))
        mask = idcs != self.meas_dim
        return idcs[mask]
