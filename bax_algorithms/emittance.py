from typing import List, Dict, Optional, Tuple, Union
from pydantic import Field
import torch
from torch import Tensor
import copy

from bax_algorithms.pathwise.base import PathwiseOptimization
from bax_algorithms.pathwise.sampling import draw_product_kernel_post_paths
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
from botorch.models.model import Model, ModelList
from gpytorch.kernels import ProductKernel, MaternKernel
from lcls_tools.common.data.emittance import compute_emit_bmag
from lcls_tools.common.data.model_general_calcs import bdes_to_kmod
from xopt.generators.bayesian.bax.algorithms import Algorithm


class EmittanceAlgorithm(Algorithm):
    x_key: str = Field(None,
        description="key designating the beamsize squared output in x from evaluate function")
    y_key: str = Field(None,
        description="key designating the beamsize squared output in y from evaluate function")
    energy: float = Field(1.0,
        description="Beam energy in [eV]")
    q_len: float = Field(
        description="the longitudinal thickness of the measurement quadrupole"
    )
    rmat_x: Tensor = Field(None,
        description="tensor shape 2x2 containing downstream rmat for x dimension"
    )
    rmat_y: Tensor = Field(None,
        description="tensor shape 2x2 containing downstream rmat for y dimension"
    )
    twiss0_x: Tensor = Field(None,
        description="1d tensor length 2 containing design x-twiss: [beta0_x, alpha0_x] (for bmag)"
    )
    twiss0_y: Tensor = Field(None,
        description="1d tensor length 2 containing design y-twiss: [beta0_y, alpha0_y] (for bmag)"
    )
    meas_dim: int = Field(None,
        description="index identifying the measurement quad dimension in the model"
    )
    n_steps_measurement_param: int = Field(3, 
        description="number of steps to use in the virtual measurement scans"
    )
    thin_lens: bool = Field(False,
        description="Whether to use thin-lens approximation in transport for emittance calc")
    use_bmag: bool = Field(True,
        description="Whether to multiply the emit by the bmag to get virtual objective.")
    results: dict = Field({},
        description="Dictionary to store results from emittance calculcation")
    maxiter_fit: int = Field(20,
        description="Maximum number of iterations in nonlinear emittance fitting.")


    @property
    def x_idx(self) -> int:
        '''
        The index of the x-beamsize model in the BAX observable ModelList passed
        to self.get_execution_paths() by Xopt's BaxGenerator.
        '''
        return self.observable_names_ordered.index(self.x_key)

    @property
    def y_idx(self) -> int:
        '''
        The index of the y-beamsize model in the BAX observable ModelList passed
        to self.get_execution_paths() by Xopt's BaxGenerator.
        '''
        return self.observable_names_ordered.index(self.y_key)

    def perform_virtual_measurement(self, model, x, bounds, tkwargs:dict=None, n_samples:int=None):
        """
        inputs:
            model: a botorch ModelListGP
            x: tensor shape (n_points, n_dim) or (n_samples, n_points, n_dim)
                    specifying points in the full-dimensional model space
                    at which to evaluate the objective.
            bounds: tensor shape (2, n_dim) specifying the upper and lower measurement bounds
        returns: 
            result: dict containing measurement results
        """
        tuning_idxs = torch.arange(bounds.shape[1])
        tuning_idxs = tuning_idxs[tuning_idxs!=self.meas_dim] # remove measurement dim index
        x_tuning = x[...,tuning_idxs]
        
        # x_tuning must be shape n_tuning_configs x n_tuning_dims
        emit, bmag = self.evaluate_posterior_emittance(model, 
                                                     x_tuning, 
                                                     bounds, 
                                                     tkwargs, 
                                                     n_samples,
                                                      )


        # store virtual measurement results
        result = {}
        if self.x_key:
            result['emit_x'] = emit[...,self.x_idx]
            best_bmag_x = torch.min(bmag[...,self.x_idx], dim=-1, keepdim=True)[0]
            result['bmag_x'] = best_bmag_x
            objective = result['emit_x']
            mean_bmag = result['bmag_x']
        if self.y_key:
            result['emit_y'] = emit[...,self.y_idx]
            best_bmag_y = torch.min(bmag[...,self.y_idx], dim=-1, keepdim=True)[0]
            result['bmag_y'] = best_bmag_y
            objective = result['emit_y']
            mean_bmag = result['bmag_y']
        if self.x_key and self.y_key:
            objective = (result['emit_x'] * result['emit_y']).sqrt()
            best_bmag_idcs = torch.min((bmag[...,self.x_idx] * bmag[...,self.y_idx]), dim=-1, keepdim=True)[1]
            best_bmag_x = torch.gather(bmag[...,self.x_idx],-1,best_bmag_idcs)
            best_bmag_y = torch.gather(bmag[...,self.y_idx],-1,best_bmag_idcs)
            result['bmag_x'] = best_bmag_x
            result['bmag_y'] = best_bmag_y
            mean_bmag = (result['bmag_x'] * result['bmag_y']).sqrt()
        if self.use_bmag:
            objective *= mean_bmag

        result['objective'] = objective

        return result

        
    def get_meas_scan_inputs(self, x_tuning: Tensor, bounds: Tensor, tkwargs: dict=None):
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
        x_meas_expanded = x_meas.reshape(-1,1).repeat(*x_tuning.shape[:-1],1)
        
        # repeat tuning configs as necessary and concat with column from the line above
        # to make xs shape: (n_tuning_configs*n_steps_quad_scan) x d ,
        # where d is the full dimension of the model/posterior space (tuning & meas)
        x_tuning_expanded = torch.repeat_interleave(x_tuning, 
                                                    self.n_steps_measurement_param, 
                                                    dim=-2)


        x = torch.cat(
            (x_tuning_expanded[..., :self.meas_dim], x_meas_expanded, x_tuning_expanded[..., self.meas_dim:]), 
            dim=-1
        )

        return x

    def evaluate_posterior_emittance(self, model, x_tuning, bounds, tkwargs:dict=None, n_samples:int=None):
        """
        inputs:
            x_tuning: tensor shape n_points x (n_dim-1) specifying points in the **tuning** space
                    at which to evaluate the objective.
        returns: 
            emit: tensor shape n_points x 1 or 2
            bmag: tensor shape n_points x 1 or 2
        """
        assert len(x_tuning.shape) in [2,3]
        # x_tuning must be shape (n_tuning_configs, n_tuning_dims) or (n_samples, n_tuning_configs, ndim)
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}
        
        x = self.get_meas_scan_inputs(x_tuning, bounds, tkwargs) # result shape n_tuning_configs*n_steps x ndim
        bss = self.evaluate_virtual_observables(model, x, n_samples) 

        # package inputs for emittance calculation

        bss = bss.reshape(-1, x_tuning.shape[-2], self.n_steps_measurement_param, bss.shape[-1])
        # bss.shape = (n_samples, x_tuning.shape[-2], self.n_steps_measurement_param, 1 or 2)
        x = x.reshape(-1, x_tuning.shape[-2], self.n_steps_measurement_param, x.shape[-1])
        # x.shape = (n_samples, x_tuning.shape[-2], self.n_steps_measurement_param, ndim)
        if len(x_tuning.shape) == 2:
            x = x.repeat(bss.shape[0],1,1,1)

        if self.x_key and not self.y_key:
            k = bdes_to_kmod(self.energy, self.q_len, x[..., self.meas_dim])
            beamsize_squared = bss[...,self.x_idx] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
            twiss0 = self.twiss0_x.repeat(*bss.shape[:2], 1)
        elif self.y_key and not self.x_key:
            k = -1 * bdes_to_kmod(self.energy, self.q_len, x[..., self.meas_dim])
            beamsize_squared = bss[...,self.y_idx] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
            twiss0 = self.twiss0_y.repeat(*bss.shape[:2], 1)
        else:
            k_x = bdes_to_kmod(self.energy, self.q_len, x[..., self.meas_dim])
            k_y = k_x * -1. # n_samples x n_tuning x n_steps
            k = torch.cat((k_x, k_y)) # shape (2*n_samples x n_tuning x n_steps)

            beamsize_squared = torch.cat((bss[...,self.x_idx], bss[...,self.y_idx]))

            rmat_x = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat_y = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat = torch.cat((rmat_x, rmat_y)) # shape (2*n_samples x n_tuning x 2 x 2)

            twiss0 = torch.cat((self.twiss0_x.repeat(*bss.shape[:2], 1),
                               self.twiss0_y.repeat(*bss.shape[:2], 1))
                              )

        # compute emittance
        rv = compute_emit_bmag(k.numpy(), 
                               beamsize_squared.detach().numpy(), 
                               self.q_len, 
                               rmat.numpy(), 
                               twiss0.numpy(),
                               thin_lens=self.thin_lens,
                               maxiter=self.maxiter_fit)

        emit = torch.from_numpy(rv['emittance'])
        bmag = torch.from_numpy(rv['bmag'])
        # emit.shape = (n_samples x n_tuning) or (2*n_samples x n_tuning) if optimizing both x and y
        # bmag.shape = (n_samples x n_tuning x nsteps) or (2*n_samples x n_tuning x nsteps) if optimizing both x and y
        
        if self.x_key and self.y_key:
            emit = torch.cat((emit[:bss.shape[0]].unsqueeze(-1), emit[bss.shape[0]:].unsqueeze(-1)), dim=-1) 
            # emit.shape = (n_samples, n_tuning, 1, 2)
            bmag = torch.cat((bmag[:bss.shape[0]].unsqueeze(-1), bmag[bss.shape[0]:].unsqueeze(-1)), dim=-1) 
            # bmag.shape = (n_samples, n_tuning, n_steps, 2)
        else:
            emit = emit.unsqueeze(-1)
            bmag = bmag.unsqueeze(-1)
        #final shapes: n_samples x n_tuning_configs (?? NEED TO CHECK THIS, don't think it's correct)

        return emit, bmag


class PathwiseMinimizeEmittance(EmittanceAlgorithm, PathwiseOptimization):

    def get_execution_paths(self, model: Model, bounds: Tensor) -> Tensor:
        # draw callable sample functions
        sample_functions_list = self.draw_sample_functions_list(model)

        best_inputs = self.execute_algorithm(sample_functions_list, bounds)
        best_meas_scan_inputs = self.get_meas_scan_inputs(best_inputs, bounds)
        best_meas_scan_outputs = torch.vstack([sample_func(best_meas_scan_inputs)
                                               for sample_func in sample_functions_list]
                                             ).T.unsqueeze(0)
        best_emit, best_bmag = self.evaluate_posterior_emittance(sample_functions_list,
                                                                      best_inputs,
                                                                      bounds
                                                                     )
        self.results = {}
        self.results['best_inputs'] = best_inputs
        self.results['best_emit'] = best_emit
        self.results['best_bmag'] = best_bmag
        self.results['sample_functions_list'] = sample_functions_list

        return best_meas_scan_inputs, best_meas_scan_outputs, {}

    def draw_sample_functions_list(self, model):
        sample_funcs_list = []
        for m in model.models:
            try:
                sample_funcs = draw_product_kernel_post_paths(m, n_samples=self.n_samples)
            except:
                sample_funcs = draw_matheron_paths(m, sample_shape=torch.Size([self.n_samples]))
            sample_funcs_list += [sample_funcs]
        return sample_funcs_list

    def _get_optimization_indeces(self, bounds) -> Tensor:
        '''
        Get indeces specifying parameters for virtual objective optimization.
        '''
        idcs = torch.tensor(range(bounds.shape[1]))
        mask = idcs != self.meas_dim
        return idcs[mask]