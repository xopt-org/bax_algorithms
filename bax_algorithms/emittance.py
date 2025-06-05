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
from lcls_tools.common.measurements.emittance_measurement import compute_emit_bmag_quad_scan
from xopt.generators.bayesian.bax.algorithms import Algorithm

class EmittanceAlgorithm(Algorithm):
    x_key: str = Field(None,
        description="key designating the beamsize squared output in x from evaluate function")
    y_key: str = Field(None,
        description="key designating the beamsize squared output in y from evaluate function")
    scale_factor: float = Field(1.0,
        description="factor by which to multiply the quad inputs to get focusing strengths")
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

    def evaluate_virtual_objective(self, model, x, bounds, tkwargs:dict=None, n_samples:int=None):
        """
        inputs:
            model: a botorch ModelListGP
            x_tuning: tensor shape n_points x n_dim specifying points in the full-dimensional model space
                    at which to evaluate the objective.
        returns: 
            res: tensor shape n_points x 1
            emit: tensor shape n_points x 1 or 2
            bmag: tensor shape n_points x 1 or 2
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
        
        if self.x_key and self.y_key:
            res = (emit[...,0] * emit[...,1]).sqrt()
            if self.use_bmag:
                bmag_mean = (bmag[...,0] * bmag[...,1]).sqrt()
                bmag_min, bmag_min_id = torch.min(bmag_mean, dim=-1, keepdim=True)
                res = bmag_min * res
        else:
            res = emit
            if self.use_bmag:
                bmag_min, bmag_min_id = torch.min(bmag, dim=-2) # NEED TO CHECK THIS DIM
                res = (bmag_min * res).squeeze(-1)
        return res

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
        bss = bss.reshape(-1, x_tuning.shape[-2], self.n_steps_measurement_param, bss.shape[-1])
        # bss.shape = (n_samples, x_tuning.shape[-2], self.n_steps_measurement_param, 1 or 2)
        x = x.reshape(-1, x_tuning.shape[-2], self.n_steps_measurement_param, x.shape[-1])
        # x.shape = (n_samples, x_tuning.shape[-2], self.n_steps_measurement_param, ndim)
        if len(x_tuning.shape) == 2:
            x = x.repeat(bss.shape[0],1,1,1)

        if self.x_key and not self.y_key:
            k = x[..., self.meas_dim] * self.scale_factor # n_samples x n_tuning x n_steps
            beamsize_squared = bss[...,0] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
            twiss0 = self.twiss0_x.repeat(*bss.shape[:2], 1)
        elif self.y_key and not self.x_key:
            k = x[..., self.meas_dim] * (-1. * self.scale_factor) # n_samples x n_tuning x n_steps
            beamsize_squared = bss[...,0] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
            twiss0 = self.twiss0_y.repeat(*bss.shape[:2], 1)
        else:
            k_x = (x[..., self.meas_dim] * self.scale_factor) # n_samples x n_tuning x n_steps
            k_y = k_x * -1. # n_samples x n_tuning x n_steps
            k = torch.cat((k_x, k_y)) # shape (2*n_samples x n_tuning x n_steps)

            beamsize_squared = torch.cat((bss[...,0], bss[...,1]))

            rmat_x = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat_y = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat = torch.cat((rmat_x, rmat_y)) # shape (2*n_samples x n_tuning x 2 x 2)

            twiss0 = torch.cat((self.twiss0_x.repeat(*bss.shape[:2], 1),
                               self.twiss0_y.repeat(*bss.shape[:2], 1))
                              )

        rv = compute_emit_bmag_quad_scan(k.numpy(), 
                                      beamsize_squared.detach().numpy(), 
                                      self.q_len, 
                                      rmat.numpy(), 
                                      twiss0.numpy(),
                                      thin_lens=self.thin_lens,
                                      maxiter=self.maxiter_fit)
        # result shapes: (n_samples x n_tuning), (n_samples x n_tuning x nsteps), (n_samples x n_tuning x 3 x 1), (n_samples x n_tuning) 
        # or (2*n_samples x n_tuning), (2*n_samples x n_tuning x nsteps), (2*n_samples x n_tuning x 3 x 1), (2*n_samples x n_tuning) 

        emit = torch.from_numpy(rv['emittance'])
        bmag = torch.from_numpy(rv['bmag'])

        if self.x_key and self.y_key:
            emit = torch.cat((emit[:bss.shape[0]].unsqueeze(-1), emit[bss.shape[0]:].unsqueeze(-1)), dim=-1) # n_samples x n_tuning x 1 or 2
            bmag = torch.cat((bmag[:bss.shape[0]].unsqueeze(-1), bmag[bss.shape[0]:].unsqueeze(-1)), dim=-1) # n_samples x n_tuning x n_steps x 1 or 2
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
        best_meas_scan_outputs = torch.vstack([sample_func(best_meas_scan_inputs) for sample_func in sample_functions_list]).T.unsqueeze(0)
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