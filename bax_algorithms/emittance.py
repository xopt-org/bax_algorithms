from typing import List, Dict, Optional, Tuple, Union
from pydantic import Field
import torch
from torch import Tensor
import copy

from bax_algorithms.pathwise.sampling import draw_product_kernel_post_paths
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
from gpytorch.kernels import ProductKernel, MaternKernel
from emitopt.analysis import compute_emit_bmag

class EmittanceAlgorithm(Algorithm, ABC):
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
    thick_quad: bool = Field(True,
        description="Whether to use thick-quad (or thin, if False) transport for emittance calc")
    use_bmag: bool = Field(True,
        description="Whether to multiply the emit by the bmag to get virtual objective.")
    results: dict = Field({},
        description="Dictionary to store results from emittance calculcation")
    maxiter_fit: int = Field(20,
        description="Maximum number of iterations in nonlinear emittance fitting.")
    
    @property
    def observable_names_ordered(self) -> list:  
        # get observable model names in the order they appear in the model (ModelList)
        return [key for key in [self.x_key, self.y_key] if key]

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

    def evaluate_virtual_objective(self, model, x, bounds, tkwargs:dict=None, n_samples=10000, use_mean=False, use_bmag=True):
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
        emit, bmag, is_valid, validity_rate, bss = self.evaluate_posterior_emittance(model, 
                                                                                     x_tuning, 
                                                                                     bounds, 
                                                                                     tkwargs, 
                                                                                     n_samples,
                                                                                     use_mean)
        self.results["emit"] = emit
        self.results["bmag"] = bmag
        self.results["is_valid"] = is_valid
        self.results["validity_rate"] = validity_rate
        self.results["bss"] = bss
        
        if self.x_key and self.y_key:
            res = (emit[...,0] * emit[...,1]).sqrt()
            if use_bmag:
                bmag_mean = (bmag[...,0] * bmag[...,1]).sqrt()
                bmag_min, bmag_min_id = torch.min(bmag_mean, dim=-1)
                res = bmag_min * res
        else:
            res = emit
            if use_bmag:
                bmag_min, bmag_min_id = torch.min(bmag, dim=-2) # NEED TO CHECK THIS DIM
                res = (bmag_min * res).squeeze(-1)
        return res

    def evaluate_posterior_emittance(self, model, x_tuning, bounds, tkwargs:dict=None, n_samples=100, use_mean=False):
        """
        inputs:
            x_tuning: tensor shape n_points x (n_dim-1) specifying points in the **tuning** space
                    at which to evaluate the objective.
        returns: 
            emit: tensor shape n_points x 1 or 2
            bmag: tensor shape n_points x 1 or 2
        """
        # x_tuning must be shape n_tuning_configs x n_tuning_dims
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}
        x = self.get_meas_scan_inputs(x_tuning, bounds, tkwargs) # result shape n_tuning_configs*n_steps x ndim
        
        if isinstance(model, ModelList):
            assert len(x_tuning.shape)==2
            p = model.posterior(x)
            if use_mean:
                bss = p.mean
                bss = bss.reshape(1, x_tuning.shape[0], self.n_steps_measurement_param, -1)
                x = x.reshape(1, x_tuning.shape[0], self.n_steps_measurement_param, -1) # result n_tuning_configs x n_steps x ndim
            else:
                bss = p.sample(torch.Size([n_samples])) # result shape n_samples x n_tuning_configs*n_steps x num_outputs (1 or 2)

                x = x.reshape(x_tuning.shape[0], self.n_steps_measurement_param, -1) # result n_tuning_configs x n_steps x ndim
                x = x.repeat(n_samples,1,1,1) 
                # result shape n_samples x n_tuning_configs x n_steps x ndim
                bss = bss.reshape(n_samples, x_tuning.shape[0], self.n_steps_measurement_param, -1)
                # result shape n_samples x n_tuning_configs x n_steps x num_outputs (1 or 2)
        else:
            assert x_tuning.shape[0]==model[0].n_samples
            beamsize_squared_list = [sample_funcs(x).reshape(*x_tuning.shape[:-1], self.n_steps_measurement_param)
                                     for sample_funcs in model]
            # each tensor in beamsize_squared (list) will be shape n_samples x n_tuning_configs x n_steps

            x = x.reshape(*x_tuning.shape[:-1], self.n_steps_measurement_param, -1)
            # n_samples x n_tuning_configs x n_steps x ndim
            bss = torch.stack(beamsize_squared_list, dim=-1) 
            # result shape n_samples x n_tuning_configs x n_steps x num_outputs (1 or 2)
        
        if self.x_key and not self.y_key:
            k = x[..., self.meas_dim] * self.scale_factor # n_samples x n_tuning x n_steps
            beamsize_squared = bss[...,0] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
            beta0 = self.twiss0_x[0].repeat(*bss.shape[:2], 1)
            alpha0 = self.twiss0_x[1].repeat(*bss.shape[:2], 1)
        elif self.y_key and not self.x_key:
            k = x[..., self.meas_dim] * (-1. * self.scale_factor) # n_samples x n_tuning x n_steps
            beamsize_squared = bss[...,0] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
            beta0 = self.twiss0_y[0].repeat(*bss.shape[:2], 1)
            alpha0 = self.twiss0_y[1].repeat(*bss.shape[:2], 1)
        else:
            k_x = (x[..., self.meas_dim] * self.scale_factor) # n_samples x n_tuning x n_steps
            k_y = k_x * -1. # n_samples x n_tuning x n_steps
            k = torch.cat((k_x, k_y)) # shape (2*n_samples x n_tuning x n_steps)

            beamsize_squared = torch.cat((bss[...,0], bss[...,1]))


            rmat_x = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat_y = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat = torch.cat((rmat_x, rmat_y)) # shape (2*n_samples x n_tuning x 2 x 2)

            beta0_x = self.twiss0_x[0].repeat(*bss.shape[:2], 1)
            beta0_y = self.twiss0_y[0].repeat(*bss.shape[:2], 1)
            beta0 = torch.cat((beta0_x, beta0_y))

            alpha0_x = self.twiss0_x[1].repeat(*bss.shape[:2], 1)
            alpha0_y = self.twiss0_y[1].repeat(*bss.shape[:2], 1)
            alpha0 = torch.cat((alpha0_x, alpha0_y))

        emit, bmag, sig, is_valid = compute_emit_bmag(k, 
                                          beamsize_squared, 
                                          self.q_len, 
                                          rmat, 
                                          beta0,
                                          alpha0,
                                          thick=self.thick_quad,
                                          maxiter=self.maxiter_fit)
        # result shapes: (n_samples x n_tuning), (n_samples x n_tuning x nsteps), (n_samples x n_tuning x 3 x 1), (n_samples x n_tuning) 
        # or (2*n_samples x n_tuning), (2*n_samples x n_tuning x nsteps), (2*n_samples x n_tuning x 3 x 1), (2*n_samples x n_tuning) 

        if self.x_key and self.y_key:
            emit = torch.cat((emit[:bss.shape[0]].unsqueeze(-1), emit[bss.shape[0]:].unsqueeze(-1)), dim=-1) # n_samples x n_tuning x 1 or 2
            bmag = torch.cat((bmag[:bss.shape[0]].unsqueeze(-1), bmag[bss.shape[0]:].unsqueeze(-1)), dim=-1) # n_samples x n_tuning x n_steps x 1 or 2
            is_valid = torch.logical_and(is_valid[:bss.shape[0]], is_valid[bss.shape[0]:])
        else:
            emit = emit.unsqueeze(-1)
            bmag = bmag.unsqueeze(-1)
        #final shapes: n_samples x n_tuning_configs (?? NEED TO CHECK THIS, don't think it's correct)
        
        validity_rate = torch.sum(is_valid, dim=0)/is_valid.shape[0]
        #shape n_tuning_configs
        
        return emit, bmag, is_valid, validity_rate, bss


class PathwiseMinimizeEmittance(EmittanceAlgorithm, PathwiseOptimization):

    def get_execution_paths(self, best_x: Tensor, bounds: Tensor):

        x_exe = torch.tensor([])
        y_exe = torch.tensor([])

        best_x =torch.from_numpy(res.x)
        best_x_tuning = best_x[torch.arange(best_x.shape[0])!=self.meas_dim].reshape(1,1,-1)
        best_meas_scan_x = self.get_meas_scan_inputs(best_x_tuning, bounds)
        best_meas_scan_y = torch.vstack([sample_func(best_meas_scan_x) for sample_func in sample_funcs_list]).T.unsqueeze(0)
        x_exe = torch.cat((x_exe, best_meas_scan_x))
        y_exe = torch.cat((y_exe, best_meas_scan_y))

        return x_exe, y_exe, results

    def draw_sample_functions_list(self, model, n_samples=1):
        cpu_models = [copy.deepcopy(m).cpu() for m in model.models]
        sample_funcs_list = [draw_product_kernel_post_paths(m, n_samples=n_samples) for m in cpu_models]
        return sample_funcs_list

    def get_virtual_optimization_bounds(self, xopt_bounds: Tensor) -> Tensor:
        virtual_opt_bounds = torch.clone(xopt_bounds.T)
        virtual_opt_bounds[self.meas_dim] = torch.tensor([0,0])
        return virtual_opt_bounds