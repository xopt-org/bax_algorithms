import torch
from torch import Tensor
from botorch.models.model import Model
from pydantic import Field, PositiveInt
from bax_algorithms.pathwise.base import PathwiseOptimization
from xopt.generators.bayesian.bax.algorithms import (
    OptimizationAlgorithmResult,
    VirtualMeasurementResult,
)


class VirtualAlignmentMeasurementResult(VirtualMeasurementResult):
    misalignment_x: Tensor = Field(
        description="The misalignment in the x transverse dimension."
    )
    misalignment_y: Tensor = Field(
        description="The misalignment in the y transverse dimension."
    )


class PathwiseSolenoidAlignment(PathwiseOptimization):
    name: str = Field("pathwise_solenoid_alignment", frozen=True)
    x_key: str | None= Field(
        None,
        description="key designating the centroid position in x from evaluate function",
    )
    y_key: str | None = Field(
        None,
        description="key designating the centroid poisition in y from evaluate function",
    )
    meas_dim: int | None = Field(
        None,
        description="index identifying the measurement quad dimension in the model",
    )
    n_steps_measurement_param: int = Field(
        5, description="number of steps to use in the virtual measurement scans"
    )
    n_batch: PositiveInt = Field(
        1,
        description="Number of sample batches to optimize, with each batch containing self.n_samples",
    )

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
        self, model: Model, x: Tensor, bounds: Tensor, n_samples: int | None = None
    ) -> VirtualAlignmentMeasurementResult:
        """
        inputs:
            model: a botorch ModelListGP
            x: tensor shape (n_points, n_dim) or (n_samples, n_points, n_dim)
                    specifying points in the full-dimensional model space
                    at which to evaluate the objective.
            bounds: tensor shape (2, n_dim) specifying the upper and lower measurement bounds
        returns:
            VirtualAlignmentMeasurementResult
        """
        tuning_idxs = torch.arange(bounds.shape[1])
        tuning_idxs = tuning_idxs[
            tuning_idxs != self.meas_dim
        ]  # remove measurement dim index
        x_tuning = x[..., tuning_idxs]

        # x_tuning must be shape n_tuning_configs x n_tuning_dims
        misalignment = self.evaluate_posterior_misalignment(
            model,
            x_tuning,
            bounds,
            n_samples,
        )

        # store virtual measurement results
        misalignment_x = misalignment[..., self.x_idx]
        misalignment_y = misalignment[..., self.y_idx]
        objective = misalignment_x + misalignment_y
        virtual_alignment_result = VirtualAlignmentMeasurementResult(
            objective=objective,
            misalignment_x=misalignment_x,
            misalignment_y=misalignment_y,
        )

        return virtual_alignment_result

    def get_meas_scan_inputs(
        self, x_tuning: Tensor, bounds: Tensor
    ) -> Tensor:
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

        x_meas = torch.linspace(
            *bounds.T[self.meas_dim], self.n_steps_measurement_param
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

    def evaluate_posterior_misalignment(
        self, model: Model, x_tuning: Tensor, bounds: Tensor, n_samples: int | None = None
    ):
        """
        inputs:
            x_tuning: tensor shape n_points x (n_dim-1) specifying points in the **tuning** space
                    at which to evaluate the objective.
        returns:
            misalignment: tensor shape n_points x 1 or 2
        """
        assert len(x_tuning.shape) in [2, 3]
        # x_tuning must be shape (n_tuning_configs, n_tuning_dims) or (n_samples, n_tuning_configs, ndim)

        x = self.get_meas_scan_inputs(
            x_tuning, bounds
        )  # result shape n_tuning_configs*n_steps x ndim
        centroid_position = self.evaluate_virtual_observables(model, x, n_samples)

        # package inputs for emittance calculation

        centroid_position = centroid_position.reshape(
            -1,
            x_tuning.shape[-2],
            self.n_steps_measurement_param,
            centroid_position.shape[-1],
        )
        # centroid_position.shape = (n_samples, x_tuning.shape[-2], self.n_steps_measurement_param, 1 or 2)
        x = x.reshape(
            -1, x_tuning.shape[-2], self.n_steps_measurement_param, x.shape[-1]
        )
        # x.shape = (n_samples, x_tuning.shape[-2], self.n_steps_measurement_param, ndim)
        if len(x_tuning.shape) == 2:
            x = x.repeat(centroid_position.shape[0], 1, 1, 1)

        # compute the independent misalignments in x and y
        misalignment_x = centroid_position[..., self.x_idx].std(dim=-1, keepdim=True)
        misalignment_y = centroid_position[..., self.y_idx].std(dim=-1, keepdim=True)
        misalignment = torch.cat((misalignment_x, misalignment_y), dim=-1)

        return misalignment

    def execute(self, model: Model, bounds: Tensor) -> OptimizationAlgorithmResult:
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
            scan_outputs = [
                sample_func(best_meas_scan_inputs).unsqueeze(-1)
                for sample_func in sample_functions_list
            ]
            best_meas_scan_outputs = torch.cat(scan_outputs, dim=-1)
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

    def _get_optimization_indeces(self, bounds: Tensor) -> Tensor:
        """
        Get indeces specifying parameters for virtual objective optimization.
        """
        idcs = torch.tensor(range(bounds.shape[1]))
        mask = idcs != self.meas_dim
        return idcs[mask]
