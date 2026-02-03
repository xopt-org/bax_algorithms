import pytest

import torch
from xopt import Xopt
from xopt.vocs import VOCS
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.bax_generator import BaxGenerator
from bax_algorithms.emittance import PathwiseMinimizeEmittance
from bax_algorithms.pathwise.optimize import DifferentialEvolution
from bax_algorithms.utils import (
    get_bax_mean_prediction,
    get_bax_model_and_bounds,
    tuning_input_tensor_to_dict,
)
from bax_algorithms.visualize import visualize_virtual_measurement_result


class TestEmittanceBax:
    def setup_method(self):
        rmat_x = torch.tensor([[1.0000, 2.2000], [0.0000, 1.0000]])

        rmat_y = torch.tensor([[1.0000, 2.2000], [0.0000, 1.0000]])

        meas_dim = 1
        var_names = ["x" + str(i) for i in range(3)]
        meas_param = var_names[meas_dim]

        def mock_measure_beamsize(input_dict):
            x = torch.tensor([])
            for key in input_dict.keys():
                x = torch.cat((x, torch.tensor([input_dict[key]])))
            return {
                "xrms": float(50.0 + x.pow(2).sum()),
                "yrms": float(50.0 + x.pow(2).sum()),
            }

        variables = {var_name: [-1, 1] for var_name in var_names}
        variables[meas_param] = [-5, 5]

        # construct vocs
        vocs = VOCS(
            variables=variables,
            observables=["xrms", "yrms"],
        )

        # Prepare Algorithm
        algo_kwargs = {
            "x_key": "xrms",
            "y_key": "yrms",
            "energy": 80e6,
            "q_len": 0.08,
            "rmat_x": rmat_x,
            "rmat_y": rmat_y,
            "twiss0_x": torch.tensor([10.0, -1.0]),
            "twiss0_y": torch.tensor([11.0, -2.0]),
            "n_samples": 3,
            "meas_dim": meas_dim,
            "n_steps_measurement_param": 11,
            "use_bmag": True,
            "observable_names_ordered": ["xrms", "yrms"],
            "optimizer": DifferentialEvolution(minimize=True, maxiter=10, verbose=True),
            "crop_scans": False,
        }

        algo = PathwiseMinimizeEmittance(**algo_kwargs)

        # construct BAX generator
        generator = BaxGenerator(
            vocs=vocs,
            # gp_constructor=model_constructor,
            # numerical_optimizer=numerical_optimizer,
            algorithm=algo,
            use_cuda=False,
        )

        evaluator = Evaluator(function=mock_measure_beamsize)

        # construct Xopt optimizer
        self.X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)

        # call X.random_evaluate() to generate random initial points and evaluate on test_func
        self.X.random_evaluate(5)

        self.X.generator.train_model()

        self.reference_point = {var: 0.0 for var in self.X.vocs.variable_names}

    def test_step(self):
        self.X.step()

    def test_get_bax_mean_prediction(self):
        mean_optimizer = DifferentialEvolution(
            minimize=True, popsize=100, maxiter=100, verbose=True
        )
        x_tuning = get_bax_mean_prediction(self.X.generator, mean_optimizer)

    def test_tuning_input_tensor_to_dict(self):
        x_tuning = torch.tensor([[0.0, 0.0]])
        x_tuning_dict = tuning_input_tensor_to_dict(self.X.generator, x_tuning)

    def test_get_bax_model_and_bounds(self):
        bax_model, bounds = get_bax_model_and_bounds(self.X.generator)

    def test_visualize_virtual_measurement_result_1d(self):
        fig, ax = visualize_virtual_measurement_result(
            self.X.generator,
            variable_names=["x0"],
            reference_point=self.reference_point,
            n_grid=100,
            n_samples=100,
            result_keys=["objective", "emit_x", "emit_y", "bmag_x", "bmag_y"],
        )

    def test_visualize_virtual_measurement_result_2d(self):
        fig, ax = visualize_virtual_measurement_result(
            self.X.generator,
            variable_names=["x0", "x2"],
            reference_point=self.reference_point,
            n_grid=10,
            n_samples=100,
            result_keys=["objective", "emit_x", "emit_y", "bmag_x", "bmag_y"],
        )
