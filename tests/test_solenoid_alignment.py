import torch
from xopt import Xopt
from xopt.vocs import VOCS
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.bax_generator import BaxGenerator
from bax_algorithms.solenoid_alignment import PathwiseSolenoidAlignment
from bax_algorithms.pathwise.optimize import DifferentialEvolution
from bax_algorithms.utils import (
    get_bax_mean_prediction,
    get_bax_model_and_bounds,
    tuning_input_tensor_to_dict,
)
from bax_algorithms.visualize import visualize_virtual_measurement_result


class TestEmittanceBax:
    def setup_method(self):
        meas_dim = 1
        var_names = ["x" + str(i) for i in range(3)]

        def mock_measure_centroid(input_dict):
            x = torch.tensor([])
            for key in input_dict.keys():
                x = torch.cat((x, torch.tensor([input_dict[key]])))
            return {
                "x_centroid": float(x[0] * x[1]),
                "y_centroid": -1 * float(x[2] * x[1]),
            }

        variables = {var_name: [-3, 3] for var_name in var_names}

        # construct vocs
        vocs = VOCS(
            variables=variables,
            observables=["x_centroid", "y_centroid"],
        )

        # Prepare Algorithm
        algo_kwargs = {
            "x_key": "x_centroid",
            "y_key": "y_centroid",
            "n_samples": 3,
            "meas_dim": meas_dim,
            "n_steps_measurement_param": 11,
            "observable_names_ordered": ["x_centroid", "y_centroid"],
            "optimizer": DifferentialEvolution(minimize=True, maxiter=10, verbose=True),
        }

        algo = PathwiseSolenoidAlignment(**algo_kwargs)

        # construct BAX generator
        generator = BaxGenerator(
            vocs=vocs,
            algorithm=algo,
        )

        evaluator = Evaluator(function=mock_measure_centroid)

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
        get_bax_mean_prediction(self.X.generator, mean_optimizer)

    def test_tuning_input_tensor_to_dict(self):
        x_tuning = torch.tensor([[0.0, 0.0]])
        tuning_input_tensor_to_dict(self.X.generator, x_tuning)

    def test_get_bax_model_and_bounds(self):
        bax_model, bounds = get_bax_model_and_bounds(self.X.generator)

    def test_visualize_virtual_measurement_result_1d(self):
        fig, ax = visualize_virtual_measurement_result(
            self.X.generator,
            variable_names=["x0"],
            reference_point=self.reference_point,
            n_grid=100,
            n_samples=100,
            result_keys=["objective", "misalignment_x", "misalignment_y"],
        )

    def test_visualize_virtual_measurement_result_2d(self):
        fig, ax = visualize_virtual_measurement_result(
            self.X.generator,
            variable_names=["x0", "x2"],
            reference_point=self.reference_point,
            n_grid=11,
            n_samples=100,
            result_keys=["objective", "misalignment_x", "misalignment_y"],
        )
