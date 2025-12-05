import torch
from xopt.generator import Generator
from xopt import VOCS


class AmortizedBOEDGenerator(Generator):
    """
    Amortized Bayesian Optimal Experimental Design generator using a pre-trained neural network.
    
    Attributes:
    -----------
    device : str
        Device to run the model on ('cpu' or 'cuda').
    theta_range : tuple
        Range of theta values where the model is valid.
    """
    
    device: str = 'cpu'
    max_measure: int = 20  # TODO: make this configurable
    n_thetas: int = 100  # TODO: make this configurable
    theta_range: tuple = (0.0, 100)
    
    # These are not Pydantic fields - they're set in __init__
    model: torch.jit.ScriptModule = None
    
    def __init__(self, model_path: str, vocs=None, **kwargs):
        """Initialize generator with a TorchScript model."""
        super().__init__(vocs=vocs, **kwargs)
        # Load the traced TorchScript model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        # Generate theta values within the specified range
        self.__dict__['theta_values'] = torch.linspace(
            self.theta_range[0], self.theta_range[1], steps=self.n_thetas
        ).unsqueeze(1).float()

    def pad(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad the input tensor to the maximum measurement size."""
        pad_size = self.max_measure - tensor.shape[1]
        if pad_size > 0:
            padding = torch.zeros(tensor.shape[0], pad_size, tensor.shape[2]).float()
            return torch.cat([tensor, padding], dim=1)
        return tensor

    def generate(self, n_candidates: int = 1) -> list[dict]:
        # Unpack current internal state of the generator
        xi = torch.tensor(self.data[self.vocs.variable_names].values).float().unsqueeze(0)
        y_obs = torch.tensor(self.data[self.vocs.observable_names].values).float().unsqueeze(0)
        # # Pad xi and y_obs to max_measure
        # xi = self.pad(xi)
        # y_obs = self.pad(y_obs)
        # Iterate to generate candidates
        candidates = []
        # TODO: vectorize over n_candidates
        for _ in range(n_candidates):
            # Sample noise (traced models can't use torch.randn inside)
            noise = torch.randn(1, xi.shape[-1]).float()  # TODO: assumes 1D design for now
            with torch.no_grad():
                candidate, log_probs = self.model(self.theta_values, y_obs, xi, noise)
            candidates.append({self.vocs.variable_names[0]: candidate.item()})
        return candidates


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from xopt import VOCS
    from xopt import Xopt, Evaluator
    import numpy as np


    # define the function
    def exp2_piecewise(x, A1, tau1, A2, tau2, T0):
        """
        Piecewise double-exponential (numpy version):
        y(x) = 0,                   x < T0
            = A1*exp(-(x-T0)/tau1) + A2*exp(-(x-T0)/tau2) - (A1+A2),  x >= T0
        This enforces y=0 at x=T0.
        """
        y_model = np.zeros_like(x, dtype=float)
        mask = x >= T0
        x_shift = np.maximum(x - T0, 0.0)
        y_model[mask] = (
            A1 * np.exp(-x_shift[mask] / tau1)
            + A2 * np.exp(-x_shift[mask] / tau2)
            - (A1 + A2)
        )
        return y_model


    # visualize the ground truth function
    A1 = 3.0
    tau1 = 0.5
    A2 = 8.0
    tau2 = 2.0
    sigma = 1.0
    ground_truth_x0 = 30.0
    test_x = torch.linspace(-10, 100, 100)

    fig, ax = plt.subplots()
    ax.plot(test_x, exp2_piecewise(test_x, A1=A1, tau1=tau1, A2=A2, tau2=tau2, T0=ground_truth_x0))

    # TODO: How to set first design point?
    vocs = VOCS(variables={"x": [-10, 100]}, observables=["y"])

    generator = AmortizedBOEDGenerator(
        vocs=vocs,
        model_path="examples/fixtures/model_traced.pt"
    )

    evaluator = Evaluator(
        function=lambda x: {
            "y": float(
                exp2_piecewise(torch.tensor(x["x"]), A1=A1, tau1=tau1, A2=A2, tau2=tau2, T0=ground_truth_x0) + (2 * sigma * np.random.rand() - sigma)
            )
        }
    )

    X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator)

    init_point = X.vocs.grid_inputs(1)
    # Replace with a fixed initial point for reproducibility
    init_point["x"][0] = 100
    X.evaluate_data(init_point)

    for _ in range(generator.max_measure - 1):
        X.step()

    X.data.plot.scatter(x="x", y="y", ax=ax, color="red")
    plt.show()
