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
        # Pad xi and y_obs to max_measure
        xi = self.pad(xi)
        y_obs = self.pad(y_obs)
        # Sample noise (traced models can't use torch.randn inside)
        noise = torch.randn(1, xi.shape[1]).float()  # TODO: assumes 1D design for now
        # Define mask
        mask = torch.ones(1, self.max_measure).bool()
        # Mask out 
        mask[len(xi):] = False 
        with torch.no_grad():
            log_probs, candidates = self.model(self.theta_values, y_obs, xi, noise, mask)
        return [
            {self.vocs.variable_names[0]: candidates[i].item()} for i in range(n_candidates)
        ]
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from xopt import VOCS
    from xopt import Xopt, Evaluator


    # define the function  TODO: update this to the double exponential plateau function
    def f(x, x0, w, b):
        return -(
            torch.tanh(-x / b - w / 2 / b + x0 / b) + torch.tanh(x / b - w / 2 / b - x0 / b)
        )


    # visualize the ground truth function
    ground_truth_x0 = 4.0  # lower edge location
    ground_truth_w = 2.5  # plateau width
    ground_truth_b = 0.1  # sharpness of the plateau edge
    test_x = torch.linspace(-5, 100, 100)

    fig, ax = plt.subplots()
    ax.plot(test_x, f(test_x, x0=ground_truth_x0, w=ground_truth_w, b=ground_truth_b))

    # TODO: update this to be default measurement for amortized BOED model.
    vocs = VOCS(variables={"x": [-10, 100]}, observables=["y"])

    generator = AmortizedBOEDGenerator(
        vocs=vocs,
        model_path="examples/fixtures/model_traced.pt"
    )

    evaluator = Evaluator(
        function=lambda x: {
            "y": float(
                f(torch.tensor(x["x"]), ground_truth_x0, ground_truth_w, ground_truth_b)
            )
        }
    )

    X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator)

    X.grid_evaluate(1)

    for _ in range(5):
        X.step()

    X.data.plot.scatter(x="x", y="y", ax=ax, color="red")
    plt.show()
