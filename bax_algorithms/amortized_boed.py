import torch
from xopt.generator import Generator
from xopt import VOCS

class AmortizedBOEDGenerator(Generator):
    """
    Amortized Bayesian Optimal Experimental Design generator using a pre-trained neural network.
    
    Attributes:
    -----------
    model_path : str
        Path to the traced TorchScript model file.
    device : str
        Device to run the model on ('cpu' or 'cuda').
    """
    
    model_path: str
    device: str = 'cpu'
    vocs: VOCS
    
    # These will be set after initialization
    model: torch.jit.ScriptModule = None
    
    def model_post_init(self, __context):
        """Called after Pydantic initialization to load the model."""
        super().__init__()
        # Load the traced TorchScript model
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()

    def generate(self, n_candidates: int = 1) -> list[dict]:
        # Unpack current internal state of the generator
        xi = torch.tensor(self.data[self.vocs.variable_names].values).float().flatten()
        y_obs = torch.tensor(self.data[self.vocs.observable_names].values).float().flatten()
        with torch.no_grad():
            # TODO: will only ever produce a single candidate, need to call multiple times or modify model
            candidates = self.model(y_obs, xi).cpu().numpy()
        # Not sure what the correct dictionary structure is here?
        # return [{'records': c} for c in candidates]
        return [
            {self.vocs.variable_names[0]: candidates[i].item()} for i in range(n_candidates)
        ]
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
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
    test_x = torch.linspace(0, 6, 100)

    fig, ax = plt.subplots()
    ax.plot(test_x, f(test_x, x0=ground_truth_x0, w=ground_truth_w, b=ground_truth_b))

    # TODO: update this to be default measurement for amortized BOED model.
    vocs = VOCS(variables={"x": [0.0, 6.0]}, observables=["y"])

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

    X.grid_evaluate(5)

    for _ in range(5):
        X.step()

    X.data.plot.scatter(x="x", y="y", ax=ax, color="red")
    plt.show()
