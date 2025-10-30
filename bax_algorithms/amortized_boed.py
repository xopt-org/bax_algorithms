import torch
from torch import nn
from xopt.generator import Generator

class AmortizedBOEDGenerator(Generator):
    def __init__(self, model_path: str, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()

    def generate(self, n_candidates: int = 1) -> list[dict]:
        # Unpack current internal state of the generator
        # TODO: not sure how to index the pandas df correctly here?
        y_obs = torch.tensor(self.data["y_obs"], device=self.device).float()
        xi = torch.tensor(self.data["xi"], device=self.device).float()
        with torch.no_grad():
            candidates = self.model.sample_designs(y_obs, xi, n_candidates).cpu().numpy()
        # Not sure what the correct dictionary structure is here?
        return [{'records': c} for c in candidates]