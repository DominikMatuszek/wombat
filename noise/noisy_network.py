import torch


class NoisyNetwork(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, noise_std: float = 0.1, seed: int = 42):
        super().__init__()
        self.noise_std = noise_std
        self.base_model = base_model

        torch.manual_seed(seed)

        for name, param in base_model.named_parameters():
            param.data = param.data.float()

            if name.startswith("mpnn1"):
                noise = torch.abs(torch.randn_like(param)) * self.noise_std
                param.data += noise

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
