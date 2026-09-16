import torch
import torch.nn.functional as F

def prelu(x: float, alpha: float = 0.25) -> torch.Tensor:

    x = torch.tensor(x, dtype=torch.float32)
    alpha = torch.tensor(alpha, dtype=torch.float32)

    output = F.prelu(x, alpha)

    return output