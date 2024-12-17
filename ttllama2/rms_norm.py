import ttnn
import torch
from . import LightweightModule

class RMSNorm(LightweightModule):
    def __init__(self, device, dim, eps: float, state_dict, prefix, dtype=ttnn.bfloat16, add_prefix=True):
        super().__init__()
        self.eps = eps
        self.device = device
        self.dtype=dtype
        # The gamma parameter
        if add_prefix:
            prefix = f"layers.{prefix}weight"
        torch_weight = state_dict[prefix].unsqueeze(0).view(1, 1, dim)
        self.weight = ttnn.as_tensor(
            # torch.transpose(state_dict[prefix], -2, -1,),
            torch_weight,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=dtype,
        )
        self.weight = ttnn.to_layout(self.weight, ttnn.TILE_LAYOUT)

    def forward(self, x):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        norm_x = ttnn.mul(x,ttnn.rsqrt(ttnn.mean(ttnn.pow(x,2), -1)) + self.eps)
        return ttnn.mul(norm_x, self.weight)