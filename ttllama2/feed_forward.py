import ttnn
import torch
from . import LightweightModule
from llama2.model import ModelArgs


class FeedForward(LightweightModule):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, layer_num, state_dict, device, dtype=ttnn.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.device = device
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            # Round the hidden_dim to the nearest multiple of the multiple_of parameter
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        prefix = f"layers.{layer_num}.feed_forward."

        self.w1 = ttnn.as_tensor(
            torch.transpose(state_dict[f"{prefix}w1.weight"], -2, -1,),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
            device=self.device
        )
        self.w2 = ttnn.as_tensor(
            torch.transpose(state_dict[f"{prefix}w2.weight"], -2, -1,),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
            device=self.device
        )
        self.w3 = ttnn.as_tensor(
            torch.transpose(state_dict[f"{prefix}w3.weight"], -2, -1,),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
            device=self.device
        )

        self.w1 = ttnn.to_layout(self.w1, ttnn.TILE_LAYOUT)
        self.w2 = ttnn.to_layout(self.w2, ttnn.TILE_LAYOUT)
        self.w3 = ttnn.to_layout(self.w3, ttnn.TILE_LAYOUT)


    def forward(self, x):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x1 = ttnn.linear(
            x,
            self.w1,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
        ) 
        swish = ttnn.silu(x1)
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = ttnn.linear(
            x,
            self.w3,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
        ) 
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = ttnn.mul(swish, x_V)
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = ttnn.linear(
            x,
            self.w2,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
        ) 
        return x