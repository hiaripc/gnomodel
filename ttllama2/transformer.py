import ttnn
import torch
from . import LightweightModule
from llama2.model import ModelArgs
from . import TransformerBlock
from . import RMSNorm
from llama2.model import precompute_freqs_cis
from typing import Optional


class Transformer(LightweightModule):
    # last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs, state_dict, device, dtype=ttnn.bfloat16):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.dtype = dtype
        self.device = device
        # self.dropout = nn.Dropout(params.dropout)
        self.layers = [
            TransformerBlock(i, params, state_dict, device)
            for i in range(self.n_layers)
        ]
        self.norm = RMSNorm(device, params.dim, params.norm_eps, state_dict, prefix="norm.weight", add_prefix=False)
        self.output = ttnn.as_tensor(
            torch.transpose(state_dict[f"output.weight"], -2, -1,),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
            device=self.device
        )
        
        self.tok_embeddings = ttnn.as_tensor(
            #state_dict[f"output.weight"],
            state_dict[f"tok_embeddings.weight"], # .unsqueeze(0).unsqueeze(0),
            # torch.transpose(state_dict[f"tok_embeddings.weight"], -2, -1,),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
            device=self.device
        )

        self.torch_tok_embeddings = torch.nn.Embedding(params.vocab_size, params.dim)
        with torch.no_grad():
                self.torch_tok_embeddings.weight.copy_(state_dict['tok_embeddings.weight'])

        # some useful precompute for the RoPE relative positional embeddings
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        # (B, Seq_Len)
        _bsz, seqlen = tokens.shape
        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        # h = self.tok_embeddings(tokens)
        h = ttnn.embedding(
            tokens, 
            self.tok_embeddings,
            )
        tokens = ttnn.to_torch(tokens)

        h = self.torch_tok_embeddings(tokens)
        h = ttnn.from_torch(
             h,
             layout=ttnn.TILE_LAYOUT,
             device=self.device,
             dtype=self.dtype
        )
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)
        # inference-time mini-optimization: only forward the output on the very last position
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h = ttnn.slice(h, [0, seqlen-1, 0], [_bsz, seqlen, self.params.dim])
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT, device=self.device)

        logits = ttnn.linear(
            h,
            self.output,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
        )

        return logits