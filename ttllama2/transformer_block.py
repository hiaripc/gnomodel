from llama2.model import ModelArgs
from . import LightweightModule
from . import AttentionFaster as Attention
from . import RMSNorm
from . import FeedForward

class TransformerBlock(LightweightModule):
    def __init__(self, layer_id: int, args: ModelArgs, state_dict, device):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, state_dict, layer_id, device)
        self.feed_forward = FeedForward(
            args.dim,
            args.hidden_dim,
            args.multiple_of,
            layer_id,
            state_dict,
            device
        )
        self.layer_id = layer_id
        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(device, args.dim, args.norm_eps, state_dict, f"{layer_id}.attention_norm.")
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(device, args.dim, args.norm_eps, state_dict, f"{layer_id}.ffn_norm.")

    def forward(self, x, freqs_cos, freqs_sin):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        norm = self.ffn_norm(h)
        ff = self.feed_forward.forward(norm)
        out = h + ff
        return out