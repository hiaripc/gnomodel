import math
import time
import torch
import ttnn
from llama2.model import ModelArgs
from typing import Tuple

# just a super simple forward without host overhead
from . import LightweightModule
from . import apply_rotary_emb_host, apply_rotary_emb

class Attention(LightweightModule):
    def __init__(self, args: ModelArgs, state_dict: dict, layer_num, device, dtype=ttnn.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.max_batch_size = 1
        self.device = device
        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_q_heads = args.n_heads
        assert args.n_heads % self.n_kv_heads == 0
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads        
        # Indicates how many times the Keys and Values should be repeated        
        self.n_rep = args.n_heads // self.n_kv_heads
        
        prefix = f"layers.{layer_num}.attention."

        self.wq = ttnn.from_torch(
            torch.transpose(state_dict[f"{prefix}wq.weight"], -2, -1,),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
            device=self.device
        )
        self.wq = ttnn.tilize(self.wq)

        self.wk = ttnn.from_torch(
            torch.transpose(state_dict[f"{prefix}wk.weight"], -2, -1,),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
            device=self.device
        )
        self.wk = ttnn.tilize(self.wk)

        self.wv = ttnn.from_torch(
            torch.transpose(state_dict[f"{prefix}wv.weight"], -2, -1,),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
            device=self.device
        )
        self.wv = ttnn.tilize(self.wv)

        self.wo = ttnn.from_torch(
            torch.transpose(state_dict[f"{prefix}wo.weight"], -2, -1,),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
            device=self.device
        )
        self.wo = ttnn.tilize(self.wo)

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        self.mask = torch.triu(mask, diagonal=1).bfloat16()
        self.mask = ttnn.from_torch(
            self.mask,  
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
            device=self.device
        )
        self.mask = ttnn.tilize(self.mask)


    def repeat_kv(self, x: ttnn.Tensor, n_rep: int) -> ttnn.Tensor:
        return ttnn.repeat_interleave(x, dim=2, repeats=n_rep)


    def forward(self, x: ttnn.Tensor, freqs_cos:torch.Tensor, freqs_sin: torch.Tensor):
        bsz, seqlen, _ = x.shape

        start = time.time()
        xq = ttnn.linear(
            x,
            self.wq,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
        )
        # xq = ttnn.untilize(xq)
        xq = ttnn.to_layout(xq, layout=ttnn.ROW_MAJOR_LAYOUT)
        xq = ttnn.reshape(xq, (bsz, seqlen, self.n_q_heads, self.head_dim))

        xk = ttnn.linear(
            x,
            self.wk,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
        )
        # xk = ttnn.untilize(xk)
        xk = ttnn.to_layout(xk, layout=ttnn.ROW_MAJOR_LAYOUT)
        xk = ttnn.reshape(xk, (bsz, seqlen, self.n_kv_heads, self.head_dim))

        xv = ttnn.linear(
            x,
            self.wv,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
        )
        # xv = ttnn.untilize(xv)
        xv = ttnn.to_layout(xv, layout=ttnn.ROW_MAJOR_LAYOUT)
        xv = ttnn.reshape(xv, (bsz, seqlen, self.n_kv_heads, self.head_dim))
        # print(f"1°: {time.time() - start:.3f}")

        # start = time.time()
        # Apply RoPE
        xq, xk = apply_rotary_emb_host(xq, xk, freqs_cos, freqs_sin, self.device)

        xk = self.repeat_kv(xk, self.n_rep)
        xv = self.repeat_kv(xv, self.n_rep)

        # premute instead of transpose 
        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = ttnn.permute(xq, (0, 2, 1, 3))
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        xk = ttnn.permute(xk, (0, 2, 1, 3))
        xv = ttnn.permute(xv, (0, 2, 1, 3))
        # print(f"2°: {time.time() - start:.3f}")
        
        #start = time.time()
        
        # xq = ttnn.tilize(xq)
        # xk = ttnn.tilize(xk)
        # xv = ttnn.tilize(xv)

        xq = ttnn.to_layout(
            xq, 
            layout=ttnn.TILE_LAYOUT, 
            memory_config=ttnn.L1_MEMORY_CONFIG
        )
        xk = ttnn.to_layout(
            xk, 
            layout=ttnn.TILE_LAYOUT, 
            memory_config=ttnn.L1_MEMORY_CONFIG
        )
        xv = ttnn.to_layout(
            xv, 
            layout=ttnn.TILE_LAYOUT, 
            memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # use flash attention, shape problem
        xk = ttnn.permute(xk, (0, 1, 3, 2))
        
        # print(f"3°: {time.time() - start:.3f}")

        if False:
            output = ttnn.transformer.scaled_dot_product_attention(
                xq, 
                xk, 
                xv, 
                attn_mask=None, 
                is_causal=True
            )
        start = time.time()
        attention_scores = ttnn.matmul(
            xq,
            xk,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
            # core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )
        # attention_scores = xq @ ttnn.permute(xk, (0, 2, 1, 3))
        attention_scores = ttnn.div(attention_scores, math.sqrt(self.head_dim))
        attention_scores = attention_scores + self.mask[:, :, :seqlen, :seqlen]
        attention_scores = ttnn.softmax(attention_scores, dim=-1)

        output = ttnn.matmul(
            attention_scores,
            xv,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype
        )
        print(f"4°: {time.time() - start:.3f}")


        output = ttnn.permute(output, (0, 2, 1, 3))

        output = ttnn.to_layout(output, layout=ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.reshape(output, (bsz, seqlen, -1))
        output = ttnn.to_layout(output, layout=ttnn.TILE_LAYOUT)

        output = ttnn.linear(
            output,
            self.wo,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
        )
        # print(f"4°: {time.time() - start:.3f}")
        return output
