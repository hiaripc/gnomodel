import math
import time
import torch
import ttnn
from llama2.model import ModelArgs
from typing import Tuple

# just a super simple forward without host overhead
from . import LightweightModule
from . import apply_rotary_emb_host, apply_rotary_emb
from ttnn.model_preprocessing import (
    preprocess_linear_bias,
    preprocess_linear_weight,
)

class Attention(LightweightModule):
    def __init__(self, args: ModelArgs, state_dict: dict, layer_num, device, dtype=ttnn.bfloat16, use_sdpa=True):
        super().__init__()
        self.use_sdpa = use_sdpa
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
            device=self.device
        )
        self.wq = ttnn.tilize(self.wq)

        self.wk = ttnn.from_torch(
            torch.transpose(state_dict[f"{prefix}wk.weight"], -2, -1,),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
            device=self.device
        )
        self.wk = ttnn.tilize(self.wk)

        self.wv = ttnn.from_torch(
            torch.transpose(state_dict[f"{prefix}wv.weight"], -2, -1,),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
            device=self.device
        )
        self.wv = ttnn.tilize(self.wv)

        self.wo = ttnn.from_torch(
            torch.transpose(state_dict[f"{prefix}wo.weight"], -2, -1,),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
            device=self.device
        )
        self.wo = ttnn.tilize(self.wo)
        
        # torch_qkv_weight = torch.cat([
        #     torch.transpose(state_dict[f"{prefix}wq.weight"], -2, -1,), 
        #     torch.transpose(state_dict[f"{prefix}wk.weight"], -2, -1,), 
        #     torch.transpose(state_dict[f"{prefix}wv.weight"], -2, -1,)], dim=-1)

        # self.qkv_weight = preprocess_linear_weight(torch_qkv_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        # self.qkv_weight = ttnn.to_device(self.qkv_weight, self.device)

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

        fallback_reshape = ttnn.get_fallback_function(ttnn.reshape)


        # Apply weights
        # start = time.time()
        """
        # print(x.shape)
        # print(self.qkv_weight.shape)
        # print(self.wq.shape)
        fused_qkv_output = ttnn.linear(
            x,
            # self.wq,
            self.qkv_weight,
            bias=None,
            # memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
            # device=self.device
            # core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )
        # print(f"Fused: {time.time() - start:.3f}")

        # Fatal: Unsupported input shape
        xq, xk, xv = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv_output,
            # memory_config=ttnn.L1_MEMORY_CONFIG,
            num_heads=self.n_q_heads,
            num_kv_heads=self.n_kv_heads,
            transpose_key=False
        )
        """
        xq = ttnn.linear(
            x,
            self.wq,
            bias=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.dtype,
        )
        
        xk = ttnn.linear(
            x,
            self.wk,
            bias=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.dtype,
        )

        xv = ttnn.linear(
            x,
            self.wv,
            bias=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.dtype,
        )
        # print(f"Matmuls: {time.time() - start:.3f}")

        # Reshape
        # start = time.time()
        xq = ttnn.to_layout(xq, layout=ttnn.ROW_MAJOR_LAYOUT)
        xq = fallback_reshape(xq, (bsz, seqlen, self.n_q_heads, self.head_dim))

        xk = ttnn.to_layout(xk, layout=ttnn.ROW_MAJOR_LAYOUT)
        xk = fallback_reshape(xk, (bsz, seqlen, self.n_kv_heads, self.head_dim))

        xv = ttnn.to_layout(xv, layout=ttnn.ROW_MAJOR_LAYOUT)
        xv = fallback_reshape(xv, (bsz, seqlen, self.n_kv_heads, self.head_dim))
        # print(f"Reshapes: {time.time() - start:.3f}")

        # freqs_cos = ttnn.from_torch(freqs_cos, layout=ttnn.TILE_LAYOUT)
        # freqs_sin = ttnn.from_torch(freqs_sin, layout=ttnn.TILE_LAYOUT)

        # Apply RoPE
        # start = time.time()
        xq, xk = apply_rotary_emb_host(xq, xk, freqs_cos, freqs_sin, self.device)
        # xq = ttnn.experimental.rotary_embedding_llama(xq, cos_cache=freqs_cos, sin_cache=freqs_sin)
        # xk = ttnn.experimental.rotary_embedding_llama(xk, cos_cache=freqs_cos, sin_cache=freqs_sin)
        # print(f"Rotatory emb: {time.time() - start:.3f}")

        # start = time.time()
        xk = self.repeat_kv(xk, self.n_rep)
        xv = self.repeat_kv(xv, self.n_rep)

        # To use SPDA, must be in DRAM
        xq = ttnn.to_layout(xq, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xk = ttnn.to_layout(xk, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xv = ttnn.to_layout(xv, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # premute instead of transpose 
        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = ttnn.permute(xq, (0, 2, 1, 3))
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        xk = ttnn.permute(xk, (0, 2, 1, 3))
        xv = ttnn.permute(xv, (0, 2, 1, 3))
        
        # print(f"Permutations: {time.time() - start:.3f}")

        if self.use_sdpa:
            # start = time.time()
            output = ttnn.transformer.scaled_dot_product_attention(
                xq, 
                xk, 
                xv, 
                attn_mask=None, 
                is_causal=True
            )
            # print(f"SDPA: {time.time() - start:.3f}")
        
        else: 
            # start = time.time()
            attention_scores = ttnn.matmul(
                xq,
                xk,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.dtype,
                # core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
            )
            # attention_scores = xq @ ttnn.permute(xk, (0, 2, 1, 3))
            attention_scores = ttnn.div(attention_scores, math.sqrt(self.head_dim))
            attention_scores = attention_scores + self.mask[:, :, :seqlen, :seqlen]
            attention_scores = ttnn.softmax(attention_scores, dim=-1)
            # print(f"Attention scores: {time.time() - start:.3f}")

            # start = time.time()
            output = ttnn.matmul(
                attention_scores,
                xv,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.dtype
            )

        # start = time.time()
        output = ttnn.permute(output, (0, 2, 1, 3))
        output = ttnn.to_layout(output, layout=ttnn.ROW_MAJOR_LAYOUT)
        output = fallback_reshape(output, (bsz, seqlen, -1))
        output = ttnn.to_layout(output, layout=ttnn.TILE_LAYOUT)
        # print(f"Output reshape: {time.time() - start:.3f}")

        # start = time.time()
        output = ttnn.linear(
            output,
            self.wo,
            bias=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.dtype,
        )
        # print(f"Output mm: {time.time() - start:.3f}")
        return output
