import torch
import ttnn
from typing import Tuple
from llama2.model import apply_rotary_emb as apply_rotary_emb_torch


def reshape_for_broadcast(freqs_cis: ttnn.Tensor, x: ttnn.Tensor):
    ndim = len(x.shape)
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) 
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return ttnn.reshape(freqs_cis, shape)


def apply_rotary_emb_host(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    device,
    dtype=ttnn.bfloat16
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # xq = ttnn.to_layout(xq, ttnn.ROW_MAJOR_LAYOUT)
    # xk = ttnn.to_layout(xk, ttnn.ROW_MAJOR_LAYOUT)

    # xq = ttnn.untilize(xq)
    # xk = ttnn.untilize(xk)

    xq = ttnn.to_torch(xq)
    xk = ttnn.to_torch(xk)

    xq_out, xk_out = apply_rotary_emb_torch(xq, xk, freqs_cos, freqs_sin)

    xq_out = ttnn.from_torch(xq_out, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    xk_out = ttnn.from_torch(xk_out, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    return xq_out, xk_out


def apply_rotary_emb(
    xq: ttnn.Tensor,
    xk: ttnn.Tensor,
    freqs_cos: ttnn.Tensor,
    freqs_sin: ttnn.Tensor
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    print("xq:", xq.shape)
    assert tuple(xq.shape)[0] == 1, "Only works with batch 1 :-C"
    xq = ttnn.reshape(xq, (tuple(xq.shape)[:-1] + (-1,2)))
    # Cannot unbind, cannot slice with [:..], must use ttnn.slice
    # xq_r, xq_i = xq.unbind(-1)
    # Squeeze because to_layour only supports 4D max tensor
    xq = ttnn.squeeze(xq, 0)
    xq = ttnn.to_layout(xq, layout = ttnn.ROW_MAJOR_LAYOUT)
    xq = ttnn.unsqueeze(xq, 0)
    xq_r = ttnn.slice(xq, [0,0,0,0,0], list(tuple(xq.shape)[:-1] + (1,)))
    xq_r = ttnn.squeeze(xq_r, -1)
    print("xq_r:", xq_r.shape)
    # ttnn.deallocate(xq_r)
    xq_i = ttnn.slice(xq, [0,0,0,0,1], list(tuple(xq.shape)[:-1] + (2,)))
    xq_i = ttnn.squeeze(xq_i, -1)    
    print("xq_i:", xq_i.shape)

    xk = ttnn.squeeze(xk, 0)
    xk = ttnn.to_layout(xk, layout = ttnn.ROW_MAJOR_LAYOUT)
    xk = ttnn.unsqueeze(xk, 0)
    xk = ttnn.unsqueeze(xk, 0)
    xk_r = ttnn.slice(xk, [0,0,0,0,0],tuple(xk.shape)[:-1] + (1,))
    xk_r = ttnn.squeeze(xk_r, -1)
    # ttnn.deallocate(xk_r)
    xk_i = ttnn.slice(xk, [0,0,0,0,1], tuple(xk.shape)[:-1] + (2,))
    xk_i = ttnn.squeeze(xk_i, -1)  

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    freqs_sin = ttnn.to_layout(freqs_sin, layout = ttnn.TILE_LAYOUT)
    freqs_cos = ttnn.to_layout(freqs_cos, layout = ttnn.TILE_LAYOUT)

    # apply rotation using real numbers
    xq_r = ttnn.to_layout(xq_r, layout = ttnn.TILE_LAYOUT)
    xq_i = ttnn.to_layout(xq_i, layout = ttnn.TILE_LAYOUT)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos

    xk_r = ttnn.to_layout(xk_r, layout = ttnn.TILE_LAYOUT)
    xk_i = ttnn.to_layout(xk_i, layout = ttnn.TILE_LAYOUT)
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    
    # there's no ttnn.stack nor ttnn.flatten :-)
    xq_out_r = ttnn.to_layout(xq_out_r, layout = ttnn.ROW_MAJOR_LAYOUT)
    xq_out_i = ttnn.to_layout(xq_out_i, layout = ttnn.ROW_MAJOR_LAYOUT)
    print(xq_out_r.shape, xq_out_i.shape)
    # Create new dimension
    xq_out_r = ttnn.unsqueeze(xq_out_r, -1)
    # Concatenate along the new dimension
    xq_out = torch.concatenate([xq_out_r, xq_out_i], dim=-1)
    # todo: implement flatten
    print(xq_out.shape)

    # xq_out = ttnn.flatten(xq_out, 3)
    return 