# RoPE implementation
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device=0):
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    # freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    a = torch.arange(0, dim, 2)[: (dim // 2)].float()
    t = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    e = ttnn.div(t, dim)
    # freqs = 1.0 / (theta ** e)
    freqs = 1 /  torch.pow(theta, torch.div(a, dim))
    # 
    freqs = ttnn.pow(theta, e)
    freqs = ttnn.div(1, freqs)
    print(a)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    t = ttnn.arange(end, device=freqs.device)  # type: ignore
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    # freqs = ttnn.outer(t, freqs).float()  # type: ignore
    freqs = ttnn.outer(t, freqs)  # type: ignore
    
    # We can computer the cos and sin
    # Shape: (Seq_len, Head_Dim / 2)
    freqs_cos = ttnn.cos(freqs)  # real part
    freqs_sin = ttnn.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: ttnn.Tensor, x: ttnn.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: ttnn.Tensor,
    xk: ttnn.Tensor,
    freqs_cos: ttnn.Tensor,
    freqs_sin: ttnn.Tensor
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:

    # reshape xq and xk to match the complex representation
    # xq.shape[:-1] + (-1, 2) -> tolgo ultima dim, e faccio reshape che abbia come ultima dim 2
    # il -1 è per fare il modo che sia compatibile: (5,1,8) -> .reshape(5,1,-1,2) -> (5,1,4,2) 
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos

    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


device = ttnn.open_device(device_id=0)
precompute_freqs_cis(4, 4, device=device)