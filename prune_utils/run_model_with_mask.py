import torch
from typing import Tuple
import torch.nn.functional as F


# reference: medsam
def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)

# reference: medsam
def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

# reference: medsam
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

# reference: medsam
def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings reference: :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

def run_blk_with_head_mask(blk, ith_head_mask, x):
    shortcut = x
    x = blk.norm1(x)
    # Window partition
    if blk.window_size > 0:
        ori_H, ori_W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, blk.window_size)

    ########################################### Attention here! ###########################################
    B, H, W, _ = x.shape
    # qkv with shape (3, B, nHead, H * W, C)
    qkv = (
        blk.attn.qkv(x).reshape(B, H * W, 3, blk.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
    )
    # q, k, v with shape (B * nHead, H * W, C)
    q, k, v = qkv.reshape(3, B * blk.attn.num_heads, H * W, -1).unbind(0)

    attn = (q * blk.attn.scale) @ k.transpose(-2, -1)

    if blk.attn.use_rel_pos:
        attn = add_decomposed_rel_pos(
            attn, q, blk.attn.rel_pos_h, blk.attn.rel_pos_w, (H, W), (H, W)
        )
    attn = attn.softmax(dim=-1)
    # Apply head mask
    v = v * ith_head_mask.unsqueeze(0).expand(B, blk.attn.num_heads).reshape(B*blk.attn.num_heads,1,1)
    x = (
        (attn @ v)
        .view(B, blk.attn.num_heads, H, W, -1)
        .permute(0, 2, 3, 1, 4)
        .reshape(B, H, W, -1)
    )
    x = blk.attn.proj(x)
    #######################################################################################################
    # Reverse window partition
    if blk.window_size > 0:
        x = window_unpartition(x, blk.window_size, pad_hw, (ori_H, ori_W))

    x = shortcut + x
    x = x + blk.mlp(blk.norm2(x))

    return x


def run_model_with_head_mask(model, head_mask, x):
    x = model.patch_embed(x)
    if model.pos_embed is not None:
        x = x + model.pos_embed

    for i in range(len(model.blocks)):
        blk = model.blocks[i]
        ith_head_mask = head_mask[i]
        x = run_blk_with_head_mask(blk, ith_head_mask, x)

    x = model.neck(x.permute(0, 3, 1, 2))

    return x
