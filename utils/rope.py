import os
import sys
import torch

from torch import nn


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))

## updated from "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L198" ##

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    cos = cos.to(device=q.device)
    sin = sin.to(device=q.device)
    q, k = q.float(), k.float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class VisionEmbedding(nn.Module):
    def __init__(self, dim: int, temperature: float=10000.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inv_freq = 1.0 / (temperature ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return torch.repeat_interleave(torch.cos(freqs), repeats=2, dim=-1), torch.repeat_interleave(torch.sin(freqs), repeats=2, dim=-1)
    
class VisionROPE(nn.Module):
    def __init__(self, dim: int, H: int, W: int, temperature: float=10000.0, *args, **kwargs):
        """
        dim: half dimension of initial dimension(channels), spliting channels in 2D-ROPE-fashion
        """
        super().__init__(*args, **kwargs)
        self.vision_embedding = VisionEmbedding(dim, temperature)
        self.w_cos, self.w_sin = self.vision_embedding(W)
        self.h_cos, self.h_sin = self.vision_embedding(H)
        self.dim = dim

    def forward(self, q, k):
        """
        q, k: [B, num_head, head_dim, H, W]
        """
        q_w, q_h = torch.split(q, self.dim, dim=2)
        k_w, k_h = torch.split(k, self.dim, dim=2)
        # [W, D] -> [D, 1, W]
        self.w_cos, self.w_sin = self.w_cos.transpose(1, 0).unsqueeze(1), self.w_sin.transpose(1, 0).unsqueeze(1)
        # [H, D] -> [D, H, 1]
        self.h_cos, self.h_sin = self.h_cos.transpose(1, 0).unsqueeze(-1), self.h_sin.transpose(1, 0).unsqueeze(-1)
        q_w_embeded, k_w_embeded = apply_rotary_pos_emb_vision(q_w, k_w, self.w_cos, self.w_sin)
        q_h_embeded, k_h_embeded = apply_rotary_pos_emb_vision(q_h, k_h, self.h_cos, self.h_sin)
        q_embeded = torch.cat([q_w_embeded, q_h_embeded], dim=2)
        k_embeded = torch.cat([k_w_embeded, k_h_embeded], dim=2)
        return q_embeded.to(device=q.device), k_embeded.to(device=k.device)


if __name__=="__main__":
    q = torch.randn((8, 4, 16, 32, 32))
    k = torch.randn((8, 4, 16, 32, 32))

    rope_embedding = VisionEmbedding(8)
    vision_rope = VisionROPE(8, 32, 32)
    q_embeded, k_embeded = vision_rope(q, k)
    print(q_embeded.shape)