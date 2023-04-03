import torch
import torch.nn as nn
class QuantRotaryEmb(torch.nn.Module):
    def __init__(self, org_module,head_dim,num_heads):
        super().__init__()
        self.ori_cos_cached = org_module.cos_cached
        self.ori_sin_cached = org_module.sin_cached
        self.reorderd_cos_cached = None
        self.reorderd_sin_cached = None
        self.num_heads = num_heads
        self.head_dim = head_dim


    def reorderd(self,index):
        bsz, _, q_len, _ = self.ori_cos_cached.shape
        # breakpoint()

        self.reorderd_cos_cached = torch.index_select(self.ori_cos_cached.repeat(1,self.num_heads,1,1).contiguous().transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim
        ), 2, index)
        self.reorderd_cos_cached =self.reorderd_cos_cached.reshape(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)


        self.reorderd_sin_cached = torch.index_select(self.ori_sin_cached.repeat(1,self.num_heads,1,1).contiguous().transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim
        ), 2, index)
        self.reorderd_sin_cached =self.reorderd_sin_cached.reshape(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)


    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    def forward(self, x, seq_len=None, q=None, k=None, offset: int = 0):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.


        if self.reorderd_cos_cached is None:

            cos=self.ori_cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
            sin=self.ori_sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)

        else:

            cos=self.reorderd_cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
            sin=self.reorderd_sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)


        cos = cos[..., offset : q.shape[-2] + offset, :]
        sin = sin[..., offset : q.shape[-2] + offset, :]

        x1 = q[..., : q.shape[-1] // 2]
        x2 = q[..., q.shape[-1] // 2 :]
        rotate_half_q = torch.cat((-x2, x1), dim=-1)


        x1 = k[..., : k.shape[-1] // 2]
        x2 = k[..., k.shape[-1] // 2 :]
        rotate_half_k = torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half_q * sin)
        k_embed = (k * cos) + (rotate_half_k * sin)

        return q_embed, k_embed


