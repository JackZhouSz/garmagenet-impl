# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/hy3dshape/hy3dshape/models/denoisers/hunyuan3ddit.py

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from einops import rearrange
from torch import Tensor, nn

scaled_dot_product_attention = nn.functional.scaled_dot_product_attention
if os.environ.get('USE_SAGEATTN', '0') == '1':
    try:
        from sageattention import sageattn
    except ImportError:
        raise ImportError('Please install the package "sageattention" to use this USE_SAGEATTN.')
    scaled_dot_product_attention = sageattn


def attention(q: Tensor, k: Tensor, v: Tensor, attn_mask=None) -> Tensor:
    x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class GELU(nn.Module):
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.gelu(x.contiguous(), approximate=self.approximate)


class MLPEmbedder(nn.Module):
    """
    timestep enbedding
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


# class POSEmbedder(nn.Module):
#     """
#     BBOX enbedding
#     """
#     def __init__(self, pos_dim: int, hidden_dim: int):
#         super().__init__()
#
#         self.pos_dim = pos_dim
#         self.hidden_dim = hidden_dim
#
#         # 用于计算
#         if pos_dim == 10:
#             bbox_min_mask = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=torch.bool)
#         elif pos_dim == 6:
#             bbox_min_mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
#         elif pos_dim == 4:
#             bbox_min_mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
#         else:
#             raise NotImplementedError("Invalid bbox dim.")
#
#         bbox_max_mask = bbox_min_mask
#         self.register_buffer("bbox_max_mask", bbox_max_mask)
#         self.register_buffer("bbox_min_mask", bbox_min_mask)
#
#     def forward(self, p: Tensor) -> Tensor:
#         bbox_center = (x[..., self.bbox_max_mask] + x[..., self.bbox_min_mask]) / 2
#         bbox_scale = x[..., self.bbox_max_mask] - x[..., self.bbox_min_mask]
#         a=1


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor, attn_mask=None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, attn_mask=attn_mask)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(nn.functional.silu(vec))[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, attn_mask=None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            img:        LatentCode
            txt:        Condition
            vec:        Time/Position Embedding
            attn_mask:
        Returns:
        """

        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, attn_mask=attn_mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, attn_mask=None) -> Tensor:
        """
        Args:
            x:          LatentCode
            vec:        Time/Position Embedding
            attn_mask:
        Returns:
        """

        mod, _ = self.modulation(vec)

        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, attn_mask=attn_mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


# class Hunyuan3DDiT(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = 64,
#         context_in_dim: int = 1536,
#         hidden_size: int = 1024,
#         mlp_ratio: float = 4.0,
#         num_heads: int = 16,
#         depth: int = 16,
#         depth_single_blocks: int = 32,
#         axes_dim: List[int] = [64],
#         theta: int = 10_000,
#         qkv_bias: bool = True,
#         time_factor: float = 1000,
#         guidance_embed: bool = False,
#         ckpt_path: Optional[str] = None,
#         **kwargs,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.context_in_dim = context_in_dim
#         self.hidden_size = hidden_size
#         self.mlp_ratio = mlp_ratio
#         self.num_heads = num_heads
#         self.depth = depth
#         self.depth_single_blocks = depth_single_blocks
#         self.axes_dim = axes_dim
#         self.theta = theta
#         self.qkv_bias = qkv_bias
#         self.time_factor = time_factor
#         self.out_channels = self.in_channels
#         self.guidance_embed = guidance_embed
#
#         if hidden_size % num_heads != 0:
#             raise ValueError(
#                 f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
#             )
#         pe_dim = hidden_size // num_heads
#         if sum(axes_dim) != pe_dim:
#             raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.latent_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
#         self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
#         self.cond_in = nn.Linear(context_in_dim, self.hidden_size)
#         self.guidance_in = (
#             MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if guidance_embed else nn.Identity()
#         )
#
#         self.double_blocks = nn.ModuleList(
#             [
#                 DoubleStreamBlock(
#                     self.hidden_size,
#                     self.num_heads,
#                     mlp_ratio=mlp_ratio,
#                     qkv_bias=qkv_bias,
#                 )
#                 for _ in range(depth)
#             ]
#         )
#
#         self.single_blocks = nn.ModuleList(
#             [
#                 SingleStreamBlock(
#                     self.hidden_size,
#                     self.num_heads,
#                     mlp_ratio=mlp_ratio,
#                 )
#                 for _ in range(depth_single_blocks)
#             ]
#         )
#
#         self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
#
#         if ckpt_path is not None:
#             print('restored denoiser ckpt', ckpt_path)
#
#             ckpt = torch.load(ckpt_path, map_location="cpu")
#             if 'state_dict' not in ckpt:
#                 # deepspeed ckpt
#                 state_dict = {}
#                 for k in ckpt.keys():
#                     new_k = k.replace('_forward_module.', '')
#                     state_dict[new_k] = ckpt[k]
#             else:
#                 state_dict = ckpt["state_dict"]
#
#             final_state_dict = {}
#             for k, v in state_dict.items():
#                 if k.startswith('model.'):
#                     final_state_dict[k.replace('model.', '')] = v
#                 else:
#                     final_state_dict[k] = v
#             missing, unexpected = self.load_state_dict(final_state_dict, strict=False)
#             print('unexpected keys:', unexpected)
#             print('missing keys:', missing)
#
#     def forward(
#         self,
#         x,
#         t,
#         contexts,
#         **kwargs,
#     ) -> Tensor:
#         cond = contexts['main']
#         latent = self.latent_in(x)
#
#         vec = self.time_in(timestep_embedding(t, 256, self.time_factor).to(dtype=latent.dtype))
#         if self.guidance_embed:
#             guidance = kwargs.get('guidance', None)
#             if guidance is None:
#                 raise ValueError("Didn't get guidance strength for guidance distilled model.")
#             vec = vec + self.guidance_in(timestep_embedding(guidance, 256, self.time_factor))
#
#         cond = self.cond_in(cond)
#         pe = None
#
#         for block in self.double_blocks:
#             latent, cond = block(img=latent, txt=cond, vec=vec, pe=pe)
#
#         latent = torch.cat((cond, latent), 1)
#         for block in self.single_blocks:
#             latent = block(latent, vec=vec, pe=pe)
#
#         latent = latent[:, cond.shape[1]:, ...]
#         latent = self.final_layer(latent, vec)
#         return latent


class HunyuanDiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        pos_dim:int = 10,
        pos_embedding_type = "learn",  # learn、RopE
        hidden_size: int = 768,
        context_in_dim: int = 1536,
        context_seq_len = 1,  # [TODO] 无条件生成任务
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        depth_double_blocks: int = 3,
        depth_single_blocks: int = 9,
        qkv_bias: bool = True,
        time_factor: float = 1000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.pos_dim = pos_dim
        self.hidden_size = hidden_size
        self.context_in_dim = context_in_dim
        self.context_seq_len = context_seq_len
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.depth_double_blocks = depth_double_blocks
        self.depth_single_blocks = depth_single_blocks
        self.qkv_bias = qkv_bias
        self.time_factor = time_factor

        if self.context_seq_len==0 and self.depth_double_blocks:
            raise ValueError("Unconditional generation task dont need mm-dit.")

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )

        self.latent_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        if self.pos_dim>0:
            self.pos_in = MLPEmbedder(in_dim=self.pos_dim, hidden_dim=self.hidden_size)
        else:
            self.pos_in = None
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.cond_in = nn.Linear(context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth_double_blocks)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
    """
    will be used soon.
    """

    def forward(
        self,
        x,
        p,
        t,
        cond,
        attn_mask=None,
    ) -> Tensor:

        # Embedding in ===
        latent = self.latent_in(x)

        if self.pos_dim>0:
            pos_emb = self.pos_in(p)
            latent = latent + pos_emb  # 位置编码直接加到latentcode上，参考了SD3

        t_emb = self.time_in(timestep_embedding(t, 256, self.time_factor).to(dtype=latent.dtype))
        cond_emb = self.cond_in(cond)

        if attn_mask is not None:
            # MM-DIT 用的mask需要额外加一个False，对应条件的token
            assert attn_mask.shape[-1] == x.shape[1]
            attn_mask_MM = torch.concat([torch.zeros((attn_mask.shape[0], 1)).to(attn_mask), attn_mask], dim=-1)
            attn_mask_float = (~attn_mask_MM).float()  # False -> 1.0, True -> 0.0
            attn_mask_float = attn_mask_float.masked_fill(attn_mask_MM, float('-inf'))  # True -> -inf
            attn_mask_float = attn_mask_float.masked_fill(~attn_mask_MM, 0.0)  # False -> 0.0
            # attn_mask_float 现在是：[[0.0, 0.0, -inf, -inf], [0.0, 0.0, 0.0, -inf]]

            # 扩展为 [B, 1, 1, T]
            attn_mask_float = attn_mask_float.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        else:
            attn_mask_float = None

        # Main model ===
        for block in self.double_blocks:
            latent, cond_emb = block(img=latent, txt=cond_emb, vec=t_emb, attn_mask=attn_mask_float)

        latent = torch.cat((cond_emb, latent), 1)
        for block in self.single_blocks:
            latent = block(latent, vec=t_emb, attn_mask=attn_mask_float)

        latent = latent[:, cond_emb.shape[1]:, ...]
        latent = self.final_layer(latent, t_emb)
        return latent


if __name__ == '__main__':
    pos_dim = 10
    z_dim = 64

    embed_dim = 768
    condition_dim = 1536
    num_heads = 12


    B = 1
    N = 32  # latent token 数
    M = 1  # 条件 token 数

    ditss = HunyuanDiT(
        in_channels = z_dim,
        pos_dim=pos_dim,
        context_in_dim = condition_dim,
        hidden_size = embed_dim,
        num_heads = num_heads,
        depth_double_blocks= 3,
        depth_single_blocks = 9,
        mlp_ratio = 4.0,
        qkv_bias = True,
        time_factor = 1000,
    )

    # 构造输入 latent 特征
    x = torch.randn(B, N, z_dim)        # 输入图像表示或中间 latent
    p = torch.randn(B, N, pos_dim)    # 每个板片的BBOX

    # 构造时间步 t（扩散 step）
    t = torch.randint(0, 1000, (B,))  # 随机时间步，范围 [0, 1000)

    # 构造条件输入（如文本编码、语义向量）
    cond = torch.randn(B, M, condition_dim)  # 条件表示（如 CLIP 编码）

    # 哪些板片是有效的
    valid_num = torch.randint(low=(N-1)//2, high=N-1, size=(B,))
    mask = torch.ones((B,N), dtype=torch.bool)
    for i in range(len(valid_num)):
        mask[i, :valid_num[i]] = False

    output = ditss(x, p, t, cond, attn_mask=mask)
    ditss.to("cuda")
    a=1
    # 若未启用 guidance_embed，可省略 guidance 参数
    # output = model(x, t, cond)

    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        norm_first=True,
        dim_feedforward=1024,
        dropout=0.1
    )
    net = nn.TransformerEncoder(
        layer, 12, nn.LayerNorm(embed_dim))
    net.to("cuda")
    a=1