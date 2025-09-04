import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from PIL import Image
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, is_torch_version
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder

from src.models.denoisers.dit_hunyuan_2.hunyuan3ddit import HunyuanDiT

from typing import Any, Callable, Dict, List, Optional, Union


def sincos_embedding(input, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param input: a N-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim //2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) /half
    ).to(device=input.device)
    for _ in range(len(input.size())):
        freqs = freqs[None]
    args = input.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.embed.weight, mode="fan_in")

    def forward(self, x):
        return self.embed(x)

@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"


class AutoencoderKLFastEncode(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
        sample_mode: str = "sample"
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.sample_mode = sample_mode

    def forward(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if self.sample_mode == "sample":
            latent_z = DiagonalGaussianDistribution(moments).sample()
        elif self.sample_mode == "mode":    
            latent_z = DiagonalGaussianDistribution(moments).mode()  # mode converge faster
        else:
            raise ValueError(f"Invalid sample mode: {self.sample_mode}")
        return latent_z


class AutoencoderKLFastDecode(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
    ):
        super().__init__()

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        decoded = self._decode(z).sample
        return decoded    





class TextEncoder:
    def __init__(self, encoder='CLIP', device='cuda'):

        self.device = device

        if encoder == 'T5':
            import transformers
            self.tokenizer = transformers.T5TokenizerFast.from_pretrained(
                "/data/lsr/models/FLUX.1-dev", subfolder='tokenizer_2')
            text_encoder = transformers.T5EncoderModel.from_pretrained(
                "/data/lsr/models/FLUX.1-dev", subfolder='text_encoder_2')
            self.text_encoder = nn.DataParallel(text_encoder).to(device).eval()
            self.text_emb_dim = 4096
            self.text_embedder_fn = self._get_t5_text_embeds
        elif encoder == 'CLIP':
            import transformers
            self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
                "/data/lsr/models/FLUX.1-dev", subfolder='tokenizer')
            text_encoder = transformers.CLIPTextModel.from_pretrained(
                "/data/lsr/models/FLUX.1-dev", subfolder='text_encoder')
            self.text_encoder = nn.DataParallel(text_encoder).to(device).eval()
            self.text_emb_dim = 768
            self.text_embedder_fn = self._get_clip_text_embeds
        elif encoder == 'GME':
            from llm_utils.gme_inference import GmeQwen2VL
            self.text_embedder_fn = self._get_gme_text_embeds
            self.gme = GmeQwen2VL(model_path='/data/lsr/models/gme-Qwen2-VL-2B-Instruct', max_length=32)
            self.text_emb_dim = 1536
        else:
            raise ValueError(f'Unsupported encoder {encoder}.')
        
        # Test encoding text
        print(f"[DONE] Init {encoder} text encoder.")


    def _get_gme_text_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 310
    ):        
        assert hasattr(self, 'gme'), "Must initialize GME model before use."

        prompt_embeds = self.gme.get_text_embeddings(texts=prompt)
                
        return prompt_embeds
        
    
    def _get_t5_text_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 16
    ):
        if not prompt or not prompt[0]: return None
        if not hasattr(self, 'tokenizer'): return None
        if not hasattr(self, 'text_encoder'): return None
                
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        with torch.no_grad():
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )
            
            text_input_ids = text_inputs.input_ids
            prompt_embeds = self.text_encoder(text_input_ids.to(self.device), output_hidden_states=True).last_hidden_state[0]
            _, seq_len, _ = prompt_embeds.shape

        return prompt_embeds
    
    
    def _get_clip_text_embeds(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 16
    ):

        # print('*** [Text Encoder] prompt = ', prompt)

        with torch.no_grad():
            prompt = [prompt] if isinstance(prompt, str) else prompt
            batch_size = len(prompt)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            # print(text_input_ids)
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            prompt_embeds = self.text_encoder(input_ids=text_input_ids.to(self.device), output_hidden_states=False)

            # Use pooled output of CLIPTextModel
            prompt_embeds = prompt_embeds.pooler_output
            prompt_embeds = prompt_embeds.to(self.device)
            
            # print("*** CLIP prompt_embeds: ", prompt_embeds.shape, prompt_embeds.min(), prompt_embeds.max())

        return prompt_embeds

    def __call__(self, prompt):
        return self.text_embedder_fn(prompt)


class PointcloudEncoder:
    def __init__(self, encoder='POINT_E', device='cuda'):

        self.device = device

        if encoder == 'POINT_E':
            from src.models.pc_backbone.point_e.evals.feature_extractor import PointNetClassifier
            """
            cache_dir 是pretrain model的路径
            """
            self.pointcloud_emb_dim = 512
            self.pointcloud_encoder = PointNetClassifier(devices=[self.device], cache_dir='/data/lsr/models/PFID_evaluator', device_batch_size=1)
            self.pointcloud_embedder_fn = self._get_pointe_pointcloud_embeds
        else:
            raise NotImplementedError

        # Test encoding text
        print(f"[DONE] Init {encoder} text encoder.")

    def _get_pointe_pointcloud_embeds(
            self,
            point_cloud
    ):
        feature_embedding =  self.pointcloud_encoder.get_features(point_cloud)
        return feature_embedding

    def __call__(self, point_cloud):
        return self.pointcloud_embedder_fn(point_cloud)


class SketchEncoder:
    def __init__(self, encoder='LAION2B', device="cuda:0"):
        self.device = device
        if encoder == 'LAION2B':
            import timm
            from safetensors import safe_open
            from src.models.sketch_feature_extractor.vit.utils.sketch_utils import _transform

            image_resolution = 224
            self.img_process = _transform(image_resolution)

            self.sketch_emb_dim = 1280
            VIT_MODEL = 'vit_huge_patch14_224_clip_laion2b'
            # 使用timm创建模型架构
            safetensors_path = '/data/lsr/models/models--timm--vit_huge_patch14_clip_224.laion2b/snapshots/b8441fa3f968a5e469c166176ee82f8ce8dbc4eb/model.safetensors'
            vit_model = timm.create_model(VIT_MODEL, pretrained=False).to(self.device)
            vit_model.eval()

            # 加载 safetensors 权重
            with safe_open(safetensors_path, framework="pt") as f:
                # safetensors 返回一个字典，包含所有张量
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
                vit_model.load_state_dict(state_dict)
            self.sketch_encoder = vit_model
            self.sketch_embedder_fn = self._get_laion2b_sketch_embeds
        elif encoder == 'RADIO_V2.5-G':
            self.sketch_emb_dim = 1536
            self.sketch_encoder = None
            self.sketch_embedder_fn = None
        elif encoder == 'RADIO_V2.5-H':
            self.sketch_emb_dim = 3840
            self.sketch_encoder = None
            self.sketch_embedder_fn = None
        else:
            raise NotImplementedError
        print(f"[DONE] Init {encoder} sketch encoder.")

    def _get_laion2b_sketch_embeds(
            self,
            sketch_fp
    ):
        image = Image.open(sketch_fp).convert('RGB')
        image =  self.img_process(image)
        image = image.to(self.device).unsqueeze(0)
        sketch_features = self.sketch_encoder.forward_features(image).squeeze()
        sketch_features = sketch_features[0].unsqueeze(0)
        return sketch_features

    def __call__(self, sketch_fp):
        return self.sketch_embedder_fn(sketch_fp)


class SurfPosNet(nn.Module):
    """
    Transformer-based latent diffusion model for surface position
    """

    def __init__(self, p_dim=6, embed_dim=768, condition_dim=-1, num_cf=-1):
        super(SurfPosNet, self).__init__()
        
        self.p_dim = p_dim
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        self.use_cf = num_cf > 0

        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=12,
            norm_first=True,
            dim_feedforward=1024,
            dropout=0.1
        )
        
        self.net = nn.TransformerEncoder(layer, 12, nn.LayerNorm(self.embed_dim))

        self.p_embed = nn.Sequential(
            nn.Linear(self.p_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        ) 

        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        
        if self.use_cf: self.class_embed = Embedder(num_cf, self.embed_dim)
        if self.condition_dim > 0: self.cond_embed = nn.Linear(self.condition_dim, self.embed_dim, bias=False)
        
        self.fc_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.p_dim),
        )

    def forward(self, surfPos, timesteps, class_label=None, condition=None, is_train=False):
        """ forward pass """
        bsz, seq_len, _ = surfPos.shape
        bsz = timesteps.size(0)
        time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)  
        p_embeds = self.p_embed(surfPos)   
        
        tokens = p_embeds + time_embeds
                    
        if self.use_cf and class_label is not None:  # classifier-free
            if is_train:
                uncond_mask = torch.rand(bsz, seq_len, 1) <= 0.1  
                class_label[uncond_mask] = 0
            c_embeds = self.class_embed(class_label.squeeze(-1))
            c_embeds = c_embeds.unsqueeze(1)
            c_embeds = c_embeds.repeat((1, seq_len, 1))
            tokens += c_embeds

        if self.condition_dim > 0 and condition is not None:
            cond_token = self.cond_embed(condition)
            if len(cond_token.shape) == 2: tokens = tokens + cond_token[:, None]
            else: tokens = torch.cat([tokens, cond_embeds], dim=1)

        output = self.net(src=tokens.permute(1,0,2))  # 输入输出都是：(seq_len, batch_size, d_model)
        output = output[:seq_len].transpose(0,1)
        pred = self.fc_out(output)

        return pred


class SurfPosNet_hunyuandit(nn.Module):
    # def __init__(self, p_dim=6, z_dim=3 * 4 * 4, out_dim=-1, embed_dim=768, num_heads=12, condition_dim=-1, num_layer=[3, 9], num_cf=-1):
    def __init__(
            self,
            p_dim=6,
            embed_dim=768,
            condition_dim=-1,
            num_cf=-1,
            num_heads = 12,
            num_layer = [3,9]
        ):
        super(SurfPosNet_hunyuandit, self).__init__()
        self.p_dim = -1
        self.z_dim = p_dim
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        # self.out_dim = out_dim
        # self.use_cf = num_cf > 0
        self.num_heads = num_heads
        self.num_layer = num_layer

        if not isinstance(self.num_layer, List):
            raise TypeError("Type of num_layer should be list.")
        if len(self.num_layer) != 2:
            raise ValueError("Length of num_layer should be 2.")

        self.net = HunyuanDiT(
            in_channels=self.z_dim,
            pos_dim=self.p_dim,
            context_in_dim=self.condition_dim,
            hidden_size=self.embed_dim,
            num_heads=self.num_heads,
            depth_double_blocks=self.num_layer[0],
            depth_single_blocks=self.num_layer[1],
            mlp_ratio=4.0,
            qkv_bias=True,
            time_factor=1000,
        )

    def forward(
            self,
            surfPos,
            timesteps,
            class_label=None,
            condition=None,
            is_train=False
        ):

        if self.condition_dim > 0 and condition is not None:
            if len(condition.shape) == 2:
                condition = condition.unsqueeze(1)
            elif len(condition.shape) == 3:
                pass
            else:
                raise NotImplementedError

        output = self.net(
            x=surfPos,
            p=None,
            t=timesteps,
            cond=condition,
            attn_mask=None
        )

        return output


class SurfZNet(nn.Module):
    """
    Transformer-based latent diffusion model for surface position
    """
    def __init__(self, p_dim=6, z_dim=3*4*4, out_dim=-1, z_projector_dim=-1, embed_dim=768, num_heads=12, condition_dim=-1, num_layer=12, num_cf=-1):
        super(SurfZNet, self).__init__()
        self.p_dim = p_dim
        self.z_dim = z_dim
        self.z_projector_dim = z_projector_dim
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        self.out_dim = out_dim
        self.use_cf = num_cf > 0
        self.n_heads = num_heads
        if isinstance(num_layer, List):
            if len(num_layer) != 1:
                raise ValueError("Length of num_layer should be 1.")
            num_layer = num_layer[0]
        self.num_layer = num_layer


        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=self.n_heads,
            norm_first=True,
            dim_feedforward=1024, 
            dropout=0.1
            )

        self.net = nn.TransformerEncoder(
            layer, self.num_layer, nn.LayerNorm(self.embed_dim))

        if z_projector_dim<0:
            self.z_embed = nn.Sequential(
                nn.Linear(self.z_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
        else:
            self.z_embed = nn.Sequential(
                nn.Linear(self.z_dim, self.z_projector_dim),
                # nn.SiLU(),
                nn.Linear(self.z_projector_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )

        self.p_embed = nn.Sequential(
            nn.Linear(self.p_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        ) 

        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        if self.use_cf: self.class_embed = Embedder(num_cf, self.embed_dim)
        if self.condition_dim > 0: 
            self.cond_embed = nn.Sequential(
                nn.Linear(self.condition_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )

        self.fc_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.z_dim) if self.out_dim<0 else nn.Linear(self.embed_dim, self.out_dim) ,
        )

        return

       
    def forward(self, surfZ, timesteps, surfPos, surf_mask, class_label, condition=None, is_train=False):
        """ forward pass """
        bsz = timesteps.size(0)

        time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        z_embeds = self.z_embed(surfZ) 
        p_embeds = self.p_embed(surfPos)

        tokens = z_embeds + p_embeds + time_embeds

        # if self.use_cf and class_label is not None:  # classifier-free
        #     if is_train:
        #         uncond_mask = torch.rand(bsz, seq_len, 1) <= 0.1
        #         class_label[uncond_mask] = 0
        #     c_embeds = self.class_embed(class_label.squeeze(-1))
        #     tokens += c_embeds
            
        if self.condition_dim > 0 and condition is not None:
            cond_token = self.cond_embed(condition)
            if len(cond_token.shape) == 2:
                tokens = tokens + cond_token[:, None]
            elif len(cond_token.shape) == 3:
                tokens = tokens + cond_token  # [B, n_surfs, emb_dim]
            else:
                raise NotImplementedError
        output = self.net(
            src=tokens.permute(1,0,2),
            src_key_padding_mask=surf_mask,
        ).transpose(0,1)

        pred = self.fc_out(output)
        return pred


class SurfZNet_hunyuandit(nn.Module):
    def __init__(self, p_dim=6, z_dim=3 * 4 * 4, out_dim=-1, embed_dim=768, num_heads=12, condition_dim=-1, num_layer=[3,9], num_cf=-1):
        super(SurfZNet_hunyuandit, self).__init__()
        self.p_dim = p_dim
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        self.out_dim = out_dim
        # self.use_cf = num_cf > 0
        self.num_heads = num_heads

        if not isinstance(num_layer, List):
            raise TypeError("Type of num_layer should be list.")
        if len(num_layer) != 2:
            raise ValueError("Length of num_layer should be 2.")
        self.num_layer = num_layer
        self.net = HunyuanDiT(
            in_channels=self.z_dim,
            pos_dim=self.p_dim,
            context_in_dim=self.condition_dim,
            hidden_size=self.embed_dim,
            num_heads=self.num_heads,
            depth_double_blocks=self.num_layer[0],
            depth_single_blocks=self.num_layer[1],
            mlp_ratio=4.0,
            qkv_bias=True,
            time_factor=1000,
        )

    def forward(self, surfZ, timesteps, surfPos, surf_mask, class_label, condition=None, is_train=False):
        """ forward pass """
        # bsz = timesteps.size(0)

        # time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        # z_embeds = self.z_embed(surfZ)
        # p_embeds = self.p_embed(surfPos)
        #
        # tokens = z_embeds + p_embeds + time_embeds

        if self.condition_dim > 0 and condition is not None:
            if len(condition.shape) == 2:
                condition = condition.unsqueeze(1)
            elif len(condition.shape) == 3:
                pass
            else:
                raise NotImplementedError

        output = self.net(
            x=surfZ,
            p=surfPos,
            t=timesteps,
            cond=condition,
            attn_mask=surf_mask
        )

        # pred = self.fc_out(output)
        return output



# class SurfImpaintingNet(nn.Module):
#     """
#     Transformer-based latent diffusion model for surface position
#     """
#
#     def __init__(self,
#             p_dim=6,
#             z_dim=256,
#             embed_dim=768,
#             num_heads=12,
#             condition_dim=-1,
#             num_cf=-1
#         ):
#         super(SurfImpaintingNet, self).__init__()
#         self.p_dim = p_dim
#         self.z_dim = z_dim
#         self.embed_dim = embed_dim
#         self.condition_dim = condition_dim
#
#         self.use_cf = num_cf > 0
#         self.n_heads = num_heads
#
#         layer = nn.TransformerEncoderLayer(
#             d_model=self.embed_dim,
#             nhead=self.n_heads,
#             norm_first=True,
#             dim_feedforward=1024,
#             dropout=0.1
#         )
#
#         self.net = nn.TransformerEncoder(
#             layer, 12, nn.LayerNorm(self.embed_dim))
#
#         self.z_embed = nn.Sequential(
#             nn.Linear(self.z_dim, self.embed_dim),
#             nn.LayerNorm(self.embed_dim),
#             nn.SiLU(),
#             nn.Linear(self.embed_dim, self.embed_dim),
#         )
#
#         self.p_embed = nn.Sequential(
#             nn.Linear(self.p_dim, self.embed_dim),
#             nn.LayerNorm(self.embed_dim),
#             nn.SiLU(),
#             nn.Linear(self.embed_dim, self.embed_dim),
#         )
#
#         self.time_embed = nn.Sequential(
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.LayerNorm(self.embed_dim),
#             nn.SiLU(),
#             nn.Linear(self.embed_dim, self.embed_dim),
#         )
#
#         if self.use_cf: self.class_embed = Embedder(num_cf, self.embed_dim)
#         if self.condition_dim > 0:
#             self.cond_embed = nn.Sequential(
#                 nn.Linear(self.condition_dim, self.embed_dim),
#                 nn.LayerNorm(self.embed_dim),
#                 nn.SiLU(),
#                 nn.Linear(self.embed_dim, self.embed_dim),
#             )
#
#         self.fc_out = nn.Sequential(
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.LayerNorm(self.embed_dim),
#             nn.SiLU(),
#             nn.Linear(self.embed_dim, self.z_dim),
#         )
#
#         return
#
#     def forward(self, surfZ, surfPos, timesteps, surf_mask, class_label=None, condition=None, is_train=False):
#         """ forward pass """
#         bsz = timesteps.size(0)
#
#         time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)
#         z_embeds = self.z_embed(surfZ)
#         p_embeds = self.p_embed(surfPos)
#
#         tokens = z_embeds + p_embeds + time_embeds
#
#         if self.use_cf and class_label is not None:  # classifier-free
#             if is_train:
#                 uncond_mask = torch.rand(bsz, seq_len, 1) <= 0.1
#                 class_label[uncond_mask] = 0
#             c_embeds = self.class_embed(class_label.squeeze(-1))
#             tokens += c_embeds
#
#         if self.condition_dim > 0 and condition is not None:
#             cond_token = self.cond_embed(condition)
#             if len(cond_token.shape) == 2:
#                 tokens = tokens + cond_token[:, None]
#             else:
#                 tokens = torch.cat([cond_embeds, tokens], dim=1)
#
#         output = self.net(
#             src=tokens.permute(1, 0, 2),
#             src_key_padding_mask=surf_mask,
#         ).transpose(0, 1)
#
#         # print('[SurfZNet] token', tokens.size(), tokens.min(), tokens.max())
#         # print('[SurfZNet] mask', surf_mask.size(), surf_mask.sum(dim=1).min(), surf_mask.sum(dim=1).max())
#         # print('[SurfZNet] output', output.size(), output.min(), output.max())
#
#         pred = self.fc_out(output)
#         return pred
#
#     # def forward(self, surfZ, timesteps, surfPos, surf_mask, class_label, condition=None, is_train=False):
#     #     """ forward pass """
#     #     bsz = timesteps.size(0)
#     #
#     #     time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)
#     #     z_embeds = self.z_embed(surfZ)
#     #     p_embeds = self.p_embed(surfPos)
#     #
#     #     tokens = z_embeds + p_embeds + time_embeds
#     #
#     #     if self.use_cf and class_label is not None:  # classifier-free
#     #         if is_train:
#     #             uncond_mask = torch.rand(bsz, seq_len, 1) <= 0.1
#     #             class_label[uncond_mask] = 0
#     #         c_embeds = self.class_embed(class_label.squeeze(-1))
#     #         tokens += c_embeds
#     #
#     #     if self.condition_dim > 0 and condition is not None:
#     #         cond_token = self.cond_embed(condition)
#     #         if len(cond_token.shape) == 2:
#     #             tokens = tokens + cond_token[:, None]
#     #         else:
#     #             tokens = torch.cat([cond_embeds, tokens], dim=1)
#     #
#     #     output = self.net(
#     #         src=tokens.permute(1, 0, 2),
#     #         src_key_padding_mask=surf_mask,
#     #     ).transpose(0, 1)
#     #
#     #     # print('[SurfZNet] token', tokens.size(), tokens.min(), tokens.max())
#     #     # print('[SurfZNet] mask', surf_mask.size(), surf_mask.sum(dim=1).min(), surf_mask.sum(dim=1).max())
#     #     # print('[SurfZNet] output', output.size(), output.min(), output.max())
#     #
#     #     pred = self.fc_out(output)
#     #     return pred