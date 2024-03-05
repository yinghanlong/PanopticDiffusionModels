import torch
import torch.nn as nn
import math
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#TODO: add cross attention between images and panoptic masks
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim , bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim , bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, L, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = einops.rearrange(q, 'B L ( H D) -> B H L D', H=self.num_heads)
        k = einops.rearrange(k, 'B L ( H D) -> B H L D', H=self.num_heads)
        v = einops.rearrange(v, 'B L ( H D) -> B H L D', H=self.num_heads)
        if ATTENTION_MODE == 'flash':
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False, enable_panoptic=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        #TODO: add an image self attention without mask, set enable_panoptic to True
        self.enable_panoptic=enable_panoptic
        if enable_panoptic==True:
            #TODO: Try cross attention for each other
            #self.attn_image = CrossAttention(
            #    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
            self.attn_image = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
            self.norm_image= norm_layer(dim)

            self.attn_mask = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
            self.norm_mask= norm_layer(dim)
            #self.mlp_mask = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
            #self.norm2_mask = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None, extra_dims=0, use_ground_truth=False, enable_panoptic=False):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip, extra_dims, use_ground_truth=use_ground_truth,enable_panoptic=enable_panoptic)
        else:
            return self._forward(x, skip,use_ground_truth=use_ground_truth,enable_panoptic=enable_panoptic)

    def _forward(self, x, skip=None, extra_dims=0, use_ground_truth=False, enable_panoptic=False):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        x = x + self.attn(self.norm1(x))
        #TODO: separate image/panoptic segmentation mask attention
        if enable_panoptic and use_ground_truth: #only activate extra attn blocks for phase two 
            B, L, D = x.shape
            len_image= extra_dims+ (L- extra_dims)//2             
            
            #TODO:try add extra feauture (text/time embedding) to masks
            extra_feat= x[:,:extra_dims,:]
            mask_tensor= x[:,len_image:,:]
            image_norm =self.norm_image(x[:,:len_image,:])
            mask_norm= self.norm_mask( torch.cat((extra_feat,mask_tensor), dim=1))
            #TODO: try cross-attention between image and mask
            #image_attn= self.attn_image( image_norm, mask_norm)
            #Use self attention for image
            image_attn= self.attn_image( image_norm)
            mask_attn= self.attn_mask( mask_norm)
            #mask_attn= self.attn_mask(  self.norm_mask( torch.cat((extra_feat,mask_tensor), dim=1)))

            #TODO: separate MLP for panoptic masks and images
            #image_attn =  self.mlp(self.norm2(image_attn))
            #mask_attn =  self.mlp_mask(self.norm2_mask(mask_attn))

            #average of extra feature outputs from both networks
            extra_attn= (image_attn[:,:extra_dims,:]+mask_attn[:,:extra_dims,:])/2.0

            all_attn= torch.cat((extra_attn, image_attn[:,extra_dims:,:],mask_attn[:,extra_dims:,:]), dim=1)
            '''

            mask_tensor= x[:,len_image:,:]
            image_attn= self.attn_image( self.norm_image(x[:,:len_image,:]))
            mask_attn= self.attn_mask( self.norm_mask(mask_tensor))
            all_attn= torch.cat((image_attn,mask_attn), dim=1)
            '''

            #residual connection
            #x = x + all_attn

            x = x + self.mlp(self.norm2(all_attn))
        else:
            x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        if len(x.shape)==3:
            x=torch.unsqueeze(x,1)
        B, C, H, W = x.shape
        #print(x.shape)
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2) #output shape= B L C, L=patch_h*patch_w
        return x


class UViT(nn.Module): #TODO: set the flags!!!
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, use_checkpoint=False,
                 clip_dim=768, num_clip_token=77, conv=True, skip=True, num_panoptic_class=8, enable_panoptic=True, use_ground_truth=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_chans = in_chans
        self.enable_panoptic=enable_panoptic

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.context_embed = nn.Linear(clip_dim, embed_dim)

        self.extras = 1 + num_clip_token

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + 2*num_patches, embed_dim)) #TODO:changed to 2

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint, enable_panoptic=enable_panoptic) #note:set to true
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint,enable_panoptic=enable_panoptic)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint,enable_panoptic=enable_panoptic)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        #TODO:add layers for panoptic segmentation masks
        #use analog bits, 8 bits for 200 classes
        self.mask_embed=PatchEmbed(patch_size=patch_size, in_chans=8, embed_dim=embed_dim) #map category id to embed dim
        self.decoder_pred_mask = nn.Linear(embed_dim, self.patch_dim, bias=True)
        
        #NOTE: predict category ids and then use cross entropy loss #set num_panoptic_class to 8 for analog bits
        self.final_layer_mask =nn.Conv2d(self.in_chans, num_panoptic_class,3, padding=1) if conv else nn.Identity()
        #NOTE: set to true if input ground truth mask
        self.use_ground_truth=use_ground_truth

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, context, mask_token=None, use_ground_truth=False, enable_panoptic=False):
        #NOTE: activate/deactivate this when calling the forward pass of the model
        self.use_ground_truth=use_ground_truth
        
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        context_token = self.context_embed(context)
        #TODO: add mask token randomly initialized
        if mask_token is not None:
            mask_embedding=self.mask_embed(mask_token)
            x = torch.cat((time_token, context_token, x, mask_embedding), dim=1)
            x = x + self.pos_embed
        else:
            x = torch.cat((time_token, context_token, x), dim=1)
            x = x + self.pos_embed[:, :self.extras + L, :]
            enable_panoptic=False #must be disabled for unconditional ones

        skips = []
        for blk in self.in_blocks:
            x = blk(x, extra_dims=self.extras, use_ground_truth=self.use_ground_truth, enable_panoptic=enable_panoptic )
            skips.append(x)

        x = self.mid_block(x, extra_dims=self.extras, use_ground_truth=self.use_ground_truth, enable_panoptic=enable_panoptic )

        for blk in self.out_blocks:
            x = blk(x, skips.pop(), extra_dims=self.extras, use_ground_truth=self.use_ground_truth, enable_panoptic=enable_panoptic )

        x = self.norm(x)
        #predict noise, use only x queries, ignore mask queries
        #if mask_token is not None:
        #    assert x.size(1) == self.extras + *L
        #noise = x[:, self.extras:self.extras + L, :]

        #TODO: generate panoptic segmentation masks, use only mask queries
        if mask_token is not None:
            #TODO: Jan17 use ground truth mask, do not predict it
            if self.use_ground_truth==True:
                mask_feature=x[:, self.extras+L:, :]
                image_feature=x[:, self.extras:self.extras + L, :]
                #merge together
                image_feature=image_feature+mask_feature
                noise = self.decoder_pred(image_feature)
                #ground truth mask
                y= mask_token
            else:
                noise = self.decoder_pred(x[:, self.extras:self.extras + L, :])
                y = self.decoder_pred_mask(x[:, self.extras+L:, :])
                #y = x[:, self.extras+L:, :]
                y = unpatchify(y, self.in_chans)
                y = self.final_layer_mask(y)

        else:
            noise = self.decoder_pred(x[:, self.extras:self.extras + L, :])

        noise = unpatchify(noise, self.in_chans)
        noise = self.final_layer(noise)

        if mask_token is not None:
            return noise, y 
        else:
            return noise
