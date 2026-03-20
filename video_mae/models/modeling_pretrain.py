from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .modeling_finetune import (Block, PatchEmbed, _cfg, get_sinusoid_encoding_table, trunc_normal_)

class PretrainVisionTransformerEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage
    
        attn_head_dim (int): Dimension of features of each head. If None, it will be set to `dim//num_heads`
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, 
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, 
                 use_learnable_pos_emb=False, with_cp=False, all_frames=16, cos_attn=False, attn_head_dim=None):
        super().__init__()
        self.num_classes=num_classes
        # num_features for consistency with other models
        self.num_features=self.embed_dim=embed_dim
        self.patch_embed=PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, 
                                    tubelet_size=tubelet_size)
        num_patches=self.patch_embed.num_patches
        self.with_cp=with_cp

        #if use_learnable_pos_emb: self.pos_embed=nn.Parameter(torch.zeros(1, num_patches+1, embed_dim)) # patches + [CLS] token prepended 
        if use_learnable_pos_emb: self.pos_embed=nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else: self.pos_embed=get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr=torch.linspace(0, drop_path_rate, depth).tolist() # stochastic depth decay rule
        self.blocks=nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                 drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, cos_attn=cos_attn, attn_head_dim=attn_head_dim) for i in range(depth)
        ])
        self.norm=norm_layer(embed_dim)
        self.head=nn.Linear(embed_dim, num_classes) if num_classes>0 else nn.Identity()
        if use_learnable_pos_emb: trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.)
        
    def get_num_layers(self): return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self): return {'pos_embed', 'cls_token'}

    def get_classifier(self): return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes=num_classes
        self.head=nn.Linear(self.embed_dim, num_classes) if num_classes>0 else nn.Identity()

    def forward_features(self, x, mask):
        """
        Args:
            x (torch.Tensor): Video frame tensor with shape (B,C,T,H,W) where B is the batch size, C is the number of input channels, T is the number of frames
                (must match `all_frames`) and (H,W) is the height and width of the frame (must match `img_size`)
            mask (torch.Tensor): Bool tensor representing whether spatio-temporal patches (tubelets) are masked out or visible, of size (B, num_patches),
                where num_patches= (T/tubelet_size) * (H/patch_size[0]) * (W/patch_size[1]). For example, if x is of shape (B, 3, 16,224,224) and patch
                is of size (16,16). num_patches will be (16/2) * (224/16) * (224/16) = 8*14*14=1568. Note 1 means mask out and 0 means kept
        Returns:
            (torch.Tensor): Backbone features of shape (B,vis,embed_dim) where vis is the number of patches visible after masking
        """
        x=self.patch_embed(x) # (B, num_patches,embed_dim)
        # (B, num_patches,embed_dim)(1, num_patches,embed_dim)=(B, num_patches,embed_dim)
        x=x+self.pos_embed.clone().to(device=x.device, dtype=x.dtype).detach()
        
        B,_, C=x.shape
        x_vis=x[~mask].reshape(B,-1,C) # keep ~mask visible patches (B,vis,embed_dim), where vis the number of patches that is still visible
        
        for i, block in enumerate(self.blocks):
            if self.with_cp: x_vis=cp.checkpoint(block, x_vis) # (B,vis,embed_dim)
            else: x_vis=block(x_vis) # (B,vis,embed_dim)
        
        x_vis=self.norm(x_vis) # (B,vis,embed_dim)
        return x_vis # (B,vis,embed_dim)

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Video frame tensor with shape (B,C,T,H,W) where B is the batch size, C is the number of input channels, T is the number of frames
                (must match `all_frames`) and (H,W) is the height and width of the frame (must match `img_size`)
            mask (torch.Tensor): Bool tensor representing whether spatio-temporal patches (tubelets) are masked out or visible, of size (B, num_patches),
                where num_patches= (T/tubelet_size) * (H/patch_size[0]) * (W/patch_size[1]). For example, if x is of shape (B, 3, 16,224,224) and patch
                is of size (16,16). num_patches will be (16/2) * (224/16) * (224/16) = 8*14*14=1568. Note 1 means mask out and 0 means kept
        Returns:
            (torch.Tensor): Output of classification of shape (B, num_classes)
        """
        x=self.forward_features(x,mask)
        x=self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """Transform hidden representation to a video patch"""
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2, with_cp=False, 
                 cos_attn=False, attn_head_dim=None):
        super().__init__()
        self.num_classes=num_classes
        # num_classes: total number of raw pixel values within a single tubelet. Below, we make sure that the decoder's final layer outputs exactly enough
        # values to reconstruct every pixel in the original video volume
        assert num_classes==3*tubelet_size*patch_size**2 # 3 is for RGB
        # num_features for consistency with other models
        self.num_features=self.embed_dim=embed_dim
        self.patch_size=patch_size
        self.with_cp=with_cp

        dpr=torch.linspace(0, drop_path_rate, depth).tolist() # stochastic depth decay rule
        self.blocks=nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, cos_attn=cos_attn, attn_head_dim=attn_head_dim) for i in range(depth)
        ])
        self.norm=norm_layer(embed_dim)
        self.head=nn.Linear(embed_dim, num_classes) if num_classes>0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.)

    def get_num_layers(self): return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self): return {'pos_embed', 'cls_token'}

    def get_classifier(self): return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes=num_classes
        self.head=nn.Linear(self.embed_dim, num_classes) if num_classes>0 else nn.Identity()

    def forward(self, x, return_token_num):
        """
        Args:
            x (torch.Tensor): Hidden representation, typically the concatenation of (encoder_output+position_embedding) and (mask_token+decode_visible), of size
                (B,num_patches,dec_embed_dim)
            return_token_num (int): Number of patches that got masked out from encoder visibility and decoder needs to estimate them
        Returns:
            (torch.Tensor): If return_token_num>0, return the values of all raw pixels of patches that got masked out, of size (B,return_token_num,C)
                Otherwise, return all values of raw pixels of all patches (B,num_patches, C), where C is the number of raw pixel values
        """
        for i, block in enumerate(self.blocks):
            if self.with_cp: x=cp.checkpoint(block, x)
            else: x=block(x)
        
        if return_token_num>0:
            # only return the mask tokens predicted pixels
            out=self.head(self.norm(x[:,-return_token_num:])) # (B,return_token_num,C), where C is the number of raw pixel values
        else:
            out=self.head(self.norm(x)) # (B,num_patches, C), where C is the number of raw pixel values
        return out

class PretrainVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_num_classes=1536, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False, tubelet_size=2,
                 with_cp=False, all_frames=16, cos_attn=False, attn_head_dim=None):
        super().__init__()
        self.encoder=PretrainVisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, num_classes=encoder_num_classes,
                                                      embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
                                                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                      drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size,
                                                      use_learnable_pos_emb=use_learnable_pos_emb, with_cp=with_cp, all_frames=all_frames, cos_attn=cos_attn,
                                                      attn_head_dim=attn_head_dim)
        self.decoder=PretrainVisionTransformerDecoder(patch_size=patch_size, num_patches=self.encoder.patch_embed.num_patches, num_classes=decoder_num_classes,
                                                      embed_dim=decoder_embed_dim, depth=decoder_depth, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                                                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                      drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size,
                                                      with_cp=with_cp,cos_attn=cos_attn, attn_head_dim=attn_head_dim)
        self.encoder_to_decoder=nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        # placeholder or query for the information that was removed during masking
        self.mask_token=nn.Parameter(torch.zeros(1,1,decoder_embed_dim))
        self.pos_embed=get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)
        trunc_normal_(self.mask_token, std=.02)

    def forward(self, x, mask, decode_mask=None):
        """
        Args:
            x (torch.Tensor): Input sampled video frames of size (B,C,T,H,W) where C is the input channels, T is the number of frames, H,w is the height
                and width
            mask (torch.Tensor): Bool tensor of size (B, num_patches) where 1 represent patches that should be masked out and 0 represent patches
                visible to the encoder
            decode_mask (torch.Tensor): Bool tensor of size (B, num_patches) where 1 represent patches that should be masked out and 0 represent patches
                visible to the decoder. If None, `decode_mask` is `mask` since it is tasked to reconstruct patches invisible to the encoder 
        Returns:
            (torch.Tensor): The values of all raw pixels of patches that got masked out, of size (B,return_token_num,K), where K is the number of raw pixel 
                values
        """
        
        decode_vis=mask if decode_mask is None else ~decode_mask
        
        x_vis=self.encoder(x, mask) # (B,N_vis,encoder_embed_dim)
        x_vis=self.encoder_to_decoder(x_vis) # (B,N_vis,decoder_embed_dim)
        
        B,N_vis,C=x_vis.shape
        # we do not unshuffle the correct visible token order, but shuffle the pos embedding accordingly
        # (1,num_patches,dec_embed_dim) -> (B,num_patches,dec_embed_dim)
        expand_pos_embed=self.pos_embed.clone().expand(B,-1,-1).to(dtype=x.dtype, device=x.device).detach()
        # positional embeddings for patches the encoder processed
        pos_emb_vis=expand_pos_embed[~mask].reshape(B,-1,C) # (B,N_vis, dec_embed_dim)
        # positional embeddings for patches the decoder is about to predict
        pos_emd_mask=expand_pos_embed[decode_vis].reshape(B,-1,C) # (B, num_patches-N_vis, dec_embed_dim)
        
        # we note that x_vis,pos_embed are both of size (B,N_vis,dec_embed_dim)
        # self.mask_token (1,1,dec_embed_dim)
        # pos_emd_mask (B, num_patches-N_vis, dec_embed_dim)
        # so cat of (B,N_vis,dec_embed_dim) (B, num_patches-N_vis, dec_embed_dim) give (B, num_patches, dec_embed_dim)
        x_full=torch.cat((x_vis+pos_emb_vis, self.mask_token+pos_emd_mask), dim=1) 
        x=self.decoder(x_full, pos_emd_mask.shape[1])
        return x