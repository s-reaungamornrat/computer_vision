from functools import partial 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.modules.utils import _pair

def _cfg(url='', **kwargs):
    return {'url':url, 'num_classes':400, 'input_size':(3,224,224), 'pool_size':None, 'crop_pct':.9, 'interpolation':'bicubic', 'mean':(0.5,.5,.5),
           'std':(.5,.5,.5), **kwargs}

def drop_path(x, drop_prob:float=0., training:bool=False, scale_by_keep:bool=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)

    This is the same as the DropConnect implementation for EfficientNet; however, the original name is misleading as 'Drop Connect' is a different form 
    of dropout in a separate paper. See discussion https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956. 
    Reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py#L158
    """
    if drop_prob==0. or not training: return x
    keep_prob=1.-drop_prob 
    shape=(x.shape[0],)+(1,)*(x.ndim-1) # work with different dim tensors, not just 2D ConvNets
    random_tensor=x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob>0. and scale_by_keep: random_tensor.div_(keep_prob)
    return x*random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample 
    Referernce:
        https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py#L158
        https://github.com/OpenGVLab/VideoMAEv2/blob/master/models/modeling_finetune.py#L198
    """
    def __init__(self, drop_prob:float=0., scale_by_keep:bool=True):
        super(DropPath, self).__init__()
        self.drop_prob=drop_prob
        self.scale_by_keep=scale_by_keep
    def forward(self, x): return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    def extra_repr(self)->str: return f"p={round(self.drop_prob,3):0.3f}"

class MLP(nn.Module):
    """
    Examples
    >>> x=torch.rand(10, 80)
    >>> module=MLP(in_features=80, hidden_features=40, out_features=12, act_layer=nn.GELU, drop=0.5)
    >>> out=module(x)
    >>> print(f"{out.shape=}") # (10,12)
    >>> nn.MSELoss()(out, torch.rand_like(out)).backward()
    
    >>> x=torch.rand(10, 30, 80)
    >>> module=MLP(in_features=80, hidden_features=40, out_features=12, act_layer=nn.GELU, drop=0.5)
    >>> out=module(x)
    >>> print(f"{out.shape=}") # (10,30,12)
    >>> nn.MSELoss()(out, torch.rand_like(out)).backward()
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features=out_features or in_features
        hidden_features=hidden_features or in_features
        self.fc1=nn.Linear(in_features, hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features, out_features)
        self.drop=nn.Dropout(drop) if drop>0. else None
    def forward(self, x):
        x=self.act(self.fc1(x))
        x=self.fc2(x)
        if self.drop is not None: x=self.drop(x)
        return x

class Attention(nn.Module):
    """
    Compute scaled-dot product attention where Attebtion= (qk^T)/sqrt(head_dim), depending on both vector magnitudes and directions. 
    The attention measures how aligned quries and keys and to which degrees the alignment are.
    Args:
        dim (int): Input embedding dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in extracting Q, K, V
        qk_scale (float): Scale Q before computing attention. If None, it will be set to 1/sqrt(head_dim)
        attn_drop (float): Probability of dropout on the attention matrix. Often set to 0 during pretraining, since it provides too noisy signals for a 
            model that already under immense constraint of training data degradation
        proj_drop (float): Probability of dropout after the output projection. Often set to 0 during pretraining, since it provides too noisy signals for a 
            model that already under immense constraint of training data degradation
        attn_head_dim (int): Dimension of features of each head. If None, it will be set to `dim//num_heads`
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads=num_heads
        head_dim=dim//num_heads
        if attn_head_dim is not None: head_dim=attn_head_dim
        all_head_dim=head_dim*self.num_heads
        self.scale=qk_scale or head_dim**(-0.5)
        print(f"{all_head_dim//self.num_heads=}")

        self.qkv=nn.Linear(dim, all_head_dim*3, bias=False)
        self.q_bias=self.v_bias=None
        if qkv_bias:
            self.q_bias=nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias=nn.Parameter(torch.zeros(all_head_dim))

        self.attn_drop=nn.Dropout(attn_drop) if attn_drop>0. else None
        self.proj=nn.Linear(all_head_dim, dim)
        self.proj_drop=nn.Dropout(proj_drop) if proj_drop>0. else None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B,N,C) where B is the batch size, C is the feature dimension, N is the sequence length/number of
                patches/number of tokens in the sequence. For example, if a 224x224 image is divided into patches whose size are 16x16 pixels, 
                we have 14x14 patches so N=196
        Returns:
            (torch.Tensor): Output features of shape (B,N,C)
        """ 
        B,N,C=x.shape
        qkv_bias=None
        if self.q_bias is not None:
            # size (3xH,) where H is the dimension of all heads, typically H=C where C is the embedding dimensions
            qkv_bias=torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias) 
            )
        qkv=F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias) # (B,N,3H)
        # (B,N,3H)->(B,N,3,num_heads,H/num_heads)->(3,B,num_heads,N,H/num_heads)
        qkv=qkv.reshape(B,N,3,self.num_heads,-1).permute(2,0,3,1,4) 
        q,k,v=qkv[0],qkv[1],qkv[2] # each is (B,num_heads,N,H/num_heads)
        q=q*self.scale
        attn=(q@k.transpose(-2,-1)) # (B,num_heads,N,H/num_heads) (B,num_heads,H/num_heads,N) -> (B,num_heads,N,N)
        attn=attn.softmax(dim=-1) # (B,num_heads,N,N)
        if self.attn_drop is not None: attn=self.attn_drop(attn)
        # (B,num_heads,N,N) (B,num_heads,N,H/num_heads) -> (B,num_heads,N,H/num_heads) -> (B,N, num_heads,H/num_heads) -> (B,N,H)
        x=(attn@v).transpose(1,2).reshape(B,N,-1)
        x=self.proj(x) #  (B,N,H)->(B,N,C)
        if self.proj_drop is not None: x=self.proj_drop(x) # (B,N,C)
        return x

class CosAttention(nn.Module):
    """
    Compute a cosine-similarity based attention, where Attention=cos(\theta), only depending on vector directions. 
    Thus, better stability (avoiding magnitude explosion) but less expressive. The cosine-similarity attention measures how aligned quries and keys.
    Args:
        dim (int): Input embedding dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in extracting Q, K, V
        qk_scale (float): Scale Q before computing attention. If None, it will be set to 1/sqrt(head_dim)
        attn_drop (float): Probability of dropout on the attention matrix. Often set to 0 during pretraining, since it provides too noisy signals for a 
            model that already under immense constraint of training data degradation
        proj_drop (float): Probability of dropout after the output projection. Often set to 0 during pretraining, since it provides too noisy signals for a 
            model that already under immense constraint of training data degradation
        attn_head_dim (int): Dimension of features of each head. If None, it will be set to `dim//num_heads`
    
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads=num_heads
        head_dim=dim//num_heads
        if attn_head_dim is not None: head_dim=attn_head_dim
        all_head_dim=head_dim*self.num_heads
        if qk_scale is None: self.scale=nn.Parameter(torch.log(10*torch.ones((num_heads, 1, 1))), requires_grad=True)
        else: self.scale=qk_scale
        self.qkv=nn.Linear(dim, all_head_dim*3, bias=False)
        self.q_bias=self.v_bias=None
        if qkv_bias:
            self.q_bias=nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias=nn.Parameter(torch.zeros(all_head_dim))
        self.attn_drop=nn.Dropout(attn_drop) if attn_drop>0. else None
        self.proj=nn.Linear(all_head_dim, dim)
        self.proj_drop=nn.Dropout(proj_drop) if proj_drop>0. else None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B,N,C) where B is the batch size, C is the feature dimension, N is the sequence length/number of
                patches/number of tokens in the sequence. For example, if a 224x224 image is divided into patches whose size are 16x16 pixels, 
                we have 14x14 patches so N=196
        Returns:
            (torch.Tensor): Output features of shape (B,N,C)
        """ 
        B,N,C=x.shape
        qkv_bias=None
        if self.q_bias is not None:
            qkv_bias=torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )
        qkv=F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv=qkv.reshape(B,N,3,self.num_heads, -1).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2] # each is (B,num_heads,N,H/num_heads)
        attn=(
            F.normalize(q, dim=-1) @ F.normalize(k,dim=-1).transpose(-2,-1) #  (B,num_heads,N,N)
        )
        # torch.log(torch.tensor(1./0.01))=4.6052
        logit_scale=torch.clamp(self.scale, max=4.6053).exp() # so maximum scale is 100.0
        
        attn=attn*logit_scale
        attn=attn.softmax(dim=-1)
        if self.attn_drop is not None: attn=self.attn_drop(attn)
        
        x=(attn@v).transpose(1,2).reshape(B,N,-1)
        x=self.proj(x)
        if self.proj_drop is not None: x=self.proj_drop(x)
        return x

class Block(nn.Module):
    """
    Attention block
    Args:
        dim (int): Input embedding dimension/number of input channels
        num_heads (int): Number of attention heads. This will be used to compute `head_dim` if `attn_head_dim` is None. Some research suggested that keeping 
            a constant head dim (like 64 or 32) while scaling the number of heads is more efficient. 
        mlp_ratio (float): Ratio of MLP hidden-layer dimension to input-feature/embedding dimension
        qkv_bias (bool): Whether to use bias in extracting Q, K, V
        qk_scale (float): Scale Q before computing attention. If None, it will be set to 1/sqrt(head_dim)
        drop (float): Probability of dropout of MLP output and of projection layers. Usually set to 0 for large datasets
        attn_drop (float): Probability of dropout on the attention matrix. Often set to 0 during pretraining, since it provides too noisy signals for a 
            model that already under immense constraint of training data degradation. Also, set to 0 to avoid 'blurring' attention
        drop_path (float): Probability of drop path/Stochastic depth rate. Often set to nonzero during pretraining to allpw training of randomly shallower
            networks, boost gradient flows and improving training stability. It is often scalled linearly across the entire model with a maximux rate of 0.1.
            For example, `Block0` with `drop_path=0.` and `Block11` with `drop_path=0.1`
        init_values (float): Initial value for layer scaling. If `init_values>0`, it initializes learnable vector (gamma_1, gamma_2) of size (dim,) that scale
            the ouput of the attention and MLP branches before residual connection. Common values are 1e-5 or 1e-6, improving training stability in very deep 
            transformers (24+ layers) by preventing the variance of the hidden states from explosion
        attn_head_dim (int): Dimension of features of each head. If None, it will be set to `dim//num_heads`
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., init_values=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None, cos_attn=False):
        super().__init__()
        self.norm1=norm_layer(dim)
        if cos_attn:
            self.attn=CosAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
                                   attn_head_dim=attn_head_dim)
        else:
            self.attn=Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
                                attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochatis depth, we shall see whether this is better than dropout
        self.drop_path=DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2=norm_layer(dim)
        mlp_hidden_dim=int(dim*mlp_ratio)
        self.mlp=MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1=self.gamma_2=None
        if init_values>0:
            # By starting gamma at a very small value (like 1e-6), we force the model to behave like an identity function at the start of training
            # x=x+DropPath(gamma*layer(norm(x))) where layer is either attn or mlp modules
            self.gamma_1=nn.Parameter(init_values*torch.ones((dim)), requires_grad=True)
            self.gamma_2=nn.Parameter(init_values*torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B,N,C) where B is the batch size, C is the feature dimension, N is the sequence length/number of
                patches/number of tokens in the sequence. For example, if a 224x224 image is divided into patches whose size are 16x16 pixels, 
                we have 14x14 patches so N=196
        Returns:
            (torch.Tensor): Output features of shape (B,N,C)
        """ 
        if self.gamma_1 is None:
            x=x+self.drop_path(self.attn(self.norm1(x)))
            x=x+self.drop_path(self.mlp(self.norm2(x)))
        else:
            x=x+self.drop_path(self.gamma_1*self.attn(self.norm1(x)))
            x=x+self.drop_path(self.gamma_2*self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        img_size (int | tuple[int,int]): The resolution of input video frames (height and width). If an integer is provided, it assumes a square input.
        patch_size (int | tuple[int, int]): The spatial dimensions of each patch (height and width). For example, 16 means each patch cover a 16x16 pixel area. 
        in_chans (int): The number of input channels per frame. Typically 3 for RGB videos
        embed_dim (int): The projection dimension (embedding size)/ the length of the vector each 'tubelet' converted into
        tubelet_size (int): The temporal length of each patch (the depth of the 3D cube). If `tubelet_size=2`, this module merges 2 consecutive frames into
            a single patch (i.e., reducing temporal redundancy).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size=_pair(img_size)
        patch_size=_pair(patch_size)
        num_spatial_patches=(img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1])
        num_patches=num_spatial_patches*(num_frames//tubelet_size)

        self.img_size=img_size
        self.tubelet_size=tubelet_size
        self.patch_size=patch_size
        self.num_patches=num_patches
        # Both kernel and stride are set to (`tubelet_size`, `patch_size[0]`, `patch_size[1]`), ensuring (1) non-overlapping extraction and 
        # (2) dimension reduction from (C,T,H,W) to (L, embed_dim) where L is the total number of tubelets/patches/tokens
        # L=(num_frames/tubelet_size) * (img_size[0]/patch_size[0]) * (img_size[1]/patch_size[1])
        self.proj=nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                            stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B,C,T,H,W) where B is the batch size, T is the number of frames, C is the input channels
        Returns:
            (torch.Tensor): Embedding of shape (B,L,embed_dim) where L is the total number of tubelets/patches/tokens 
                L=(num_frames/tubelet_size) * (img_size[0]/patch_size[0]) * (img_size[1]/patch_size[1])
        """ 
        B,C,T,H,W=x.shape
        assert H==self.img_size[0] and W==self.img_size[1], f"Input image size ({H},{W}) does not match model ({self.img_size[0]},{self.img_size[1]})"
        # (B,L,T//tubelet_size,H//patch_size[0],W//patch_size[1]) -> (B,L,(T//tubelet_size * H//patch_size[0] * W//patch_size[1]) )
        # (B, (T//tubelet_size * H//patch_size[0] * W//patch_size[1]) , L)
        x=self.proj(x).flatten(2).transpose(1,2)
        return x

def get_sinusoid_encoding_table(n_position, d_hid, dtype=torch.float32):
    """Sinusoid position encoding table. This is faster and pytorch-based implement version of `get_sinusoid_encoding_table` (see Reference for the old version)
    Args:
        n_position (int): Sequence length/the number of tubelets/tokens/patches
        d_hid (int): Hidden dimension, must be equal to the embed dimension of `Block` and `PatchEmbed` and must be even since the function
            splits this into sine and cosine pairs
    Returns:
        (torch.Tensor): Sin-Cos poositional embedding of shape (1,n_position,d_hid)
    Reference: 
        https://github.com/OpenGVLab/VideoMAEv2/blob/master/models/modeling_finetune.py#L74
        https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
    """

    # Create a vector of position [n_position, 1], e.g., [0,1,2,...,n_position-1]
    position=torch.arange(n_position, dtype=dtype).unsqueeze(1) # (n_position,1) where n_positionL is the sequence length
    # Create the division term (the denominator)
    # We only need it for half the dimensions since we apply sin/cos to pairs. Logic: 10000^(2i/d_hid)
    denominator=torch.exp(-torch.log(torch.tensor(10000.))*(2*(torch.arange(d_hid,dtype=dtype)//2)/d_hid))
    
    # Create an empty table
    sinusoid_table=position*denominator  # (n_position,d_hid)
    # Apply sin to even indices and cos to odd indices
    # position*div_term results in shape (n_position, d_hid//2)
    sinusoid_table[:,0::2]=torch.sin(sinusoid_table[:,0::2]) # (L,d_hid//2)
    sinusoid_table[:,1::2]=torch.cos(sinusoid_table[:,1::2]) # (L,d_hid//2)
    return sinusoid_table.unsqueeze(0).detach()


def trunc_normal_(tensor, mean=0., std=1.):
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    
def _init_weights(self, m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0.)
        nn.init.constant_(m.weight, 1.)
        
class VisionTransformer(nn.Module):
    """Vision transfomer with support for patch or hybrid CNN input stage
    Args:
        img_size (int | tuple[int,int]): The resolution of input video frames (height and width). If an integer is provided, it assumes a square input.
        patch_size (int | tuple[int, int]): The spatial dimensions of each patch (height and width). For example, 16 means each patch cover a 16x16 pixel area. 
        in_chans (int): The number of input channels per frame. Typically 3 for RGB videos
        num_classes (int): Output number of classes
        embed_dim (int): The projection dimension (embedding size)/ the length of the vector each 'tubelet' converted into
        depth (int): Number of transformer blocks to stack on top of each other
        num_heads (int): Number of attention heads. This will be used to compute `head_dim` if `attn_head_dim` is None. Some research suggested that keeping 
            a constant head dim (like 64 or 32) while scaling the number of heads is more efficient. 
        mlp_ratio (float): Ratio of MLP hidden-layer dimension to input-feature/embedding dimension
        qkv_bias (bool): Whether to use bias in extracting Q, K, V
        qk_scale (float): Scale Q before computing attention. If None, it will be set to 1/sqrt(head_dim)
        drop_rate (float): Probability of dropout of features after adding position embeddings and of transformer MLP output and of transformer projection 
            layers. Usually set to 0 for large datasets
        attn_drop_rate (float): Probability of dropout on the attention matrix. Often set to 0 during pretraining, since it provides too noisy signals for a 
            model that already under immense constraint of training data degradation. Also, set to 0 to avoid 'blurring' attention
        drop_path_rate (float): Probability of drop path/Stochastic depth rate. Often set to nonzero during pretraining to allpw training of randomly shallower
            networks, boost gradient flows and improving training stability. It is often scalled linearly across the entire model with a maximux rate of 0.1.
            For example, `Block0` with `drop_path=0.` and `Block11` with `drop_path=0.1`
        head_drop_rate (float): Probability of dropout on features before passing to head.
        norm_layer (nn.Module): Normalization module
        init_values (float): Initial value for layer scaling. If `init_values>0`, it initializes learnable vector (gamma_1, gamma_2) of size (dim,) that scale
            the ouput of the attention and MLP branches before residual connection. Common values are 1e-5 or 1e-6, improving training stability in very deep 
            transformers (24+ layers) by preventing the variance of the hidden states from explosion
        use_learnable_pos_emb (bool): Whether to use learnable position embedding; otherwise use a fixed sine-cosine position embeddings
        init_scale (float): The initial value used to scale the output of the classification head or the final layers. In many MAE-based models, this is part 
            of a "LayerScale" or a specialized initialization strategy (like Truncated Normal) to prevent signal explosion in deep transformers. Setting it to
            a small value (e.g., 1e-5 or 0) helps the model start training with a more stable, identity-like transformation before the weights diverge.
        all_frames (int): The total number of temporal frames sampled from the input video clip. This defines the temporal dimension of the input tensor before 
            it is partitioned into 3D "tubes" (voxels). For example, if all_frames=16 and your tubelet_size=2, the model will produce 8 temporal tokens.
        tubelet_size (int): The temporal length of each patch (the depth of the 3D cube). If `tubelet_size=2`, this module merges 2 consecutive frames into
            a single patch (i.e., reducing temporal redundancy).
        use_mean_pooling (bool): Whether to use global average pooling of backbone features or use the return feature from the first [CLS] token as output from
            transformer blocks
        with_cp (bool): Whether to use checkpoint
        cos_attn (bool): Whether to use CosAttention
        attn_head_dim (int): Dimension of features of each head. If None, it will be set to `dim//num_heads`
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., head_drop_rate=0., norm_layer=nn.LayerNorm, init_values=0.,
                 use_learnable_pos_emb=False, init_scale=0., all_frames=16, tubelet_size=2, use_mean_pooling=True, with_cp=False, cos_attn=False, 
                 attn_head_dim=None):
        super().__init__()
        self.num_classes=num_classes
        # num_features for consistency with other models
        self.num_features=self.embed_dim=embed_dim
        self.tubelet_size=tubelet_size
        self.patch_embed=PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, 
                                    tubelet_size=tubelet_size)
        num_patches=self.patch_embed.num_patches
        self.with_cp=with_cp

        if use_learnable_pos_emb: self.pos_embed=nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else: self.pos_embed=get_sinusoid_encoding_table(num_patches, embed_dim) # sine-cosine positional embedding

        self.pos_drop=nn.Dropout(p=drop_rate) if drop_rate>0. else None
        dpr=torch.linspace(0, drop_path_rate, depth).tolist() # stochastic depth decay rule
        self.blocks=nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                 drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, cos_attn=cos_attn, attn_head_dim=attn_head_dim) for i in range(depth)
        ])
        self.norm=nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm=norm_layer(embed_dim) if use_mean_pooling else None
        self.head_dropout=nn.Dropout(head_drop_rate) if head_drop_rate>0. else None
        self.head=nn.Linear(embed_dim, num_classes) if num_classes>0 else nn.Identity()
        if use_learnable_pos_emb: trunc_normal_(self.pos_embed, std=0.02) # TEST whether this work compared to the old one
        self.apply(self._init_weights)
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward_features(self, x):
        """Process a raw input video into a high-level representation (a single vector per video) that classification head can use
        Args:
            x (torch.Tensor): Input sampled video frames of size (B,C,T,H,W) where C is the input channels, T is the number of video frames (must matches
                `all_frames`), (H,W) is the height and width (must match `img_size`)
        Returns:
            (torch.Tensor): Backbone embeddings of shape (B, embed-dim)
        """
        B=x.size(0)

        x=self.patch_embed(x) # (B, sequence_length, embed_dim)

        if self.pos_embed is not None:
            # (B, sequence_length, embed_dim)+(1, sequence_length, embed_dim) =(B, sequence_length, embed_dim) 
            x=x+self.pos_embed.clone().to(device=x.device, dtype=x.dtype).detach()
        if self.pos_drop is not None: x=self.pos_drop(x)
        
        for i, block in enumerate(self.blocks):
            if self.with_cp: x=cp.checkpoint(block, x) # (B, sequence_length, embed_dim) 
            else: x=block(x) # (B, sequence_length, embed_dim) 
        
        # global average pooling: use when the model is designed to look at the entirety of the spatial-temporal information equally
        #      By averaging, the model becomes less sensitive to where a specific action/object is located in the video
        # cls token approach: classic approach from BERT (NLP). Only extracts the very first token in the sequence (index 0). The [CLS] token
        #      is a dummy token prepended to the input. It is forced to learn an aggregate representation of all other tokens. It provides a consistent
        #      dedicate slot for classification summary, regardless of how many patches the input has
        if self.fc_norm is not None: out=self.fc_norm(x.mean(1)) 
        else: out=self.norm(x[:,0])
        
        return out

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sampled video frames of size (B,C,T,H,W) where C is the input channels, T is the number of video frames (must matches
                `all_frames`), (H,W) is the height and width (must match `img_size`)
        Returns:
            (torch.Tensor): Output of classification of shape (B, num_classes)
        """
        x=self.forward_features(x)
        if self.head_dropout is not None: x=self.head_dropout(x) 
        x=self.head(x)
        return x
        