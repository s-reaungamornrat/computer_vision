# modify from https://github.com/OpenGVLab/VideoMAEv2/blob/master/run_class_finetuning.py#L24
import numbers
from collections import OrderedDict

import torch

from computer_vision.video_mae.utils import load_state_dict

def learned_positional_embedding_size_change(model, pos_embed_checkpoint, num_frames, tubelet_size):
    """Learned positional embeddedings may be trained on images with different size than images used in fine-tuning, thus resulting the different number of 
    patch grid (i.e., different grid size or the number of patch in each dimension). We stretch or compress the learned embedding to match the new grid

    In other words, the spatial grid may change since the images used in finetuning may be bigger, so we need to adjust the learned positional embedding. We do 
    not interpolate or modify the temporal dimension
    
    Args:
        model (torch.nn.Module): Current model
        pos_embed_checkpoint (torch.Tensor): Learned positional embeddeding loaded from checkpoints of a pretrained model of shape (1, num_patches, embed_dim)
        num_frames (int): Number of frames used to train a model
        tubelet_size (int): Temporal length of a single video patch
    """
    # Positional embeddings are a map telling the model where each patch is located in spacetime
    # - If we move from a 14x14 grid to a 28x28 grid, the model doesn't have embeddings for the new locations
    # - The code use `orig_size` and `new_size` to perform bicubic interpolation to stretch the old 14x14 grid of embeddings to the new
    #   28x28 grid, ensuring the model retains the spatial relationships it learned during pretraining
    embedding_size=pos_embed_checkpoint.shape[-1]
    t=num_frames//tubelet_size # grid dimension along temporal dimension
    num_patches=model.patch_embed.num_patches
    # Vision transformers often have a [CLS] token, This line determines if there are 'extra' tokens that should not be part of the spatial 
    # interpolation (usually 1 for teh class token or 0 if using global average pooling)
    num_extra_tokens=model.pos_embed.shape[-2]-num_patches

    # Find the spatial width/height (equal) of checkpoint pos_embed 
    # - first remove extra-token
    # - divide the number of temporal steps
    # - taking square root
    # Example: if the checkpoint had 1568 patches, 16 frames, and a tubelet size of 2, that is 1568/8=196. The sqrt(196)=14, so the original
    # spatial grid was 14x14
    orig_size=int(
        ( 
          (pos_embed_checkpoint.shape[-2]-num_extra_tokens)//t
        )**0.5
    )
    # Find the spatial width/height (equal) of the current model. 
    new_size=int(
        (
            num_patches//t
        )**0.5
    )
    if orig_size!=new_size:
        print(f"Positional embedding interpolated from {orig_size}x{orig_size} to {new_size}x{new_size}")
        extra_tokens=pos_embed_checkpoint[:,:num_extra_tokens] # extra_token is the beginning of embeddings
        # only positional tokens are interpolated
        pos_token=pos_embed_checkpoint[:,num_extra_tokens:] # positional embeddings are every token after the extra_token
        # (B,L,C)=(B,[num_frames//tube_size * H//patch_size * W//patch_size], C)->(B,num_frames//tube_size,H//patch_size, W//patch_size,C)
        pos_token=pos_token.reshape(-1, t, orig_size, orig_size, embedding_size)
        # -> ([B * num_frames//tube_size], H//patch_size, W//patch_size, C) -> ([B * num_frames//tube_size], C, H//patch_size, W//patch_size)
        pos_token=pos_token.reshape(-1, orig_size, orig_size, embedding_size).permute(0,3,1,2)
        pos_token=torch.nn.functional.interpolate(pos_token, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # -> ([B * num_frames//tube_size], H//patch_size, W//patch_size, C)->(B,num_frames//tube_size,H//patch_size, W//patch_size, C)
        pos_token=pos_token.permute(0,2,3,1).reshape(-1, t, new_size, new_size, embedding_size)
        pos_token=pos_token.flatten(start_dim=1,end_dim=3) # (B, [num_frames//tube_size *H//patch_size * W//patch_size], C)-> (B,L,C)
        new_pos_embed=torch.cat((extra_tokens, pos_token), dim=1) # combining (B,x,C) extra_token with (B,L,C)
        return new_pos_embed


def positional_embedding_image_size_change(pos_tokens, tubelet_size, input_size, patch_size, org_num_frames=16):
    """Modify the size of positional embedding (i.e., number of patches) according to the new image size
    Args:
        pos_tokens (torch.Tensor):  Positional embeddeding of the current model of shape (1, num_patches, embed_dim)
        tubelet_size (int): Temporal size of patch
        input_size (int): New input size (assuming a squared image)
        patch_size (int): Patch size (assuming the same squared patch size) 
        org_num_frames (int): Original number of frames (assuming does not change)
    Returns:
        (torch.Tensor): Positional embeddeding of shape (1, new_num_patches, embed_dim) after modify number of patches based on the new image size
    """
    assert all(isinstance(x, numbers.Number) for x in [input_size, patch_size, tubelet_size])
    
    T=org_num_frames//tubelet_size # number of patches along temporal direction 
    P=int((pos_tokens.shape[1]//T)**0.5) # number of patches along H and W direction 
    C=pos_tokens.shape[-1]
    new_P=input_size//patch_size
    # (B,L,C)=(B,[T*P*P],C)->(B,T,P,P,C)
    pos_tokens=pos_tokens.reshape(-1, T, P, P, C)
    # -> (B*T, P, P, C)->(B*T,C,P,P)
    pos_tokens=pos_tokens.reshape(-1, P,P,C).permute(0,3,1,2)
    pos_tokens=torch.nn.functional.interpolate(pos_tokens, size=(new_P, new_P), mode='bicubic', align_corners=False)
    # -> (B*T,P,P,C)->(B,T,P,P,C)
    pos_tokens=pos_tokens.permute(0,2,3,1).reshape(-1, T, new_P, new_P, C)
    pos_tokens=pos_tokens.flatten(1,3) # (B,[T*P*P],C)=(B,L,C)
    return pos_tokens

def positional_embedding_num_frames_change(pos_tokens, num_frames, tubelet_size, org_num_frames=16):
    """Linearly interpolate the positional embedding of the current model along the temporal dimension to adjust for change in the number of frames
    Args:
        pos_tokens (torch.Tensor):  Positional embeddeding of the current model of shape (1, num_patches, embed_dim)
        num_frames (int): New number of frames
        tubelet_size (int): Temporal size of patch
        org_num_frames (int): Original number of frames
    Returns:
        (torch.Tensor): Positional embeddeding of shape (1, new_num_patches, embed_dim) after modify number of patches based on the new image size
    """
    T=org_num_frames//tubelet_size
    new_T=num_frames//tubelet_size
    P=int((pos_tokens.shape[1]//T)**0.5)
    C=pos_tokens.shape[-1]
    pos_tokens=pos_tokens.reshape(-1,T,P,P,C) # (B,[T*P*P],C)->(B,T,P,P,C)
    # -> (B,P,P,C,T) -> ([B*P*P],C,T)
    pos_tokens=pos_tokens.permute(0,2,3,4,1).reshape(-1,C,T) 
    pos_tokens=torch.nn.functional.interpolate(pos_tokens, size=new_T, mode='linear')
    # -> (B,P,P,C,T) -> (B,T,P,P,C)
    pos_tokens=pos_tokens.reshape(-1,P,P,C,new_T).permute(0,4,1,2,3)
    pos_tokens=pos_tokens.flatten(1,3) # (B,[T*P*P],C)
    return pos_tokens

def load_pretrained_model(args, model, weight_fpath, org_num_frames=16):
    """Load pretrained weights to model to perform finetuning
    Args:
        args (Namespace): Arguments
        model (nn.Module): Model to load pretrained parameter values to
        weight_fpath (str|Path): Path to a file storing the pretrained model parameter values
        
    """
    # ###
    # print('In load_pretrain.load_pretrained_model', flush=True)
    # initial_model_params={name:param.clone() for name, param in model.named_parameters()}
    
    checkpoint=torch.load(weight_fpath, map_location='cpu', weights_only=False)
    checkpoint_model=checkpoint['model'] if 'model' in checkpoint else checkpoint['module']
    state_dict=model.state_dict()
    
    all_keys=list(checkpoint_model.keys())
    new_dict=OrderedDict()
    for key in all_keys:
        if key.startswith('encoder.'): 
            new_key=key[len('encoder.'):]
            if new_key in state_dict:
                if checkpoint_model[key].shape!=state_dict[new_key].shape: continue
                else: new_dict[new_key]=checkpoint_model[key]
            else:
                print('load_pretrain.load_pretrained_model: ', key[len('encoder.'):], ' is in model state_dict: ', key[len('encoder.'):] in state_dict)
        elif checkpoint_model[key].shape==state_dict[key].shape:
            new_dict[key]=checkpoint_model[key]
    checkpoint_model=new_dict
    
    # interpolate learned position embedding
    if 'pos_embed' in checkpoint_model:
        # checkpoint_model['pos_embed'] of shape (1, num_patches, embed_dim)
        new_pos_embed=learned_positional_embedding_size_change(model, pos_embed_checkpoint=checkpoint_model['pos_embed'], num_frames=args.num_frames, 
                                                              tubelet_size=model.patch_embed.tubelet_size)
        if new_pos_embed is not None: checkpoint_model['pos_embed']=new_pos_embed
    elif args.input_size!=244: 
        # model.pos_embed of shape (1, num_patches, embed_dim)
        model.pos_embed=positional_embedding_image_size_change(pos_tokens=model.pos_embed,tubelet_size=args.tubelet_size,
                                                               input_size=args.input_size, patch_size=args.patch_size[0], org_num_frames=org_num_frames)
    if args.num_frames!=org_num_frames: 
        model.pos_embed=positional_embedding_num_frames_change(pos_tokens=model.pos_embed, num_frames=args.num_frames, tubelet_size=args.tubelet_size, 
                                                               org_num_frames=org_num_frames)
    
    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    # ####
    # for name, param in model.named_parameters():
    #     same=torch.allclose(param, initial_model_params[name])
    #     if not same: print(f"{name} does change", flush=True)