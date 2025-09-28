import contextlib 

import torch

from computer_vision.yolov11.utils.ops import make_divisible
from computer_vision.yolov11.modules import (Conv, C3k2, SPPF, C2PSA, Concat, Detect)

def parse_model(d, ch, verbose=True):
    '''
    Parse yolov11n only
    Args: 
        d (dict): Model dictionary
        ch (int): Input channel
        verbose (bool): Whether to print model construction process
    Returns:
        model (torch.nn.Sequential): PyTorch model
        save (list): Sorted list of feature indices need to be stored for skip connection
    '''
    import ast
    legacy=True # backward compatibility for v3/v5/v8/v9
    max_channels=float('inf')
    nc, act, scales=(d.get(x) for x in ('nc','activation', 'scales'))
    if verbose: print(f'nc: {nc}, act: {act}, scales: {scales}')
    depth, width, kpt_shape=(d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    scale=d.get('scale')
    if verbose: print(f'depth: {depth}, width: {width}, kpt_shape: {kpt_shape}, scale {scale}')
    if scales: depth, width, max_channels=scales[scale]
    if verbose: print(f'depth: {depth}, width: {width}, max_channels: {max_channels}')

    ch=[ch]
    layers, save, c2=[],[],ch[-1] # layers, savelist, ch out
    base_modules=frozenset({Conv, C3k2, SPPF, C2PSA})
    repeat_modules=frozenset({C3k2,C2PSA})

    for i, (f, n, m, args) in enumerate(d['backbone']+d['head']): # from, number, module, args
        if verbose: print(i, '-'*100)
        m=(
            getattr(torch.nn, m[3:]) if 'nn.' in m else \
            getattr(__import__('torchvision').ops, m[16:]) if 'torchvision.ops' in m else \
            globals()[m]
        ) # get module
        for j, a in enumerate(args):
            if not isinstance(a, str): continue
            with contextlib.suppress(ValueError):
                args[j]=locals()[a] if a in locals() else ast.literal_eval(a)
        # depth gain: repeats
        n=n_=max(round(n*depth), 1) if n>1 else n # always 1 for yolov11n
        if verbose: print(f'i: {i}, m: {m.__name__} f {f} args {args}')
        if m in base_modules:
            c1,c2=ch[f], args[0] # in-channels, out-channels
            if c2!=nc: # not equal to number of classes 
                c2=make_divisible(min(c2, max_channels)*width, 8)
            args=[c1, c2, *args[1:]]
            if verbose: print(f'\t baseline args: {args}, c2: {c2}') 
            if m in repeat_modules:
                args.insert(2, n) # number of repeates
                n=1
                if verbose: print('\targs repeate ',args)
            if m is C3k2: #for M/L/X sizes
                legacy=False
                if scale in 'mlx': args[3]=True
                if verbose: print('\targs C3k2 ',args)
        elif m is Concat:
            c2=sum(ch[x] for x in f)
            if verbose: print(f'\t cat args: {args}, c2: {c2}') 
        elif m in frozenset({Detect}):
            args.append([ch[x] for x in f])
            m.legacy=legacy
            if verbose: print(f'\t Detect args: {args}, m.legacy: {m.legacy}') 
        else:
            c2=ch[f]
            if verbose: print(f'\t others args: {args}, c2: {c2}') 
        m_=torch.nn.Sequential(*(m(*args) for _ in rnage(n))) if n>1 else m(*args)
        t=str(m)[str(m).rfind('.')+1:-2].replace('__main__.', '') # module type
        if verbose: print('t ', t)
        m_.np=sum(x.numel() for x in m_.parameters()) # number of parameters
        m_.i, m_.f, m_.type=i, f, t # index, `from` index, type
        save.extend(x%i for x in ([f] if isinstance(f, int) else f) if x!=-1)
        if verbose: print('save ', save)
        layers.append(m_)
        if i==0:  
            ch=[]
            if verbose: print(f'ch {ch}')
        ch.append(c2) # storing output channels
        if verbose: print(f'ch {ch}')
    return torch.nn.Sequential(*layers), sorted(save)
        