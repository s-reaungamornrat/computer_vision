from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib import patches

import torch
import numpy as np

from .misc import smooth

def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] = {},
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
):
    """
    Plot metric-confidence curve.

    Args:
        px (np.ndarray): X values for the metric-confidence curve.
        py (np.ndarray): Y values for the metric-confidence curve.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    plt.rcParams.update({'font.size': 12})
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.1)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] = {}
):
    """
    Plot precision-recall curve.

    Args:
        px (np.ndarray): X values for the PR curve.
        py (np.ndarray): Y values for the PR curve.
        ap (np.ndarray): Average precision values.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    plt.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    
def plot_predictions(images:torch.Tensor, prediction:list[dict[str, torch.Tensor]], fname:str|Path, max_subplots:int=16, xywh:bool=False,
                     figsize:tuple[int,int]=(10,10)):
    """
    Plot predicted bounding boxes on image. The function checks if the maximum value of bounding box information is <=1.1, then it denormalizes
    the bounding box to pixel units.
    Args:
        images (torch.Tensor): BxCxHxW images associated with predictions
        prediction (list[dict[str, Any]]): list of the prediction dict for each image, each dict containing `bboxes`, `cls`, and `conf` as
            Nx4 tensor, (N,) class indices, and (N,) confidence
        fnames (str | Path): Path to save figure
        max_subplots (int): Maximum numbers of subplots
        xywh (bool): Whether the bounding box format is (x,y) for the center of box and (w,h) for the width and height
        figsize (tuple[int,int]): Size of figure
    """
    max_cmaps=10
    if isinstance(prediction, dict): max_cmaps=prediction['bboxes'].shape[0]
    elif isinstance(prediction, list): max_cmaps=max(l['bboxes'].shape[0] for l in prediction)
        
    cmap = plt.get_cmap('tab10', max_cmaps)
    plt.rcParams.update({'font.size': 18})
    
    if torch.max(images[0])<=1.: images*=255 # denormalize image
    # Handle 2 and n channel images
    c=images.shape[1]
    if c==2:
        zero=torch.zeros_like(images[:, :1]) # BxCxHxW
        images=torch.cat((images, zero), dim=1) # Bx3xHxW
    elif c>3: images=images[:,:3] # crop multispectral images to the first 3 channels
    
    bs=len(prediction)
    bs=min(bs, max_subplots) # limit plot images
    ns=int(np.ceil(bs**0.5)) # number of subplots (square)
    
    ncols=ns
    nrows=1 if ns*ns>bs and bs==ns else ns
    fig, ax=plt.subplots(nrows,ncols,figsize=figsize)
    for r in range(nrows):
        for c in range(ncols):
            indx=r*ncols+c
            if indx>len(prediction)-1: break
            pred=prediction[indx]
            assert all(x in pred for x in ['bboxes', 'cls', 'conf'])
            im=images[indx].permute(1,2,0).contiguous().cpu().numpy() # CxHxW to HxWxC
            h, w=im.shape[:2]
            if len(pred['bboxes'])>0 and pred['bboxes'].max()<= 1.1:  # if normalized with tolerance 0.1
                pred['bboxes'][:,[0,2]]*=w
                pred['bboxes'][:,[1,3]]*=h
                if xywh:
                    # convert xywh to xyxy
                    width_height=pred['bboxes'][:,2:]
                    pred['bboxes'][:,:2]-=width_height/2
                    pred['bboxes'][:,2:]=pred['bboxes'][:,:2]+width_height
            try: ax[r,c].imshow(im.astype(np.uint8))
            except: ax[indx].imshow(im.astype(np.uint8))
            if len(pred['bboxes'])==0: continue
            for j, (box, cls, conf) in enumerate(zip(pred['bboxes'], pred['cls'], pred['conf'])):
                # Create a Rectangle patch
                rect = patches.Rectangle(box[:2], box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor=cmap(j), facecolor='none')
                try:
                    ax[r,c].add_patch(rect)
                    t=ax[r,c].text(*box[2:], f'{int(cls.squeeze().item())}:{conf.squeeze().item():.3f}',
                                 color=cmap(j))#, fontsize=12)
                except:
                    ax[indx].add_patch(rect)
                    t=ax[indx].text(*box[2:], f'{int(cls.squeeze().item())}:{conf.squeeze().item():.3f}',
                                 color=cmap(j))#, fontsize=12)
                t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor=cmap(j)))
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    #plt.savefig(fname, transparent=None)
    fig.savefig(fname, dpi=250)
    plt.close(fig)

def plot_labels(labels: dict[str, Any], fname:str|Path, images:torch.Tensor=torch.zeros(0,3,640,640, dtype=torch.float32),
                      max_subplots:int=16, xywh:bool=True, figsize:tuple[int,int]=(10,10)):

    """
    Plot bounding box overlaid on image. The function checks if the maximum value of bounding box information is <=1.1, the function denormalizes
    the bounding box to pixel units.
    Args:
        labels (dict[str, Any]): Dict containing `bboxes`, `cls`, `img` and `conf` and `batch_idx` as optional. 
        fnames (str | Path): Path to save figure
        images (torch.Tensor): BxCxHxW images associated with labels if labels does not contain `img`.
        max_subplots (int): Maximum numbers of subplots
        xywh (bool): Whether the bounding box format is (x,y) for the center of box and (w,h) for the width and height
        figsize (tuple[int,int]): Size of figure
    """
    max_cmaps=10
    if isinstance(labels, dict): max_cmaps=labels['bboxes'].shape[0]
    elif isinstance(labels, list): max_cmaps=max(l['bboxes'].shape[0] for l in labels)
        
    cmap = plt.get_cmap('tab10', max_cmaps)
    plt.rcParams.update({'font.size'   : 18})
    
    classes=labels.get('cls', torch.zeros(0, dtype=torch.int64))
    bboxes=labels.get('bboxes', torch.zeros(classes.shape, dtype=torch.float32))
    confs=labels.get('conf', None)
    images=labels.get('img', images) # default to input images
    batch_idx=labels.get('batch_idx', None)

    if torch.max(images[0])<=1.: images*=255 # denormalize image
        
    # Handle 2 and n channel images
    c=images.shape[1]
    if c==2:
        zero=torch.zeros_like(images[:, :1]) # BxCxHxW
        images=torch.cat((images, zero), dim=1) # Bx3xHxW
    elif c>3: images=images[:,:3] # crop multispectral images to the first 3 channels
    
    bs,_,h,w=images.shape # batch_size, _, height, width
    bs=min(bs, max_subplots) # limit plot images
    ns=int(np.ceil(bs**0.5)) # number of subplots (square)
    
    ncols=ns
    nrows=1 if ns*ns>bs and bs==ns else ns
    fig, ax=plt.subplots(nrows,ncols,figsize=figsize)
    
    for r in range(nrows):
        for c in range(ncols):
            indx=r*ncols+c
            if indx>images.shape[0]-1: break
                
            this_im=images[indx] # CxHxW
            _, h, w=this_im.shape
            is_this=batch_idx==indx
            
            this_cls=classes[is_this] if classes is not None else None
            this_bbox=bboxes[is_this] if bboxes is not None else None
            this_conf=confs[is_this] if confs is not None else None
            if this_bbox is not None and this_bbox.max()<= 1.1:  # if normalized with tolerance 0.1
                this_bbox[:,[0,2]]*=w
                this_bbox[:,[1,3]]*=h
                if xywh:
                    # convert xywh to xyxy
                    width_height=this_bbox[:,2:]
                    this_bbox[:,:2]-=width_height/2
                    this_bbox[:,2:]=this_bbox[:,:2]+width_height
                
            try: ax[r,c].imshow(this_im.permute(1,2,0).cpu().numpy().astype(np.uint8))
            except: ax[indx].imshow(this_im.permute(1,2,0).cpu().numpy().astype(np.uint8))
            if this_bbox is None: continue
            for j, (box, cls) in enumerate(zip(this_bbox, this_cls)):
                conf=this_conf[j] if this_conf is not None else None
                # Create a Rectangle patch
                rect = patches.Rectangle(box[:2], box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor=cmap(j), facecolor='none')
                try:
                    ax[r,c].add_patch(rect)
                    t=ax[r,c].text(*box[2:], f'{int(cls.squeeze().item())}:{conf.squeeze().item() if isinstance(conf, torch.Tensor) else np.nan:.3f}',
                                 color=cmap(j))#, fontsize=12)
                except:
                    ax[indx].add_patch(rect)
                    t=ax[indx].text(*box[2:], f'{int(cls.squeeze().item())}:{conf.squeeze().item() if isinstance(conf, torch.Tensor) else np.nan:.3f}',
                                 color=cmap(j))#, fontsize=12)
                t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor=cmap(j)))
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    #plt.savefig(fname, transparent=None)
    fig.savefig(fname, dpi=250)
    plt.close(fig)
