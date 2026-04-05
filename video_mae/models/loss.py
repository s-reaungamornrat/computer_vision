"""Cross Entropy with smoothing or soft targets

Hacked together by /Copyright 2021 Ross Wightman
https://github.com/huggingface/pytorch-image-models/blob/main/timm/loss/cross_entropy.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """NLL loss with label smoothing"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing<1.0
        self.smoothing=smoothing
        self.confidence=1.-self.smoothing
        
    def forward(self, x:torch.Tensor, target:torch.Tensor)->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Logic output from a network of shape (B, num_classes) and of type float32
            target (torch.Tensor): Label of shape (B,) of type long
        Returns:
            (torch.Tensor): Differentiable loss
        """
        B,num_classes=x.shape
        neg_logprobs=-F.log_softmax(x, dim=-1) # (B, num_classes)

        # minimize negative log probability (log likihood) is equivalent to maximize probability
        smooth_loss=(neg_logprobs.sum(dim=-1))/float(max(num_classes,1)) # (B,num_classes) to (B,)
        
        # (B, 1) : gather values at index along dim=-1
        nll_loss=neg_logprobs.gather(dim=-1, index=target.unsqueeze(1)) # mismatch between the model output and the actual data
        nll_loss=nll_loss.squeeze(1) # (B,)

        loss=self.confidence*nll_loss + self.smoothing*smooth_loss
        return loss.sum()/float(max(B,1))


class SoftTargetCrossEntropy(nn.Module):
    
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
        
    def forward(self, x:torch.Tensor, target:torch.Tensor)->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Logic output from a network of shape (B, num_classes) and of type float32
            target (torch.Tensor): Smoothed one-hot encoding of shape (B, num_classes) and of type float32
        Returns:
            (torch.Tensor): Differentiable loss
        """
        loss=torch.sum(-target*F.log_softmax(x, dim=-1), dim=-1) # (B,)
        return loss.sum()/float(max(x.shape[0],1))