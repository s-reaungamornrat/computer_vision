import torch

def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k
    Args:
        output (torch.Tensor): Raw logit or class scores after softmax of type float and size (B, n_classes) where B is the batch size
        target (torch.Tensor): Ground -truth class label of type int64 and size (B,)
    Returns:
        (list[torch.Tensor]): List of topk accuracies whose length=`len(topk)`
    """

    with torch.inference_mode():
        maxk=max(topk)
        batch_size=target.size(0)
        if target.ndim==2: target=target.argmax(dim=1) #max(dim=1)[1]  
        _, pred_idx=output.topk(maxk, dim=1, largest=True, sorted=True) # Bxk where k is the max(topk)
        pred_idx=pred_idx.t() # (k,B)
        correct=pred_idx.eq(target[None]) # (k,B).eq((1,B))=(k,B)
    
        res=[]
        for k in topk:
            correct_k=correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k*(100./batch_size)) # percent of correctly-identified batch
        return res