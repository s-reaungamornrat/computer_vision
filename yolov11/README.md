Always freeze .dfl module.
```
for k, v in self.model.named_parameters():
    if '.dfl' in k: v.requires_grad=False
    elif not v.requires_grad and v.dtype.is_floating_point: 
        v.requires_grad=True # only floating point tensors can require gradients
```