import torch
import torch.nn as nn
import torchvision

class CNNTransformers(nn.Module):
    
    def __init__(self, num_classes, d_model=512, n_head=8, n_frames=8, num_layers=3, pretrained=True, freeze_backbone=False): 
        """
        Args:
            num_classes (int): Number of output classes
            d_model (int): Number of transformer dimension
            n_head (int): Number of heads in the multiheadattention models
            n_frames (int): Number of video frames or sequence length
            num_layers (int): Number of sub-encoder-layers in the encoder 
            pretrained (bool): Whether to load pretrained backbone
            freeze_backbone (bool): Whether to freeze backbone
        """
        super(CNNTransformers, self).__init__()
        self.freeze_backbone=freeze_backbone
        # see pretrained weights at https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
        resnet=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        if freeze_backbone:
            for param in resnet.parameters(): param.requires_grad=False
            assert all(not param.requires_grad for param in resnet.parameters())

        self.backbone=nn.Sequential(*(list(resnet.children())[:-1]) ) # ended with AdaptiveAvgPool2d(output_size=(1, 1))
        # projection layer to match transformer d_model
        self.projection=nn.Linear(resnet.fc.in_features, d_model)
        # positional encoder
        self.pos_emb=nn.Parameter(torch.zeros(1, n_frames, d_model))
        # transformer encoder
        encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # classifier
        self.fc=nn.Linear(d_model, num_classes)

    def _freeze_backbone(self)->None:
        """Block all the backbone parameters from being optimized. Also prevent the running mean and running variance of 
        normalization from being updated"""
        for name, module in model.backbone.named_modules():
            module.eval()
            for param in module.parameters(): param.requires_grad=False

    def train(self,mode:bool=True)->None:
        """Set the optimization status when training"""
        super().train(mode)
        if self.freeze_backbone: self._freeze_backbone() # we turn backbone to eval that does not require gradients

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): Video frames of shape (N,T,C,H,W) where N is the batch size, T is the number of video frames
        Returns:
            (torch.Tensor): Raw, unnormalized output or logits with shape (N, num_classes)
        """
        batch_size, n_frames, C, H, W=inputs.shape
        # extract feature frame by frame then convert output from (B*F,backbone-dim, 1,1) to (B, F, backbone-dim) 
        # where B is batch size, F is the number of frames and backbone-dim is the size of feature of backbone
        feats=self.backbone(inputs.view(-1, C, H, W)).view(batch_size, n_frames, -1) 
        # project and add positional embeddings
        x=self.projection(feats)+self.pos_emb # (B,F,d_model)+(1,F,d_model) = (B,F,d_model)
        #transformer attention
        x=self.transformer_encoder(x) # (B,F,d_model)
        # global average pool across frames (or just take the first/last token)
        x=x.mean(dim=1) # (B, d_model)
        return self.fc(x) # (B, num_classes)

if __name__ == "__main__":

    n_frames=8
    model=CNNTransformers(num_classes=101, d_model=512, n_head=8, n_frames=n_frames, num_layers=3, pretrained=False, freeze_backbone=False)
    x=torch.rand(4,n_frames,3,224,224)
    out=model(x)
    print(f'{out.shape=}')
    nn.MSELoss()(out, torch.rand_like(out)).backward()