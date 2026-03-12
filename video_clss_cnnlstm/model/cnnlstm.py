import torch
import torch.nn as nn
import torchvision

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to download and use pretrained weight
            freeze_backbone (bool): Whether to freeze backbone resnet
        Reference: https://github.com/pranoyr/cnn-lstm/blob/master/models/cnnlstm.py
        """
        super(CNNLSTM, self).__init__()
        self.freeze_backbone=freeze_backbone
        # see pretrained weights at https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
        self.resnet=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        if freeze_backbone:
            for param in self.resnet.parameters(): param.requires_grad=False
            assert all(not param.requires_grad for param in self.resnet.parameters())
            
        self.resnet.fc=nn.Sequential(nn.Linear(2048, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(512, 300)
                                     )
        #nn.Linear(self.resnet.fc.in_features, 300) # we allow training of the new resnet fc layer
        self.lstm=nn.LSTM(input_size=300, hidden_size=256, num_layers=3, batch_first=True)
        self.fc=nn.Linear(256, num_classes)

    def _freeze_backbone(self)->None:
        """Block all the backbone parameters from being optimized. Also prevent the running mean and running variance of normalization from being updated"""

        for name, child in self.resnet.named_children():
            if name=='fc': continue # we train this fc layer
            for n, m in child.named_modules():
                m.eval()
                for p in m.parameters(): p.requires_grad=False
                    
    def train(self, mode:bool=True)->None:
        """Set the optimization status when training"""
        super().train(mode)
        if self.freeze_backbone: 
            self._freeze_backbone() # we turn backbone to eval that does not require gradients
                    
    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): Video frames of shape (N,T,C,H,W) where N is the batch size, T is the number of video frames
        """
        batch_size, seq_len, C, H, W=inputs.shape
        # Combine batch and frames
        x=self.resnet(inputs.view(batch_size*seq_len, C, H, W))  # output-size: (batch_size*seq_len, resnet-dim)
        
        # Reshape back for LSTM (batch_size, seq_len, resnet-dim)
        x=x.view(batch_size, seq_len, -1)

        # For GPU efficiency, we flatten LSTM parameters
        self.lstm.flatten_parameters()

        # Passing features to LSTM
        x, (h_n, c_n)=self.lstm(x) # (batch_size, seq_len, lstm-dim), (num-layers,batch_size, hidden-dim), (num-layers,batch_size, hidden-dim)
        
        # Select the last frame output for classification with x[:,-1] of size (batch_size, lstm-dim)
        out=self.fc(x[:,-1]) # (N, num_classes)
        
        return out

if __name__ == "__main__":
    
    model=CNNLSTM(num_classes=101, pretrained=False, freeze_backbone=False)
    x=torch.rand(4,32,3,224,224)
    out=model(x)
    print(f'{out.shape=}')
    nn.MSELoss()(out, torch.rand_like(out)).backward()