import torch
import torch.nn.functional as F


class Frame_MobileNet(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # copy all the layers of backbone
        for name, module in backbone.named_children():
            self.add_module(name, module)

    def _forward(self, x, return_fmaps: bool = False):
        '''
        Modified from MobileNet
        '''
        fmaps = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if return_fmaps:
                fmaps.append(x)
        
        # split the time dimension

        features = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        x = self.classifier(x).squeeze()
        
        if features.dim() == 1 and x.dim() == 1:
            # squeezed batch dimension
            features = features.unsqueeze(0)
            x = x.unsqueeze(0)
        
        if return_fmaps:
            return x, fmaps
        else:
            return x, features
    
    def forward(self, x, return_fmaps: bool = False):
        B, C, Freq, T = x.shape
        _T = T//50
        x = x.view(B*_T, 1, Freq, 50)
        x, features = self._forward(x, return_fmaps=return_fmaps)
        x = x.view(B, _T, -1)
        features = features.view(B, _T, -1)
        return x, features