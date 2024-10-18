import torch
import torch.nn.functional as F


class Frame_MobileNet(torch.nn.Module):
    def __init__(self, backbone, frame_length: int = 50):
        super().__init__()
        # copy all the layers of backbone
        self.features = backbone.features
        self.classifier = backbone.classifier
        self.frame_length = frame_length
        self.condition_classifier = torch.nn.Linear(416 + 512, 416)

    def _forward(self, x, vision=None, return_fmaps: bool = False):
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
        
        if vision is not None:
            number_repeat = x.shape[0] // vision.shape[0]
            vision = torch.repeat_interleave(vision, number_repeat, dim=0).squeeze()
            x = self.condition_classifier(torch.cat([x, vision], dim=1))

        if features.dim() == 1 and x.dim() == 1:
            # squeezed batch dimension
            features = features.unsqueeze(0)
            x = x.unsqueeze(0)
        
        if return_fmaps:
            return x, fmaps
        else:
            return x, features
    
    def forward(self, x, vision=None, return_fmaps: bool = False, ):
        B, C, Freq, T = x.shape
        _T = T//self.frame_length
        x = x.view(B*_T, 1, Freq, self.frame_length)
        x, features = self._forward(x, vision, return_fmaps=return_fmaps)
        x = x.view(B, _T, -1)
        features = features.view(B, _T, -1)
        return x, features
    

class Frame_Conformer(torch.nn.Module):
    def __init__(self, backbone, frame_length: int = 50):
        super().__init__()
        # copy all the layers of backbone
        self.features = backbone.features
        self.classifier = backbone.classifier
        
        self.frame_length = frame_length
        self.condition_classifier = torch.nn.Linear(416 + 512, 416)

        transformer_layer = torch.nn.TransformerEncoderLayer(d_model=416, nhead=8)
        self.transformer_encoder = torch.nn.TransformerEncoder(transformer_layer, num_layers=4)
    def _forward(self, x, return_fmaps: bool = False):
        '''
        Modified from MobileNet
        '''
        fmaps = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if return_fmaps:
                fmaps.append(x)
        return x

    
    def forward(self, x, vision=None, return_fmaps: bool = False, ):
        B, C, Freq, T = x.shape
        _T = T//self.frame_length
        x = x.view(B*_T, 1, Freq, self.frame_length)
        features = self._forward(x, return_fmaps=return_fmaps)
        x = self.classifier(features)
        x = x.view(B, _T, -1) # B, T, C
        x = self.transformer_encoder(x)
        x = x.view(B*_T, -1)
        
        if vision is not None:
            number_repeat = x.shape[0] // vision.shape[0]
            vision = torch.repeat_interleave(vision, number_repeat, dim=0).squeeze()
            x = self.condition_classifier(torch.cat([x, vision], dim=1))
        x = x.view(B, _T, -1)
        return x, features