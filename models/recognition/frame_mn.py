import torch
import torch.nn as nn
import torch.nn.functional as F
from models.recognition.model import get_model as get_mobilenet_model
from models.recognition.model import NAME_TO_WIDTH
from models.recognition.preprocess import AugmentMelSTFT

class Sound_Event_Detector(nn.Module):
    def __init__(self, model_name = 'mn10_as', num_classes=527, frame_duration=None):
        super().__init__()
        self.preprocess = AugmentMelSTFT()
        self.backbone = get_mobilenet_model(num_classes=num_classes, pretrained_name=model_name, width_mult=NAME_TO_WIDTH(model_name), 
                                         strides=[2, 2, 2, 2], head_type='mlp')
        if frame_duration is not None:
            frame_per_second = 50
            frame_length = int(frame_per_second * frame_duration)
            self.backbone = Frame_MobileNet(self.backbone, num_classes, frame_length)
        
    def forward(self, x, return_fmaps=False):
        if not isinstance(x, torch.Tensor):
            x, vision = x
            x = self.preprocess(x)
            x = self.backbone(x.unsqueeze(1), vision, return_fmaps=return_fmaps)
        else:
            x = self.preprocess(x)
            x = self.backbone(x.unsqueeze(1), return_fmaps=return_fmaps)
        
        return x

class Frame_MobileNet(torch.nn.Module):
    def __init__(self, backbone, num_classes, frame_length: int = 50):
        super().__init__()
        # copy all the layers of backbone
        self.features = backbone.features
        self.classifier = backbone.classifier
        self.frame_length = frame_length
        self.condition_classifier = torch.nn.Linear(num_classes + 512, num_classes)

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
            return x
    
    def forward(self, x, vision=None, return_fmaps: bool = False, ):
        B, C, Freq, T = x.shape
        _T = T//self.frame_length
        x = x.view(B*_T, 1, Freq, self.frame_length)
        x = self._forward(x, vision, return_fmaps=return_fmaps)
        x = x.view(B, _T, -1)
        return x
    

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