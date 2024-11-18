from models.mn.model import get_model as get_mobilenet_model
from models.mn.model import NAME_TO_WIDTH
from models.frame_mn import Frame_MobileNet, Frame_Conformer
from models.mn.preprocess import AugmentMelSTFT

import torch.nn as nn
import torch

class Sound_Event_Detector(nn.Module):
    def __init__(self, model_name = 'mn10_as', num_classes=527, frame_duration=None):
        super().__init__()
        self.preprocess = AugmentMelSTFT()
        self.backbone = get_mobilenet_model(num_classes=num_classes, pretrained_name=model_name, width_mult=NAME_TO_WIDTH(model_name), 
                                         strides=[2, 2, 2, 2], head_type='mlp')
        if frame_duration is not None:
            frame_length = int(50 * frame_duration)
            self.backbone = Frame_MobileNet(self.backbone, frame_length)

    def forward(self, x, return_fmaps=False):
        if isinstance(x, list):
            x, vision = x
            x = self.preprocess(x)
            x, feature = self.backbone(x.unsqueeze(1), vision, return_fmaps=return_fmaps)
        else:
            x = self.preprocess(x)
            x, feature = self.backbone(x.unsqueeze(1), return_fmaps=return_fmaps)
        
        return x, feature

if __name__ == "__main__":
    model_name = 'mn10_as'
    model = get_mobilenet_model(pretrained_name=model_name, width_mult=NAME_TO_WIDTH(model_name), 
                                         strides=[2, 2, 2, 2], head_type='mlp')
    preprocess = AugmentMelSTFT()
    x = torch.randn(1, 16000)
    x = preprocess(x)
    x, feature = model(x.unsqueeze(1), return_fmaps=True)
    