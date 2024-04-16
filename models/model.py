import torch.nn as nn
from models.deepear import DeepEAR_backbone, DeepEAR_classifier
from models.deepbsl import DeepBSL_backbone, DeepBSL_classifier
from models.deephrtf import DeepHRTF_backbone, DeepHRTF_classifier
import torch

class AudioClassifier(nn.Module):
    def __init__(self, backbone = 'DeepEAR', classifier = 'DeepEAR'):
        super(AudioClassifier, self).__init__()
        self.audiobackbone = globals()[backbone + '_backbone']()
        self.classifier = globals()[classifier + '_classifier']()
    def pretrained(self,):
        ckpt = torch.load('best_model_deepbsl.pth')
        self.load_state_dict(ckpt)
        for param in self.audiobackbone.parameters():
            param.requires_grad = False
    def forward(self, x):
        x = self.audiobackbone(x)
        x = self.classifier(x)
        return x


