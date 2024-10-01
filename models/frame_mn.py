import torch
import torch.nn.functional as F


class Frame_MobileNet(torch.nn.Module):
    def __init__(self, backbone, frame_length: int = 50):
        super().__init__()
        # copy all the layers of backbone
        for name, module in backbone.named_children():
            self.add_module(name, module)
        self.frame_length = frame_length
        # self.vision_proj = torch.nn.Sequential(
        #     torch.nn.Linear(512, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 960)
        # )
        # self.vision_classifier = torch.nn.Linear(960, 416) # manual set
        self.condition_classifier = torch.nn.Linear(416 + 512, 416)
    def _forward_vision(self, x, vision):
        '''
        vision only output
        '''
        features = self.vision_proj(vision)
        x = self.classifier(x).squeeze()
        return x, features
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
            # vision = self.vision_proj(vision).squeeze()
            # vision_cls = self.vision_classifier(vision)
            # number_repeat = x.shape[0] // vision_cls.shape[0]
            # vision_cls = vision_cls.repeat(number_repeat, 1)
            # x = x + vision_cls
            number_repeat = x.shape[0] // vision.shape[0]
            vision = torch.repeat_interleave(vision, number_repeat, dim=0).squeeze()

            # vision = vision.reshape(-1, 416)
            # x = x * 0.01 + vision
            # x = vision
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