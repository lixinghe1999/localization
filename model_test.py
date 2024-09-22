from models.audio_models import Sound_Event_Detector
import torch

if __name__ == "__main__":
    template_x = torch.randn(2, 80000)


    model_name = 'mn10_as'
    model = Sound_Event_Detector(model_name)
    y, feature = model(template_x)

    print(y.shape, feature.shape)