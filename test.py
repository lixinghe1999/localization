from asteroid.models import FasNetTAC
import torch


audio = torch.randn(2, 4, 16000)
model = FasNetTAC(n_src=2, sample_rate=16000)
estimation = model(audio)
print(estimation.shape)