from models.seldnet_model import SeldModel
from models.deepbeam import BeamformerModel

import numpy as np

class DeafSpace():
    def __init__(self, ):
        pass
    def init_localization(self, ):
        self.localization_model = SeldModel(mic_channels=15, unique_classes=3)
        def inference_localization(self, model, audio, source=1):
            outputs = model(audio)
            B, T, N = outputs.shape # [batch, time, source*3(xyz)]
            assert source == N // 3
            outputs = outputs.reshape(B, T, source, 3)
            pred_sed = ((outputs**2).sum(dim=-1, keepdims=True)) ** 0.5 > 0.5  # [batch, time, source]
            outputs = outputs * pred_sed
            return outputs
        return inference_localization
    
    def init_beamforming(self, ):
        model = BeamformerModel(ch_in=4, synth_mid=64, synth_hid=96, block_size=16, kernel=3, synth_layer=4, synth_rep=4, lookahead=0)

    def init_recognition(self, ):
        pass
    
    def forward(self, audio, vision, imu):
        return NotImplementedError
    
if __name__ == '__main__':
    deafspace = DeafSpace()