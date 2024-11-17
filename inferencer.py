import numpy as np
import torch
import time
from models.seldnet_model import SeldModel
from models.deepbeam import BeamformerModel
from utils.beamforming_dataset import shift_mixture

from utils.window_feature import spectrogram, gcc_mel_spec 

def init_model(model_type='seldnet', ckpt_path=None):
    if model_type == 'seldnet':
        model = SeldModel(mic_channels=15, unique_classes=6, activation='tanh')
        if ckpt_path is None:
            ckpt_path = 'lightning_logs/audioset_2/checkpoints/epoch=9-step=18750.ckpt'
        ckpt = torch.load(ckpt_path, weights_only=True)['state_dict']
        ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        print("SELDNet Model loaded")
    elif model_type == 'deepbeam':
        model = BeamformerModel(ch_in=5, synth_mid=64, synth_hid=96, block_size=16, kernel=3, synth_layer=4, synth_rep=4, lookahead=0)
        if ckpt_path is None:
            ckpt_path = 'lightning_logs/timit_beamforming/checkpoints/epoch=44-step=26010.ckpt'
        ckpt = torch.load(ckpt_path, weights_only=True)['state_dict']
        ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        print("DeepBeam Model loaded")
    elif model_type == 'recognition':
        raise NotImplementedError
    return model

def inference_loc(model, audio, num_test=1, device='cuda'):
    # preprocess audio
    spec = spectrogram(audio)
    spatial_feature = gcc_mel_spec(spec)    
    spatial_feature = spatial_feature[None, ...]

    spatial_feature = torch.tensor(spatial_feature, device=device).float()
    warm_up = 10
    if num_test > 1:
        for i in range(num_test):
            if i == warm_up: # warm up
                start = time.time()
            output = model(spatial_feature)
        end = time.time()
        print("Time taken for 100 iterations: ", (end - start)/(num_test - warm_up))
    else:
        output = model(spatial_feature)
    return output

def inference_beam(model, audio, target_pos, num_test=1, device='cuda'):
    audio, shift = shift_mixture(audio, target_pos, 16000)
    audio = torch.tensor(audio, device=device).float()[None, :, 8:]

    warm_up = 10
    if num_test > 1:
        for i in range(num_test):
            if i == warm_up: # warm up
                start = time.time()
            output = model(audio)
        end = time.time()
        print("Time taken for 100 iterations: ", (end - start)/(num_test - warm_up))
    else:
        output = model(audio)
    return output

if __name__ == '__main__':
    model = init_model(model_type='seldnet')

    dummpy_audio = np.random.randn(5, 80000)
    device = 'cuda'
    model.to(device)
    output = inference_loc(model, dummpy_audio, num_test=100, device=device)

    device = 'cpu'
    model.to(device)
    output = inference_loc(model, dummpy_audio, num_test=100, device=device)

    # model = init_model(model_type='deepbeam')
    # dummpy_audio = np.random.randn(5, 40000)
    # device = 'cuda'
    # model.to(device)
    # azimuth = 90
    # target_pos = np.array([np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(azimuth))])
    # output = inference_beam(model, dummpy_audio, target_pos=target_pos, num_test=100, device=device)