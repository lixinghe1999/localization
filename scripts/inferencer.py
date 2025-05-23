import numpy as np
import torch
import time
import sys
sys.path.append('..')
from models.localization.seldnet_model import SeldModel
from models.beamforming.deepbeam.deepbeam import BeamformerModel
from models.audio_models import Sound_Event_Detector
from models.SpatialCodec.spatialcodec import SpatialCodec_Model

from utils.beamforming_dataset import shift_mixture
from utils.window_feature import spectrogram, gcc_mel_spec 
from tqdm import tqdm

def test_latency(model, audio, func, device='cpu', num_test=100):
    warm_up = 10
    for i in tqdm(range(num_test)):
        if i == warm_up: # warm up
            start = time.time()
        _ = func(model, audio, device)
    end = time.time()
    total_time = end - start
    each_time = total_time / (num_test - warm_up)
    return each_time

def init_model(model_type='seldnet', ckpt_path=None):
    if model_type == 'seldnet':
        model = SeldModel(mic_channels=15, unique_classes=6, activation='tanh')
        if ckpt_path is None:
            pass
        else:
            if ckpt_path == 'default':
                ckpt_path = 'lightning_logs/audioset_2/checkpoints/epoch=9-step=18750.ckpt'
            ckpt = torch.load(ckpt_path, weights_only=True)['state_dict']
            ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
            print("SELDNet Model loaded")
    elif model_type == 'deepbeam':
        model = BeamformerModel(ch_in=5, synth_mid=64, synth_hid=96, block_size=16, kernel=3, synth_layer=4, synth_rep=4, lookahead=0)
        if ckpt_path is None:
            pass
        else:
            if ckpt_path == 'default':
                ckpt_path = 'lightning_logs/timit_beamforming/checkpoints/epoch=44-step=26010.ckpt'
            ckpt = torch.load(ckpt_path, weights_only=True)['state_dict']
            ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
            print("DeepBeam Model loaded")
    elif model_type == 'recognition':
        model = Sound_Event_Detector('mn10_as', 416, frame_duration=1)
        if ckpt_path is None:
            pass
        else:
            if ckpt_path == 'default':
                ckpt_path = 'lightning_logs/timit_beamforming/checkpoints/epoch=44-step=26010.ckpt'
            ckpt = torch.load(ckpt_path, weights_only=True)['state_dict']
            ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
            print("Recognition Model loaded")
    elif model_type == 'codec':
        model = SpatialCodec_Model(n_mics=5)
    
    return model

def inference_loc(model, data, device):
    spec = spectrogram(data)
    spatial_feature = gcc_mel_spec(spec)    
    spatial_feature = spatial_feature[None, ...]

    spatial_feature = torch.tensor(spatial_feature, device=device).float()
    output = model(spatial_feature)
    return output

def inference_beamforming(model, data, device):
    audio, target_pos = data
    audio, shift = shift_mixture(audio, target_pos, 16000)
    audio = torch.tensor(audio, device=device).float()[None, :, 8:]
    output = model(audio)
    return output

def inference_recognition(model, data, device):
    audio, context = data
    audio = torch.tensor(audio, device=device).float()
    output = model(audio)
    return output

def inference_codec(model, data, device):
    audio = torch.tensor(data, device=device).float()
    output = model(audio)
    return output
if __name__ == '__main__':
    N_channel = 5
    duration = 5
    fs = 16000
    device = 'cuda'

    # model = init_model(model_type='codec', ckpt_path=None)
    # dummpy_audio = np.random.randn(N_channel, duration*fs)
    # model.to(device)
    # latency = test_latency(model, dummpy_audio, inference_codec, device, num_test=50)
    # print(f"Latency: {latency}, RTF: {latency / duration}")


    model = init_model(model_type='seldnet', ckpt_path=None)
    dummpy_audio = np.random.randn(N_channel, duration*fs)
    model.to(device)
    latency = test_latency(model, dummpy_audio, inference_loc, device, num_test=50)
    print(f"Latency: {latency}, RTF: {latency / duration}")


    # fs = 8000
    # model = init_model(model_type='deepbeam')
    # dummpy_audio = np.random.randn(N_channel, duration*fs)

    # device = 'cpu'
    # model.to(device)
    # azimuth = 90
    # target_pos = np.array([np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(azimuth))])
    # latency = test_latency(model, (dummpy_audio, target_pos), inference_beamforming, device, num_test=50)
    # print(f"Latency: {latency}, RTF: {latency / duration}")

    # device = 'cpu'
    # fs = 16000
    # model = init_model(model_type='recognition')
    # dummpy_audio = np.random.randn(1, duration*fs)
    # latency = test_latency(model, (dummpy_audio, None), inference_recognition, device, num_test=50)
    # print(f"Latency: {latency}, RTF: {latency / duration}")
