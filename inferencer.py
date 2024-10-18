'''
Running all the audio-based techniques on in-the-wild dataset, including
1. sound localization based on SELD or pra
2. beamforming
3. audio recognition
'''
import numpy as np
from utils.doa import init, inference
import os
from utils.frame_audio_dataset import AudioSet_dataset
from models.audio_models import Sound_Event_Detector
import torch
import librosa
from models.seldnet_model import SeldModel
from utils.window_feature import spectrogram, gcc_mel_spec 
class ALL_IN_ONE_Inferencer():
    def __init__(self,mic_array, fs=16000, window=0.1):
        self.mic_array = mic_array
        self.fs = fs
        self.window = window
        
    def init_doa(self):
        print('Initializing DOA by signal processing...')
        sample_window = int(self.window * self.fs)
        doa = init(self.mic_array, self.fs, sample_window, algorithm='music')
        doa_inference = inference
        return doa, doa_inference

    def init_cls(self):
        print('Initializing audio recognition...')
        root = os.path.join('dataset', 'audioset')
        dataset = AudioSet_dataset(root=root, split='train', frame_duration=1, vision=False, label_level='frame')
        model = Sound_Event_Detector('mn40_as', len(dataset.class_map), frame_duration=1)
        checkpoint = 'lightning_logs/frame_1_mn40_full/checkpoints/epoch=9-step=44160.ckpt'

        # dataset = AudioSet_dataset(root=root, split='eval', frame_duration=1, vision=False, label_level='frame')
        # model = Sound_Event_Detector('mn10_as', len(dataset.class_map), frame_duration=None)
        # checkpoint = 'lightning_logs/clip/checkpoints/epoch=29-step=24210.ckpt'

        model.class_map = dataset.class_map

        checkpoint = torch.load(checkpoint, weights_only=True)['state_dict']
        # remove the prefix 'model.'
        checkpoint = {k[6:]: v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=True) # make sure it is strict
        model.eval()
        print('Audio recognition model loaded...')

        def cls_inference(model, audio, sr=16000, window=1):
            '''
            audio [time]
            '''
            if type(audio) == np.ndarray:
                audio = torch.from_numpy(audio).float()

            sample_window = window * sr
            # pad zeros to make sure the audio can be divided by sample_window
            pad_len = sample_window - audio.shape[-1] % sample_window
            if pad_len > 0:
                audio = torch.cat([audio, torch.zeros(pad_len)])
            # audio = audio / torch.max(torch.abs(audio), dim=-1, keepdim=True)[0]
            audio = audio.reshape(-1, sample_window)
            y_hat, _ = model(audio) # [batch, time, class]
            y_hat = y_hat.reshape(-1, len(model.class_map))
            # convert to class_name
            y = torch.sigmoid(y_hat)
            y = y.detach().cpu().numpy() > 0.5
            y_class = []
            for b in range(y.shape[0]):
                y_class.append([])
                for c in range(y.shape[1]):
                    if y[b, c]:
                        y_class[b].append(model.class_map[c])
            return y_class
        return model, cls_inference

    def init_clap(self):
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False, device=torch.device('cuda:0'))
        model.load_ckpt()
        def inference_clap_audio(model, audio):
            batch, time = audio.shape
            sample_window = self.fs * 10
            if time % sample_window != 0:
                pad_len = sample_window - time % sample_window
                audio = np.concatenate([audio, np.zeros((batch, pad_len))], axis=-1)
            audio = audio.reshape(-1, sample_window)
            audio_embed = model.get_audio_embedding_from_data(x = audio, use_tensor=False)
            return audio_embed
        def inference_clap_text(model, text_data):
            # Get text embedings from texts:
            text_embed = model.get_text_embedding(text_data)
            return text_embed
        return model, inference_clap_audio, inference_clap_text
    
    def init_dist(self):
        model = SeldModel(mic_channels=3, unique_classes=26, activation=None)
        checkpoint = 'lightning_logs/seldnet/checkpoints/epoch=49-step=6000.ckpt'
        checkpoint = torch.load(checkpoint, weights_only=True)['state_dict']
        # remove the prefix 'model.'
        checkpoint = {k[6:]: v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        print("SeldNet loaded...")

        def dist_inference(model, audio):
            '''
            audio [2, time]
            '''
            spec = spectrogram(audio)
            gcc = gcc_mel_spec(spec).astype(np.float32)
            gcc = torch.from_numpy(gcc).unsqueeze(0) # [batch, channel, time, freq]
            y = model(gcc) # [batch, time, class]
            y = torch.sigmoid(y)
            y_dist = y[:, :, 6:8].detach().cpu().numpy() # [batch, time, 2]
            y_dist = y_dist > 0.5
            return y_dist
        return model, dist_inference

    def run(self, model, algo, audio):
        print('Running the ALL-IN-ONE inferencer...')
        predictions = algo(model, audio)
        return predictions
    
if __name__ == '__main__':
    inferencer = ALL_IN_ONE_Inferencer(mic_array=np.array([[-0.1, 0], [0.1, 0]]))
    
    dataset_dir = 'dataset/earphone/20241017/audio'
    data_files = os.listdir(dataset_dir)
    audio_files = [os.path.join(dataset_dir, file) for file in data_files if file.endswith('.wav')]
    imu_files = [os.path.join(dataset_dir, file) for file in data_files if file.endswith('.npy')]

    audio_files.sort()
    imu_files.sort()
    audio_files = audio_files[:1]
    imu_files = imu_files[:1]
    for audio_file, imu_file in zip(audio_files, imu_files):
        print('Processing:', audio_file, imu_file)
        audio, sr = librosa.load(audio_file, sr=16000, mono=False)
        imu = np.load(imu_file)

    # audio, sr = librosa.load('__LU8E6dUsI_40000.flac', sr=16000, mono=True)
    # audio, sr = librosa.load('__p-iA312kg_70000.flac', sr=16000, mono=True)
    # audio, sr = librosa.load('__tbvLNH6FI_110000.flac', sr=16000, mono=True)
    # audio, sr = librosa.load('__uEjp7_UDw_100000.flac', sr=16000, mono=True)
    # audio = audio[np.newaxis, :]

    # clap, clap_audio, clap_text = inferencer.init_clap()
    # audio_embed = clap_audio(clap, audio[:1])
    # text_embed = clap_text(clap, ['smooth music', 'cat', 'dog', 'femal speech'])
    # audio_text_cosine = np.dot(audio_embed, text_embed.T) / (np.linalg.norm(audio_embed) * np.linalg.norm(text_embed))
    # print('Audio-Text cosine similarity:', audio_text_cosine)

    doa, doa_inference = inferencer.init_doa()  
    predictions = inferencer.run(doa, doa_inference, (audio, imu))
    # print('DOA predictions:', predictions)

    # cls, cls_inference = inferencer.init_cls()
    # predictions = inferencer.run(cls, cls_inference, audio[0])
    # print('cls predictions:', predictions)

    # dist, dist_inference = inferencer.init_dist()
    # predictions = inferencer.run(dist, dist_inference, audio)
    # print('Distance predictions:', predictions)