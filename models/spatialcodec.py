import json
from models.SpatialCodec.models import Quantizer
from models.SpatialCodec.spatial_model_subband import SpatialEncoder, SpatialDecoder
from models.SpatialCodec.frequency_codec import Encoder, Generator
# from SpatialCodec.quantizer import Quantizer
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import soundfile
import librosa
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_cov_matrix(array_audio, nfft=640, hop_length=320):
    '''
    array_audio: n_channels, n_samples
    '''
    array_stft = torch.stft(array_audio, nfft, hop_length, win_length=nfft, window=torch.hann_window(nfft).to(array_audio.device),
                    center=True, pad_mode='constant', normalized=False, onesided=True, return_complex=True) # n_channels, F, T
    ref_stft = array_stft[3].unsqueeze(0)
    n_channels, n_freqs, n_frames = array_stft.shape
    
    array_stft = array_stft.permute(1,2,0).unsqueeze(2) # F, T, 1, n_channels
    array_stft_t = array_stft.permute(0,1,3,2) # F, T, n_channels, 1
    
    cov_matrix = torch.matmul(array_stft_t, torch.conj(array_stft)) # F, T, n_channels, n_channels
    
    mask = torch.ones(n_channels, n_channels).to(array_audio.device)
    mask = (torch.triu(mask)==1)
    cov_matrix_upper = cov_matrix[:, :, mask]
    cov_matrix_upper = torch.view_as_real(cov_matrix_upper).permute(2,3,0,1).reshape(-1, n_freqs, n_frames)
    ref_stft = torch.view_as_real(ref_stft)
    
    features = torch.cat([cov_matrix_upper, ref_stft.squeeze(0).permute(2,0,1)], dim=0)
    return cov_matrix_upper, ref_stft, features

class SpatialCodec_Model(nn.Module):
    def __init__(self, n_mics=8):
        super(SpatialCodec_Model, self).__init__()
        h = AttrDict({"n_code_groups": 6,
                    "n_codes": 1024,
                    "residual_layer": 2,
                    "codebook_loss_lambda": 1.0,
                    "commitment_loss_lambda": 0.25,})
        self.quantizer = Quantizer(h, n_dim=256*6)
        # self.quantizer = Quantizer(code_dim=256*6, codebook_num=2, codebook_size=512)
        self.encoder = SpatialEncoder(channels=[(n_mics + 1) * n_mics + 2, 64, 64, 128, 128, 256, 256])
        self.decoder = SpatialDecoder(n_mics=n_mics-1)

        self.ref_quantizer = Quantizer(h, n_dim=256*6)
        # self.ref_quantizer = Quantizer(code_dim=256*6, codebook_num=2, codebook_size=512)

        self.ref_encoder = Encoder()
        self.ref_decoder = Generator()

    def ref_encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        ref_emb = self.ref_encoder(x[:, 3, ...].unsqueeze(1))
        q_ref, _, _ = self.ref_quantizer(ref_emb)
        return q_ref
    
    def spatial_encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, C, T = x.shape
        covs = []
        for i in range(B):
            _, _, cov = get_cov_matrix(x[i])
            covs.append(cov)
        cov = torch.stack(covs, dim=0)
        
        ref_emb = self.ref_encoder(x[:, 3, ...].unsqueeze(1))
        q_ref, _, _ = self.ref_quantizer(ref_emb)

        # encoder input: # bs, 72, F, T
        cov = cov.permute(0,1,3,2)
        c_emb = self.encoder(cov) # bs, 9*512, n_frames
        q_rtf, loss_q_rtf, _ = self.quantizer(c_emb) # bs, 22*512, n_frames
        return q_ref, q_rtf

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, C, T = x.shape
        covs = []
        for i in range(B):
            _, _, cov = get_cov_matrix(x[i])
            covs.append(cov)
        cov = torch.stack(covs, dim=0)
        
        bs, _, n_freqs, n_frames = cov.shape # bs, 72, F, T
        ref_emb = self.ref_encoder(x[:, 3, ...].unsqueeze(1))
        q_ref, _, _ = self.ref_quantizer(ref_emb)
        ref_g = self.ref_decoder(q_ref) # bs, 1, n_samples
        
        ref_g_stft = torch.stft(ref_g.reshape(bs, -1), 640, hop_length=320, win_length=640, window=torch.hann_window(640).to(x.device),
                        center=True, pad_mode='constant', normalized=False, onesided=True, return_complex=True) # bs, F, T
        
        # encoder input: # bs, 72, F, T
        cov = cov.permute(0,1,3,2)
        c_emb = self.encoder(cov) # bs, 9*512, n_frames
        q_rtf, loss_q_rtf, _ = self.quantizer(c_emb) # bs, 22*512, n_frames
        # decode
        rtf_g = self.decoder(q_rtf) # bs, 7, n_freqs, 5, n_frames, 2
        # decide to use est ref (real)
        ref_stft = ref_g_stft.unsqueeze(1)
        ref_stft_df = F.unfold(ref_stft, kernel_size=(3, 9), padding=(1,4)).reshape(bs, 1, 3,9,n_freqs, n_frames) # bs, 7, 3, 9, n_freqs, n_frames
        rtf_g = torch.view_as_complex(rtf_g).contiguous() # [2, 7, 3, 9, 321, 100]
        est_stfts = (rtf_g * ref_stft_df).sum(2).sum(2) # bs, 7, n_freqs, n_framesest_stfts
        # est_stfts = rtf_g * ref_stft
        est_stfts = est_stfts.reshape(bs*(C-1), n_freqs, n_frames)
        est_reverb_clean = torch.istft(est_stfts, 640, 320, 640, window=torch.hann_window(640).to(est_stfts.device)) # bs*7, T
        est_reverb_clean = est_reverb_clean.reshape(bs, (C-1), -1)        
        valid_len = np.minimum(est_reverb_clean.shape[-1], ref_g.shape[-1])

        
        est_reverb_clean_all = torch.cat([est_reverb_clean[:, :3, :valid_len], ref_g, est_reverb_clean[:, 3:, :valid_len]], dim=1) # bs, 8, n_samples
        est_reverb_clean_all = est_reverb_clean_all[0].T.detach().cpu().numpy()       
        return est_reverb_clean_all

if __name__ == '__main__':
    audio, fs = librosa.load('./SA1.wav', sr=16000, mono=True)
    audio = np.repeat(audio[np.newaxis, :], 8, axis=0)
    audio = torch.tensor(audio).to('cuda')

    model = SpatialCodec_Model(n_mics=8).to('cuda')

    ckpt = torch.load('../resources/SpatialCodec_ckpt.ckpt', map_location='cuda', weights_only=True)
    model.encoder.load_state_dict(ckpt['encoder'], strict=True)
    model.decoder.load_state_dict(ckpt['decoder'], strict=True)
    model.quantizer.load_state_dict(ckpt['quantizer'], strict=True)

    ckpt = torch.load('../resources/ref_subband_codec_ckpt.ckpt', map_location='cuda', weights_only=True)
    model.ref_encoder.load_state_dict(ckpt['encoder'], strict=True)
    model.ref_decoder.load_state_dict(ckpt['generator'], strict=True)
    model.ref_quantizer.load_state_dict(ckpt['quantizer'], strict=True)

    output = model(audio)
    soundfile.write('output.wav', output, 16000)
    

