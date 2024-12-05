import os
import numpy as np
import soundfile as sf

dataset_dir = 'adhoc/measurement'
data_files = os.listdir(dataset_dir)
pcm_files = [f for f in data_files if f.endswith('.pcm')]

# load pcm file and convert to numpy array, two-channel audio
for pcm_file in pcm_files:
    pcm_file = os.path.join(dataset_dir, pcm_file)
    with open(pcm_file, 'rb') as f:
        pcm_data = f.read()
        pcm_data = np.frombuffer(pcm_data, dtype=np.int16)
        pcm_data = pcm_data.reshape(-1, 2)
        
    sf.write(pcm_file.replace('.pcm', '.wav'), pcm_data, 48000)
    