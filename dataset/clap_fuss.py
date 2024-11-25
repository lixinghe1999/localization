import laion_clap
import os
from tqdm import tqdm
import numpy as np
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt() # download the default pretrained checkpoint.

folder = 'FUSS/ssdata'
for split in ['eval', 'validation', 'train']:
    folder_split = f'{folder}/{split}'
    files = os.listdir(folder_split)
    files = [f for f in files if f.endswith('sources')]
    for f in tqdm(files):
        source_folder = f'{folder_split}/{f}'
        sources = [f'{source_folder}/{s}' for s in os.listdir(source_folder)]
        audio_embeddings = model.get_audio_embedding_from_filelist(x=sources, use_tensor=False)
        embedding_file = f'{folder_split}/{f}.npy'
        np.save(embedding_file, audio_embeddings)
    print(f'Finished {split} of {folder}')
    