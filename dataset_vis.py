from utils.torch_dataset import Main_dataset, STARSS23_dataset
from utils.parameter import mic_array_starss23 as mic_array
from utils.tdoa import tdoa
from utils.doa import doa
import json
import soundfile as sf

def STARSS23_vis(datasample, class_names):
    import matplotlib.pyplot as plt
    import numpy as np

    audio = datasample['audio']
    label = datasample['label']
    print(audio.shape, label)
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    mono_audio = audio[0]
    axs[0].plot(mono_audio)
    axs[0].set_title('Mono audio')

    for key in label:
        label_info = label[key]
        axs[1].plot(label_info['frame'], [key[0]] * len(label_info['frame']))
        axs[1].text(label_info['frame'][0], key[0], f'{label_info["location"]}')
    axs[1].set_yticks(range(len(class_names)), class_names)

    plt.savefig('STARSS23.png')
    sf.write('STARSS23.wav', mono_audio, 24000)
if __name__ == '__main__':
    # dataset = Main_dataset('simulate/TIMIT/1/hrtf_TEST', config=json.load(open('configs/binaural_TIMIT.json', 'r')))
    dataset = STARSS23_dataset('dataset/STARSS23', 'dev-test-sony', config=json.load(open('configs/gcc.json', 'r')))
    
    datasample = dataset.__getitem__(0)
    STARSS23_vis(datasample, dataset.class_names)
    # print(datasample['audio'].shape, datasample['label'])

    # # tdoa(datasample['audio'], plot=True)
    # doa(datasample['audio'], mic_array, fs=24000, nfft=1024, plot=True)


    