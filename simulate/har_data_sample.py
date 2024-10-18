from audio_dataset import ESC50
from simulator import active_frame
import matplotlib.pyplot as plt
import os

def plot_active(audio, mask, sr=16000):

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(audio)
    ax[1].plot(mask)
    plt.savefig('test.png')

dataset = ESC50(root='ESC-50-master', split='TRAIN')
print(len(dataset))

class_name_home = ['dog', 'cat',
            'water_drops', 'pouring_water', 'toilet_flush',
            'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
            'door knock', 'mouse click', 'keyboard typing', 'door_wood_knock', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking',
            ]
output_dir = 'ESC-50-home'
audio_dir = os.path.join(output_dir, 'audio')
meta_dir = os.path.join(output_dir, 'meta')




filtered_idx = []
for i in range(len(dataset)):
    audio, label, _ = dataset[i]
    class_name = dataset.class_name[label]
    if class_name in class_name_home:
        filtered_idx.append(i)
        audio, mask = active_frame(audio, frame=0.1, sr=dataset.sr)
        print(audio.shape, mask.shape, class_name)
        plot_active(audio, mask)    
        break
print(len(filtered_idx))