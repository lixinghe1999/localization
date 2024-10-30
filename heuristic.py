from utils.localization_dataset import Localization_dataset
import matplotlib.pyplot as plt
import numpy as np
def visualize_starss23(label):
    '''
    label: [time, (event, x, y, z, class_idx) + (event, x, y, z, class_idx) + (event, x, y, z, class_idx)]
    '''
    T, N = label.shape
    num_source = N // 5
    label = label.reshape(T, num_source, 5)
    for t in range(T):
        for s in range(num_source):
            if label[t, s, 0]:
                plt.scatter(t, label[t, s, -1], c='r')
    plt.savefig('starss23.png')

def pesudo_result(label, num_class=13):
    '''
    label: [time, (event, x, y, z, class_idx) + (event, x, y, z, class_idx) + (event, x, y, z, class_idx)]

    Return:
    detected sound location: [time, (event, x, y, z) + (event, x, y, z) + (event, x, y, z)]
    frame-level sound classification: [time, class_idx] 
    '''
    T, N = label.shape
    num_source = N // 5
    label = label.reshape(T, num_source, 5)
    sound_location = label[:, :, :4]
    permutation = np.random.permutation(num_source)
    sound_location = sound_location[:, permutation, :].reshape(T, num_source * 4)


    sound_classification = np.zeros((T, num_class))
    for t in range(T):
        for s in range(num_source):
            if label[t, s, 0]:
                sound_classification[t, int(label[t, s, -1])] = 1
    return sound_location, sound_classification
    

def simple_heuristic(sound_location, sound_classification):
    '''
    Given:
    detected sound location: [time, (event, x, y, z) + (event, x, y, z) + (event, x, y, z)]
    frame-level sound classification: [time, class_idx]
    Note that the sound_classification is multi-label
    the order of the sound_classification is "not" the same as the sound_location, if yes, it will be trivial

    Return:
    detected sound location: [time, (event, x, y, z, class_idx) + (event, x, y, z, class_idx) + (event, x, y, z, class_idx)]
    '''
    T, N = sound_location.shape
    num_source = N // 4
    sound_location = sound_location.reshape(T, num_source, 4)
    T, num_class = sound_classification.shape


    fig, axs = plt.subplots(2, 1)
    axs[0].set_title('Sound classification')
    axs[1].set_title('Sound location')


    refined_prediction = np.zeros((T, num_source, 5))
    for t in range(T):

        detected_class = np.where(sound_classification[t])[0]
        axs[0].scatter([t] * len(detected_class), detected_class, c='r')

        detected_sources = []
        for s in range(num_source):
            if sound_location[t, s, 0]:
                xyz = sound_location[t, s, 1:]
                azimuth = np.degrees(np.arctan2(xyz[1], xyz[0]))
                axs[1].scatter(t, azimuth, c='b')
                detected_sources.append(sound_location[t, s])
        

        if len(detected_class) == 0:
            continue
        elif len(detected_class) == 1: # only one sound detected, directly assign the class
            refined_prediction[t, 0, :4] = detected_sources[0]
            refined_prediction[t, 0, -1] = detected_class[0]
        else:
            for det in detected_class:
                
                for s in range(num_source):
                    if det == refined_prediction[t-1, s, -1]:
                        refined_prediction[t, s, :4] = detected_sources[s]
                        refined_prediction[t, s, -1] = det
                        break
        print(refined_prediction[t])

    plt.savefig('simple_heuristic.png')
        





if __name__ == '__main__':

    config = {
        "dataset": "smartglass",
        "train_datafolder": "/home/lixing/localization/dataset/starss23/dev-test-sony",
        "test_datafolder": "/home/lixing/localization/dataset/starss23/dev-test-sony",
        "cache_folder": None,
        "encoding": "Multi_ACCDOA",
        "duration": 5,
        "frame_duration": 0.1,
        "batch_size": 64,
        "epochs": 50,
        "model": "seldnet",
        "label_type": "framewise",
        "raw_audio": False,
        'num_channel': 15,
        'num_class': 1, # no need to do classification now
        "pretrained": False,
        "test": False,
        "class_names": [
                "Female speech, woman speaking",
                "Male speech, man speaking",
                "Clapping",
                "Telephone",
                "Laughter",
                "Domestic sounds",
                "Walk, footsteps",
                "Door, open or close",
                "Music",
                "Musical instrument",
                "Water tap, faucet",
                "Bell",
                "Knock"
            ],
        "motion": False,
    }
    print(config)

    train_dataset = Localization_dataset(config['train_datafolder'], config)
    for data in train_dataset:
        print(data['spatial_feature'].shape)
        print(data['audio'].shape)
        print(data['label'].shape)
        # visualize_starss23(data['label'])
        sound_location, sound_classification = pesudo_result(data['label'])
        simple_heuristic(sound_location, sound_classification)
        break
    