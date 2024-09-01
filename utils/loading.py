'''
Load custom dataset
'''
import os 
import scipy
import librosa
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataset.pixel2doa import undistorted_img, calculate_angles
def preprocess_imu(imu_file):
    '''
    imu: numpy array of shape (n, 7)
    [gx, gy, gz, ax, ay, az, t]
    '''
    def low_high_seperate(data):
        b_l, a_l = scipy.signal.butter(3, 20, 'low', fs=200)
        b_h, a_h = scipy.signal.butter(3, 20, 'high', fs=200)
        low = scipy.signal.filtfilt(b_l, a_l, data, axis=0)
        high = scipy.signal.filtfilt(b_h, a_h, data, axis=0)
        return low, high
    imu = np.loadtxt(imu_file)[:, :6]
    imu_low_pass, imu_high_pass = low_high_seperate(imu)
    imu_low_high = np.concatenate([imu_low_pass, imu_high_pass], axis=1)
    return imu_low_high  #[gx, gy, gz, ax, ay, az]_l + high

def load_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return np.stack(frames)
def localize_peaks_ir(ir_video):
    peak_loc = []
    for frame in ir_video:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # get the weight center
        M = cv2.moments(binary_img)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        peak_loc.append([cX, cY])
    return peak_loc
        # plt.imshow(binary_img)
        # plt.scatter(cX, cY, c='r')
        # plt.savefig('test.png')
        # break
def peak2loc(peak, depth):
    for (x, y), frame in zip(peak, depth):
        print(frame.shape)
        frame = undistorted_img(frame)
        depth_number = frame[y, x]
        plt.scatter(x, y, c='r')
        plt.imshow(frame)
        plt.savefig('test.png')
        break

def sync_data(data_dict):
    max_timestamp = max([data['timestamp'] for data in data_dict.values()])

    for key, data in data_dict.items():
        offset = float(max_timestamp) - float(data['timestamp']) # >= 0
        offset = int(offset * data['sr'])
        data['data'] = data['data'][offset:]
        # pad the data
        pad = np.zeros_like(data['data'])[:offset]
        data['data'] = np.concatenate([data['data'], pad], axis=0)
    return data_dict



def load_folder(folder):
    files = os.listdir(folder)
    files_parser = [os.path.splitext(file) for file in files]
    data_dict = {}
    for file_name, file_ext in files_parser:
        timestamp, sensor_id = file_name.split('_')
        if file_ext == '.wav':
            data_dict['audio_' + sensor_id]= {'data': librosa.load(os.path.join(folder, file_name + file_ext), sr=16000, mono=False)[0].T, 
                                              'timestamp': timestamp, 'sr': 16000}
        elif file_ext == '.txt':
            data_dict['imu_' + sensor_id] = {'data': preprocess_imu(os.path.join(folder, file_name + file_ext)), 'timestamp': timestamp, 'sr': 200}
        else:
            video = load_video(os.path.join(folder, file_name + file_ext))
            if sensor_id == 'ir':
                ir_peak = localize_peaks_ir(video)
            if sensor_id == 'depth':
                data_dict['video_' + sensor_id] = {'data': video, 'timestamp': timestamp, 'sr': 30}
    # peak2loc(ir_peak, depth_video)

  
    data_dict = sync_data(data_dict)
    # for key, data in data_dict.items():
    #     print(key, data['timestamp'], data['data'].shape)

    return data_dict