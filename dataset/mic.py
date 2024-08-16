import sounddevice as sd
import wave
import os
import time
def receive_audio(dataset_folder, fs=48000, duration=5, channels=2):
    '''
    receive by sounddevice and save the audio data
    '''
    # Set the parameters
    sd.default.device = 1
    default_channels = sd.query_devices(sd.default.device[1])['max_input_channels']
    default_sample_rate = sd.query_devices(sd.default.device[1])['default_samplerate']
    if default_sample_rate != fs:
        fs = default_sample_rate
    # assert the recording channels equal to the channels
    assert default_channels == channels, 'The recording channels is not equal to the channels'
    filename = os.path.join(dataset_folder, f'{str(time.time())}_{channels}.wav')
    # Record the audio
    print('Recording audio start...')
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait()
    print('Recording audio done...')
    # Save the audio
    print('Saving audio...')
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(channels) 
    waveFile.setsampwidth(2)
    waveFile.setframerate(fs)
    waveFile.writeframes(myrecording)
    waveFile.close()

if __name__ == "__main__":
    receive_audio()