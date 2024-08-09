import sounddevice as sd
import wave
import os
import time
def receive_audio(dataset_folder, duration=5):
    '''
    receive by sounddevice and save the audio data
    '''
    # Set the parameters
    sd.default.device = 1
    fs = 48000
    channels = 8
    filename = os.path.join(dataset_folder, f'{str(time.time())}_glasses.wav')
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