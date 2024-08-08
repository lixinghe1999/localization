import sounddevice as sd
import wave
def receive_audio():
    '''
    receive by sounddevice and save the audio data
    '''
    # Set the parameters
    sd.default.device = 1
    fs = 48000
    duration = 5  # seconds
    channels = 8
    filename = 'output.wav'
    # Record the audio
    print('Recording...')
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait()
    print('Recording done...')
    # Save the audio
    print('Saving...')
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(channels) 
    waveFile.setsampwidth(2)
    waveFile.setframerate(fs)
    waveFile.writeframes(myrecording)
    waveFile.close()

if __name__ == "__main__":
    receive_audio()