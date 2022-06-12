import pyaudio
import time
import librosa
import librosa.display
import numpy as np
import math
import matplotlib.pyplot as plt
from threading import Thread

# shortest perceivable duration of wave a human can hear is 10ms , you wanna make each frame at least 10ms,
# each frame = (1/sample_rate) * number of samples per frame.
# we also wanna make number of samples in a frame a power of 2, for ease of computation
# using sample rate = 44100 and number of samples per frame = 512
# we get a frame duration of 0.0116s = 11.6 ms, just above the perceivable human hearing
class Hear():
    # Audio preprocessing and handling
    # Class that handles the audio reading for the AI, collects real time audio in sync with the images collected
    def __init__(self,
                 sample_rate = 44100,
                 n_mfcc = 20,
                 number_of_samples_per_frame = 512,
                 channels = 2):

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.waveform = []
        self.number_of_samples_per_frame = number_of_samples_per_frame
        self.channels = channels
        self.format = pyaudio.paInt32

    def listen(self):
        #initializes the process of collecting audio waveform and stores it in a dynamic array
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format, channels=self.channels, rate=self.sample_rate, input=True, output=True,
                        frames_per_buffer=self.number_of_samples_per_frame)

        while self.listening:
            frame = self.stream.read(self.number_of_samples_per_frame)
            self.waveform += frame

    def start_listening(self):
        #starts the listening thread
        self.listening = True
        self.thread = Thread(target=self.listen)
        self.thread.start()

    def read(self):
        #sends out the collected waveform up to the point it is called in an MFCC format
        while not self.waveform:
            time.sleep(0.01)

        current_mfcc = self.convert_to_MFCC(np.array(self.waveform, dtype=np.float32))
        self.waveform = []
        return current_mfcc

    def stop_listening(self):
        #closes the process and termintates the audio stream
        self.listening = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


    def convert_to_MFCC(self, waveform):
        #converts the current waveform to MFCC
        #MFCC is an image format
        #MFCC is a feature extraction of a sound wave that mimics the information collected by the human ear
        return(librosa.feature.mfcc(y=waveform, sr=self.sample_rate, n_mfcc=self.n_mfcc))

    def save_current_MFCC(self):
        MFCC = self.read()
        plt.figure(figsize=(8, 8))
        librosa.display.specshow(MFCC, sr=self.sample_rate, x_axis='time')
        plt.savefig('Temp/MFCCs.png')

    def show_current_MFCC(self):
        MFCC = self.read()
        plt.figure(figsize=(8, 8))
        librosa.display.specshow(MFCC, sr=self.sample_rate, x_axis='time')
        plt.show()