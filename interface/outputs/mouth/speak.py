import numpy as np
from matplotlib import pyplot as plt
import pyaudio
from threading import Thread

class Speak():
    def __init__(self,
                 max_frquency=20000,
                 min_frequency=20,
                 sample_rate=44000,
                 number_of_samples_per_frame = 512):
        self.max_frequency = max_frquency
        self.min_frequency = min_frequency
        self.duration = (1/sample_rate) * number_of_samples_per_frame
        self.sample_rate = sample_rate
        self.number_of_samples_per_frame = number_of_samples_per_frame
        self.p = pyaudio.PyAudio()

    def talk(self):
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  input= False,
                                  rate=self.sample_rate,
                                  frames_per_buffer=self.number_of_samples_per_frame,
                                  output=True)

        while self.talking:
            self.stream.write(self.sound_wave)

    def start_speaking(self):
        self.talking = True
        self.thread = Thread(target=self.talk)
        self.thread.start()

    def map_frequency(self, frequency):
        return np.interp(frequency, [0, 1], [self.min_frequency, self.max_frequency])

    def generate_sine_wave(self, frequency, loudness):
        frequency = self.map_frequency(frequency)
        return loudness * (
            np.sin(2 * np.pi * np.arange(self.sample_rate * self.duration) * frequency / self.sample_rate))

    def add_all_sine_wave_frequencies_into_one_waveform(self, input):
        sound_wave = self.generate_sine_wave(input[0], input[1])
        for i in range(2, len(input), 2):
            sine_wave = self.generate_sine_wave(input[i], input[i+1])
            sound_wave = sound_wave + sine_wave

        return sound_wave

    #takes in input as tensor withing the range (0 to 1), converts input tensor to sound waveform and produces sound
    def write(self, input):
        self.sound_wave = self.add_all_sine_wave_frequencies_into_one_waveform(input).astype(np.float32).tobytes()

    def stop_speaking(self):
        self.talking = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()