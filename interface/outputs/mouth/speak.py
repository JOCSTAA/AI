import numpy as np
from matplotlib import pyplot as plt
import pyaudio
import librosa


class Speak():
    def __init__(self,
                 mouth_shape=(512,512),
                 sample_rate=44000,
                 channels=2,
                 number_of_samples_per_frame = 512):
        self.nn_output_shape = mouth_shape[0] * mouth_shape[1]
        self.spectogram_frequency_bins = mouth_shape[0]
        self.channels = channels
        self.sample_rate = sample_rate
        self.number_of_samples_per_frame = number_of_samples_per_frame
        self.restart_playback = False
        self.waveform_index = 0

        self.p = pyaudio.PyAudio()
        self.waveform = self._silence_waveform()
        self.talk()

    def talk(self):
        def callback(_, frame_count, __, ___):
            if self.restart_playback:
                self.waveform_index = 0
                self.restart_playback = False

            audio_data = self.waveform[self.waveform_index:self.waveform_index + frame_count]
            if len(audio_data) <= frame_count:
                audio_data = np.concatenate((audio_data, self._silence_waveform()))

            self.waveform_index += frame_count
            
            return (audio_data.tobytes(), pyaudio.paContinue)

        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.number_of_samples_per_frame,
                                  stream_callback=callback)
        
    def write(self, nn_output):
        sequenced_waveform=[]
        for nn_output_frame in nn_output:
            self.validate_nn_output(nn_output_frame)
            waveform = self.process_audio(nn_output_frame)
            sequenced_waveform.extend(waveform)
        self.waveform = sequenced_waveform
        self.restart_playback = True

    def validate_nn_output(self, nn_output):
        if nn_output is None:
            raise ValueError(f"model output can not be empty, no output provided")
        if nn_output.ndim != 1:
            raise ValueError(f"Invalid output shape, ({self.nn_output_shape},) required {nn_output.shape} given")
        if len(nn_output) != self.nn_output_shape:
            raise ValueError(f"Invalid output size from model, {self.nn_output_shape} required {len(nn_output)} given")
        if np.any(nn_output == None):
            raise ValueError(f"Model output contains a none element")

    def process_audio(self, nn_output):
        spectogram = nn_output.reshape(self.spectogram_frequency_bins, -1)
        waveform = librosa.istft(spectogram)
        return waveform

    def stop_speaking(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def _silence_waveform(self):
        return np.zeros(10000, dtype=np.float32)
    
    def speak_spectogram(self, spectogram):
        self.waveform = librosa.istft(spectogram)
        self.restart_playback = True
