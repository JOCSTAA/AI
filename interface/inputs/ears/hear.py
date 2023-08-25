import pyaudio
import time
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


# shortest perceivable duration of wave a human can hear is 10ms , you wanna make each frame at least 10ms,
# each frame = (1/sample_rate) * number of samples per frame.
# we also wanna make number of samples in a frame a power of 2, for ease of computation
# using sample rate = 44100 and number of samples per frame = 512
# we get a frame duration of 0.0116s = 11.6 ms, just above the perceivable human hearing


# Time taken between frames = O(1): time taken to remap `waveform = self.waveforms`
# this is most efficient time for collecting MFCC 
# this time is due to the streaming, audio is being collected in another stream, while
# audio processing is happening
class Hear():
    # Audio pre processing and handling
    # Class that handles the audio reading for the AI, collects real time audio in sync with the images collected
    def __init__(self,
                 image_size=(224, 224, 1),
                 sample_rate=44100,
                 n_mfcc=20,
                 number_of_samples_per_frame=512,
                 channels=2,
                 normalize = True,
                 standardize = False,
                 ):

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.waveforms = []
        self.number_of_samples_per_frame = number_of_samples_per_frame
        self.channels = channels
        self.format = pyaudio.paFloat32

        dpi = 100
        fig_width = image_size[0] / dpi
        fig_height = image_size[1] / dpi
        self.image_size = (fig_width, fig_height)

        if normalize & standardize:
            raise ValueError(f"can't use normalizaition and standardization for images")
        self.normalize = normalize
        self.standardize = standardize

        self.listen()

    def listen(self):
        self.p = pyaudio.PyAudio()
        def audio_callback(in_data, _, __, ___):
            self.waveforms.append(np.frombuffer(in_data, dtype=np.float32))
            return (None, pyaudio.paContinue)
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  input=True,
                                  output = False,
                                  frames_per_buffer=self.number_of_samples_per_frame,
                                  stream_callback=audio_callback)
        

    def read(self):
        # sends out the collected waveform up to the point it is called in an MFCC format
        while not self.waveforms:
            time.sleep(0.01)

        waveform = self.waveforms
        self.waveforms = []
        waveform = np.concatenate(waveform)

        self.process_audio(waveform)
        return self.current_mfcc_image

    def stop_listening(self):
        # closes the process and terminates the audio stream
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def process_audio(self, waveform):
        # convert waveform to MFCC and then MFCC to image with a specified image size so that input to neural network is of the same shape
        #
        # returns: a grayscale image in numpy array format with dimensions (IM_SIZE, IM_SIZE, 1).
        # image can be used directly in a CONV2D layer
        self.current_mfcc = librosa.feature.mfcc(y=waveform, sr=self.sample_rate, n_mfcc=self.n_mfcc)

        plt.figure(figsize=self.image_size)
        librosa.display.specshow(self.current_mfcc, sr=self.sample_rate, x_axis='time')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight', pad_inches=0)
        plt.close()
        buffer.seek(0)

        self.current_mfcc_image = np.expand_dims(np.array(Image.open(buffer).convert("L")), axis=-1)

        if self.normalize: self.current_mfcc_image = self.current_mfcc_image / 255.0

        if self.standardize:
            mean = np.mean(self.current_mfcc_image)
            std = np.std(self.current_mfcc_image)
        
            self.current_mfcc_image = (self.current_mfcc_image - mean) / std


    def show_current_MFCC(self):
        # collects and shows the image representation of the current recieved audio signal
        img = self.read()
        plt.imshow(img)
        plt.axis('off')
        plt.show()
