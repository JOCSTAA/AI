import cv2
import Inputs.Ears.Hear as Hear
import numpy as np

class Input():
    def __init__(self, eyes_shape, ears_shape, lstm_cnt):
        self.eyes = cv2.VideoCapture(0)
        self.ears = Hear.Hear()
        self.ears.start_listening()
        self.eyes_shape = eyes_shape
        self.ears_shape = ears_shape
        self.lstm_cnt = lstm_cnt
        self.eyes_time_sequence_observation = []
        self.ears_time_sequence_observation = []

        if len(self.eyes_shape) == 3:
            self.eyes_shape = self.eyes_shape[:2]
        if len(self.ears_shape) == 3:
            self.ears_shape = self.ears_shape[:2]

    def observe(self):
        ret, image = self.eyes.read()
        audio = self.ears.read()
        image = cv2.resize(image, self.eyes_shape)
        audio = cv2.resize(audio, self.ears_shape)
        return [image, audio]

    def shape(self):
        image, audio = self.observe()
        return image.shape, audio.shape

    def collect_input(self):
        if not self.eyes_time_sequence_observation:
            for _ in range(self.lstm_cnt):
                observation = self.observe()
                self.eyes_time_sequence_observation.append(observation[0])
                self.ears_time_sequence_observation.append(observation[1])

        else:
            observation = self.observe()
            self.eyes_time_sequence_observation.pop(0)
            self.eyes_time_sequence_observation.append(observation[0])
            self.ears_time_sequence_observation.pop(0)
            self.ears_time_sequence_observation.append(observation[1])

        input_to_predict = (
        np.array(self.eyes_time_sequence_observation), np.array(self.ears_time_sequence_observation))
        return input_to_predict



