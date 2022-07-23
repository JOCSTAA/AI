import cv2
import Hear
import numpy as np

class Input():
    def __init__(self, lstm_cnt):
        self.eyes = cv2.VideoCapture(0)
        self.ears = Hear.Hear()
        self.ears.start_listening()
        self.lstm_cnt = lstm_cnt
        self.eyes_time_sequence_observation = []
        self.ears_time_sequence_observation = []

    def observe(self):
        ret, image = self.eyes.read()
        audio = self.ears.read()
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



