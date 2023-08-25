import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..", "AI")
sys.path.append(main_scope)

import interface.inputs.eyes.see as See
import interface.inputs.ears.hear as Hear
import numpy as np

from collections import deque


class Input():
    def __init__(self, eyes_shape=(224, 224, 3), ears_shape=(224, 224, 1), lstm_cnt=3):

        if len(eyes_shape) != 3:
            raise ValueError(f"improper eyes image shape provided")
            
        if len(ears_shape) != 3:
            raise ValueError(f"improper eyes image shape provided")

        self.eyes = See.See(image_size=ears_shape)
        self.ears = Hear.Hear(image_size=ears_shape)

        self.eyes_shape = eyes_shape
        self.ears_shape = ears_shape
        self.lstm_cnt = lstm_cnt
        self.eyes_time_sequence_observation = deque(maxlen=lstm_cnt)
        self.ears_time_sequence_observation = deque(maxlen=lstm_cnt)
        self._prime_inputs()

    def _prime_inputs(self):
        for _ in range(self.lstm_cnt):
            self._collect_audio()
            self._collect_image()
    
    def _collect_image(self):
        # can be used in parrallel
        image = self.eyes.read()
        self.eyes_time_sequence_observation.append(image)
        return np.array(self.eyes_time_sequence_observation)

    def _collect_audio(self):
        # can be used in parrallel
        audio = self.ears.read()
        self.ears_time_sequence_observation.append(audio)
        return np.array(self.ears_time_sequence_observation)

    def close_inputs(self):
        self.eyes.stop_seeing()
        self.ears.stop_listening()

    def shape(self):
        return self.eyes_shape, self.ears_shape

    def collect_input(self):
        return(self._collect_image(), self._collect_audio())
