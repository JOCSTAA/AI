import Speak
import numpy as np

class Output():
    def __init__(self):
        self.mouth = Speak.Speak()
        self.mouth.start_speaking()
        self.mouth.write(np.array([0, 0]))

    def act(self, sound):
        self.mouth.write(sound)