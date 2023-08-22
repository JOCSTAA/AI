import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..", "AI")
sys.path.append(main_scope)

import interface.outputs.mouth.speak as Speak
import numpy as np

class Output():
    def __init__(self):
        self.mouth = Speak.Speak()
        self.mouth.start_speaking()
        self.mouth.write(np.array([0, 0]))

    def act(self, sound):
        self.mouth.write(sound)