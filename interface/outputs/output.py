import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..", "AI")
sys.path.append(main_scope)

import interface.outputs.mouth.speak as Speak


class Output():
    def __init__(self,
                 action_space,
                 mouth_shape=(512,512)
                 ):

        self.action_space = action_space
        self.mouth_shape = mouth_shape
        self.mouth = Speak.Speak(mouth_shape=mouth_shape)
        self.define_outputs()
        self.ValidateOutputArguments()

    def ValidateOutputArguments(self):
        if sum(self.output_sizes.values()) != self.action_space:
            raise ValueError(f"given number of output elements does not match required amount, {self.action_space} given and {sum(self.output_sizes.values())} required")
        
    def define_outputs(self):
        self.output_sizes = {
            "mouth": self.mouth_shape[0] * self.mouth_shape[1]
        }

        self.outputs = {
            "mouth": self.mouth
        }

    def act(self, nn_output):
        index = 0
        for output, size in self.output_sizes.items():
            self.outputs[output].write(nn_output[:,index: index + size])
            index += size

    def close_outputs(self):
        self.mouth.stop_speaking()
