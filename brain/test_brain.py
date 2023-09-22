import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..", "AI")
sys.path.append(main_scope)

from interface.inputs import input as Input
import interface.outputs.output as output
import brain as brain

sequence_length = 3
eyes_shape = (4, 4, 3)
ears_shape = (4, 4, 1)
shape = 128
mouth_shape = (shape, shape) 
action_space = shape * shape # breaks at > 4096 # factor of mouth_shape
outputs = output.Output(action_space, mouth_shape=mouth_shape)
inputs = Input.Input(eyes_shape, ears_shape, sequence_length)

brain = brain.Brain(inputs,
                    outputs,
                    sequence_length,
                    action_space,
                    )

brain.run()