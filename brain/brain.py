import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..", "AI")
sys.path.append(main_scope)

from models.model import BrainModel as Model

class Brain():
    def __init__(self,
                 inputs,
                 outputs,
                 sequence_length,
                 action_space,
                 ):
        eyes_input_shape, ears_input_shape = inputs.shape()
        self.model = Model(eyes_input_shape, 
                           ears_input_shape, 
                           action_space, 
                           sequence_length,
                           )
        
        self.inputs = inputs
        self.outputs = outputs

    def run(self):
        import time
        while True:
            begin = time.time()

            outputs = self.model.predict(self.inputs.collect_inputs())
            self.outputs.act(outputs)
            self.model.update_model()

            end = time.time()
            duration = end - begin
            print("time taken :", duration, " seconds")