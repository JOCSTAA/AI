import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..", "AI")
sys.path.append(main_scope)

import numpy as np
from models.transformers.decoder import TransformerDecoderClass

import unittest

def Setup_and_run_model( width = 224,
                height = 224,
                seq_length = 256,
                model_dimension = 256,
                batch_size = 2):
    model = TransformerDecoderClass(model_dimension=256, sequence_length=seq_length)

    sample_sequenced_images_for_input_1 = np.random.rand(
        batch_size, seq_length, height, width, 3)
    sample_sequenced_images_for_input_2 = np.random.rand(
        batch_size, seq_length, height, width, 1)
    sample_sequenced_inputs = (
        sample_sequenced_images_for_input_1, sample_sequenced_images_for_input_2)
    sample_encoder_output = np.random.rand(batch_size, seq_length, model_dimension)

    sample_output = model.predict([sample_sequenced_inputs, sample_encoder_output])
    return sample_output, model

class TestDecoder(unittest.TestCase):

    def testValidDecoderOutputShape(self):
        batch_size = 2
        seq_length = 256
        model_dimension = 256
        model_output, _ = Setup_and_run_model(batch_size = batch_size,
                                              seq_length = seq_length,
                                              model_dimension = model_dimension)
        expected_output_shape = (batch_size, seq_length, model_dimension)
        self.assertEqual(model_output.shape, expected_output_shape)


def test_decoder():
    sample_output, model = Setup_and_run_model()
    print(sample_output.shape)
    model.summary()

# test_decoder()