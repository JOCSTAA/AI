import unittest
import numpy as np
import models.embeddings.input_embedding as Embedder
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..", "AI")
sys.path.append(main_scope)


class TestOutputs(unittest.TestCase):

    def testValidImageEmbeddingShapeRGB(self):
        pixels = 3
        width = 10
        height = 10
        seq_length = 3
        batch_size = 3

        vector_length = 10

        sample_sequenced_images = np.random.rand(
            batch_size, seq_length, height, width, pixels)
        expected_shape = (batch_size, seq_length, vector_length)

        sample_embedder = Embedder.ImageEmbedding(vector_length=vector_length)
        sample_output = sample_embedder.predict(sample_sequenced_images)

        self.assertEqual(expected_shape, sample_output.shape)

    def testValidImageEmbeddingShapeBW(self):
        pixels = 1
        width = 10
        height = 10
        seq_length = 3
        batch_size = 3

        vector_length = 10

        sample_sequenced_images = np.random.rand(
            batch_size, seq_length, height, width, pixels)
        expected_shape = (batch_size, seq_length, vector_length)

        sample_embedder = Embedder.ImageEmbedding(vector_length=vector_length)
        sample_output = sample_embedder.predict(sample_sequenced_images)

        self.assertEqual(expected_shape, sample_output.shape)

    def testValidInputEmbeddingShape(self):
        width = 10
        height = 10
        seq_length = 3
        batch_size = 3

        model_dimension = 10

        sample_sequenced_images_for_input_1 = np.random.rand(
            batch_size, seq_length, height, width, 3)
        sample_sequenced_images_for_input_2 = np.random.rand(
            batch_size, seq_length, height, width, 1)
        sample_sequenced_inputs = (
            sample_sequenced_images_for_input_1, sample_sequenced_images_for_input_2)

        expected_shape = (batch_size, seq_length, model_dimension)

        sample_embedder = Embedder.InputEmbedding(
            model_dimension=model_dimension)
        sample_output = sample_embedder.predict(sample_sequenced_inputs)

        self.assertEqual(expected_shape, sample_output.shape)
    
    def testMismatchingModelDimensionsforInputs(self):
        model_dimension = 3

        with self.assertRaises(ValueError) as context:
            Embedder.InputEmbedding(model_dimension=model_dimension)

        self.assertIn("model dimensions must be equally divisible by the number of inputs", str(context.exception)) 


    def testValidTransformerInputEmbeddingShape(self):
        width = 10
        height = 10
        seq_length = 3
        batch_size = 3

        model_dimension = 4

        sample_sequenced_images_for_input_1 = np.random.rand(
            batch_size, seq_length, height, width, 3)
        sample_sequenced_images_for_input_2 = np.random.rand(
            batch_size, seq_length, height, width, 1)
        sample_sequenced_inputs = (
            sample_sequenced_images_for_input_1, sample_sequenced_images_for_input_2)

        expected_shape = (batch_size, seq_length, model_dimension)

        sample_embedder = Embedder.TransformerInputEmbedding(sequence_length=seq_length,
                                                             model_dimension=model_dimension,
                                                             conc_pos_enc=False)
        sample_output = sample_embedder.predict(sample_sequenced_inputs)

        self.assertEqual(expected_shape, sample_output.shape)

    def testValidTransformerInputEmbeddingwithConcatenatedPosEncShape(self):
        width = 10
        height = 10
        seq_length = 3
        batch_size = 3

        model_dimension = 4

        sample_sequenced_images_for_input_1 = np.random.rand(
            batch_size, seq_length, height, width, 3)
        sample_sequenced_images_for_input_2 = np.random.rand(
            batch_size, seq_length, height, width, 1)
        sample_sequenced_inputs = (
            sample_sequenced_images_for_input_1, sample_sequenced_images_for_input_2)

        expected_shape = (batch_size, seq_length, model_dimension)

        sample_embedder = Embedder.TransformerInputEmbedding(sequence_length=seq_length,
                                                             model_dimension=model_dimension,
                                                             )
        sample_output = sample_embedder.predict(sample_sequenced_inputs)

        self.assertEqual(expected_shape, sample_output.shape)

    def testMismatchingModelDimensionAndSequenceLengthForTransformerInputEmbeddingWithConcPosEnc(self):
        seq_length = 3
        model_dimension = 2

        with self.assertRaises(ValueError) as context:
            Embedder.TransformerInputEmbedding(sequence_length=seq_length,
                                                         model_dimension=model_dimension)

        self.assertIn("concatenated positional encoding was selected but the number of bits(which is a factor of sequence length) to be concatenated is greater than or equal to the available space required for model dimension", str(context.exception)) 


def testImageEmbedding():
    sample_embedder = Embedder.ImageEmbedding(10)
    sample_embedder.build(input_shape=(None, None, 224, 224, 3))
    sample_embedder.summary()


def testInputEmbedding():
    sample_embedder = Embedder.InputEmbedding(100)
    sample_embedder.build(input_shape=(2, None, None, 224, 224, 3))
    sample_embedder.summary()


def testTransformerEmbedding():
    width = 10
    height = 10
    seq_length = 4
    batch_size = 2

    model_dimension = 4

    sample_sequenced_images_for_input_1 = np.random.rand(
        batch_size, seq_length, height, width, 3)
    sample_sequenced_images_for_input_2 = np.random.rand(
        batch_size, seq_length, height, width, 1)
    sample_sequenced_inputs = (
        sample_sequenced_images_for_input_1, sample_sequenced_images_for_input_2)

    sample_embedder = Embedder.TransformerInputEmbedding(sequence_length=seq_length,
                                                         model_dimension=model_dimension)
    sample_output = sample_embedder.predict(sample_sequenced_inputs)

    print(sample_output)


# testImageEmbedding()
# testInputEmbedding()
# testTransformerEmbedding()
