import numpy as np
from tensorflow import expand_dims, concat, shape, tile
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, TimeDistributed, MaxPooling2D, Flatten, Dense, concatenate


class ImageEmbedding(Model):
    """
    Image Embedding:
    Performs image embedding of the individual inputs

    Converts the time sequenced input images to time sequenced vectors.

    input_shape = batch_size x sequence_length x height x width x pixels
    output_shape = batch_size x sequence_length x vector_length
    """

    def __init__(self,
                 vector_length,
                 hidden_num_dense=64,
                 num_conv1=32,
                 conv1_filter_shape=(3, 3),
                 num_conv2=64,
                 conv2_filter_shape=(3, 3),
                 pool_size=(2, 2),
                 activation="relu",
                 output_activation="tanh"
                 ):
        super(ImageEmbedding, self).__init__()

        self.model = Sequential()
        self.model.add(TimeDistributed(Conv2D(num_conv1,
                                              conv1_filter_shape,
                                              activation=activation,
                                              padding="same")))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size,
                                                    padding="same")))
        self.model.add(TimeDistributed(Conv2D(num_conv2,
                                              conv2_filter_shape,
                                              activation=activation,
                                              padding="same")))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size,
                                                    padding="same")))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(TimeDistributed(
            Dense(hidden_num_dense, activation=activation)))
        self.model.add(TimeDistributed(
            Dense(vector_length, activation=output_activation)))

    def call(self, images):
        return self.model(images)


class InputEmbedding(Model):
    """
    Input Embedding:
    Performs the collected embedding of all inputs

    Converts all the time sequenced inputs to time sequenced vectors.

    Input = Time sequenced inputs
    Output = Time sequenced vectors of length model_dimension

    input_shape = batch_size x num_inputs x sequence_length x height x width x pixels
    output_shape = batch_size x sequence_length x model_dimension
    """

    def __init__(self,
                 model_dimension,
                 ):
        super(InputEmbedding, self).__init__()

        if model_dimension%2 != 0:
            raise ValueError(f"model dimensions must be equally divisible by the number of inputs, model dimensions :{model_dimension}, num of inputs : 2")
        eye_vector_length = int(model_dimension/2)
        ear_vector_length = int(model_dimension/2)

        self.eyes_embedding = ImageEmbedding(eye_vector_length)
        self.ears_embedding = ImageEmbedding(ear_vector_length)

    def call(self, observations):
        visual_images, audio_images = observations[0], observations[1]
        visual_vectors = self.eyes_embedding(visual_images)
        audio_vectors = self.ears_embedding(audio_images)
        input_vectors = concatenate((visual_vectors, audio_vectors))
        return input_vectors


class TransformerInputEmbedding(Model):
    """
    Transformer Embedding:
    Collects all inputs and preps them for the transformer layers

    - transforms each input given as pixels to a vector
    - encodes positional information to inputs

    Input = Time sequenced inputs
    Output = Positioned time sequenced inputs

    input_shape = batch_size x num_inputs x sequence_length x height x width x pixels
    output_shape = batch_size x sequence_length x model_dimension
    """

    def __init__(self,
                 sequence_length,
                 model_dimension,
                 conc_pos_enc=True
                 ):
        super(TransformerInputEmbedding, self).__init__()

        if conc_pos_enc:
            self.number_of_bits = int(np.ceil(np.log2(sequence_length)))
            if self.number_of_bits >= model_dimension:
                raise ValueError(
                    f"concatenated positional encoding was selected but the number of bits(which is a factor of sequence length) to be concatenated is greater than or equal to the available space required for model dimension, numbers of bits {self.number_of_bits}, model dimension provided: {model_dimension}")
            self.concatenate = True
            self.model_dimension = model_dimension - self.number_of_bits
            self.conc_positional_encoding(sequence_length)

        else:
            self.concatenate = False
            self.model_dimension = model_dimension
            self.sin_positional_encoding(sequence_length, model_dimension)

        self.input_embedding = InputEmbedding(self.model_dimension)

    def get_angles(self, sequence_length, i, model_dimension):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(model_dimension))
        return sequence_length * angle_rates

    def sin_positional_encoding(self, sequence_length, model_dimension):
        angle_rads = self.get_angles(np.arange(sequence_length)[:, np.newaxis],
                                np.arange(model_dimension)[np.newaxis, :],
                                model_dimension)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])   
        pos_encoding = angle_rads[np.newaxis, ...]
            
        self.pos_enc = pos_encoding

    def conc_positional_encoding(self, sequence_length):
        binary_indices = np.array([list(format(i, f'0{self.number_of_bits}b'))
                                   for i in range(sequence_length)], dtype=np.float32)

        self.pos_enc = binary_indices

    def encode_position(self, input_vectors):
        if self.concatenate:
            batch_size = shape(input_vectors)[0]
            expanded_bits = expand_dims(self.pos_enc, axis=0)
            expanded_bits = tile(expanded_bits, [batch_size, 1, 1])
            return concat([expanded_bits, input_vectors], axis=2)

        else:
            return input_vectors + self.pos_enc

    def call(self, observations):
        input_vectors = self.input_embedding(observations)
        input_vectors_with_positional_encoding = self.encode_position(
            input_vectors)
        return input_vectors_with_positional_encoding
