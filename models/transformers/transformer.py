import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..", "AI")
sys.path.append(main_scope)

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from models.transformers.encoder import TransformerEncoderClass
from models.transformers.decoder import TransformerDecoderClass

class Transformer(Model):

    def __init__(self, 
                 input_embedding_length,
                 sequence_length,
                 output_vector_length,
                 num_layers=6, 
                 num_heads=8, 
                 hidden_ff_dimension=1024,
                 dropout_rate=0.1
                 ):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoderClass(
            num_layers=num_layers,
            model_dimension=input_embedding_length,
            num_heads=num_heads,
            hidden_ff_dimension=hidden_ff_dimension,
            sequence_length=sequence_length,
            dropout_rate=dropout_rate,
        )

        self.decoder = TransformerDecoderClass(
            num_layers=num_layers,
            model_dimension=input_embedding_length,
            num_heads=num_heads,
            hidden_ff_dimension=hidden_ff_dimension,
            sequence_length=sequence_length,
            dropout_rate=dropout_rate,
        )

        self.output_ffn = Dense(output_vector_length, activation="tanh")

    def call(self, inputs, training):
        encoder_outputs = self.encoder(inputs, training)
        decoder_output = self.decoder([inputs, encoder_outputs], training)
        output = self.output_ffn(decoder_output)

        return output

