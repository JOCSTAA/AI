import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..", "AI")
sys.path.append(main_scope)

from tensorflow import ones
from tensorflow.linalg import band_part
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Layer
import models.embeddings.input_embedding as Embedder


class DecoderLayer(Layer):
    def __init__(self, 
                 model_dimension, 
                 num_heads, 
                 hidden_ff_dimension, 
                 look_ahead_mask,
                 dropout_rate=0.1,):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_ff_dimension // num_heads,
            dropout=dropout_rate
        )
        self.cross_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_ff_dimension // num_heads,
            dropout=dropout_rate
        )
        self.look_ahead_mask = look_ahead_mask
        self.ffn = self.point_wise_feed_forward_network(model_dimension, hidden_ff_dimension)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)
    
    def point_wise_feed_forward_network(self, model_dimension, hidden_ff_dimension):
        return Sequential([
            Dense(hidden_ff_dimension),
            Dense(model_dimension, activation='relu')
        ])

    def call(self, inputs_with_enc_output, training):
        inputs, enc_output = inputs_with_enc_output
        self_attn_output = self.self_attn(inputs, inputs, inputs, attention_mask=self.look_ahead_mask)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        norm1_output = self.layernorm1(inputs + self_attn_output)

        cross_attn_output = self.cross_attn(norm1_output, enc_output, enc_output)
        cross_attn_output = self.dropout2(cross_attn_output, training=training)
        norm2_output = self.layernorm2(norm1_output + cross_attn_output)

        ffn_output = self.ffn(norm2_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        decoder_layer_output = self.layernorm3(norm2_output + ffn_output)

        return decoder_layer_output


class TransformerDecoderClass(Model):

    def __init__(self, 
                 num_layers=6, 
                 model_dimension=256, 
                 num_heads=8, 
                 hidden_ff_dimension=1024, 
                 sequence_length=256, 
                 dropout_rate=0.1,
                 ):
        
        super(TransformerDecoderClass, self).__init__()

        self.input_embedding = Embedder.TransformerInputEmbedding(model_dimension=model_dimension,
                                                                  sequence_length=sequence_length)
        self.look_ahead_mask = self.create_look_ahead_mask(sequence_length)
        self.input_embedding_drop_out = Dropout(dropout_rate)

        self.decoder_layers = [DecoderLayer(model_dimension, num_heads, hidden_ff_dimension, self.look_ahead_mask, dropout_rate) for _ in range(num_layers)]

    def create_look_ahead_mask(self, sequence_length):
        mask = 1 - band_part(ones((sequence_length, sequence_length)), -1, 0)
        return mask
    
    def call(self, inputs_with_encoder_outputs, training=False):
        inputs, encoder_outputs = inputs_with_encoder_outputs
        x = self.input_embedding_drop_out(self.input_embedding(inputs))
        for layer in self.decoder_layers:
            x = layer((x, encoder_outputs), training)
        decoder_output = x

        return decoder_output
