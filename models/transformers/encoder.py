import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..", "AI")
sys.path.append(main_scope)

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Layer
import models.embeddings.input_embedding as Embedder


class EncoderLayer(Layer):
    def __init__(self, model_dimension, num_heads, hidden_ff_dimension, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_ff_dimension // num_heads,
            dropout=dropout_rate
        )
        self.ffn = self.point_wise_feed_forward_network(model_dimension, hidden_ff_dimension)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def point_wise_feed_forward_network(self, model_dimension, hidden_ff_dimension):
        return Sequential([
            Dense(hidden_ff_dimension),
            Dense(model_dimension, activation='relu')
        ])

    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        norm1_output = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(norm1_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        encoder_layer_output = self.layernorm2(norm1_output + ffn_output)

        return encoder_layer_output


class TransformerEncoderClass(Model):

    def __init__(self, 
                 num_layers=6, 
                 model_dimension=256, 
                 num_heads=8, 
                 hidden_ff_dimension=1024, 
                 sequence_length=256, 
                 dropout_rate=0.1
                 ):
        super(TransformerEncoderClass, self).__init__()

        self.input_embedding = Embedder.TransformerInputEmbedding(model_dimension=model_dimension,
                                                                  sequence_length=sequence_length)
        self.input_embedding_drop_out = Dropout(dropout_rate)

        self.encoder = Sequential()
        for _ in range(num_layers):
            self.encoder.add(EncoderLayer(model_dimension, num_heads, hidden_ff_dimension, dropout_rate))
    
    def call(self, inputs, training):
        input_embedding = self.input_embedding_drop_out(
            self.input_embedding(inputs),
            training=training,
            )
        encoder_output = self.encoder(input_embedding)

        return encoder_output