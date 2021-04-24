from tensorflow import keras
from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError


def create_hyperband_model(hp):
    dropout = hp.Float('dropout', 0.05, 0.4, sampling='log')
    image_size = 128  # We'll resize input images to this size
    # Size of the patches to be extract from the input images
    patch_size = hp.Int('transformer_layers', 4, 8, step=2)
    num_patches = (image_size // patch_size) ** 2
    projection_dim = hp.Int('projection_dim', 8, 40, step=8)
    num_heads = hp.Int('attention_heads', 2, 8, step=2)
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = hp.Int('transformer_layers', 2, 6, step=1)
    mlp_head_units = hp.Int('mlp_head_units', 10, 20, step=2)
    model = create_model()
    return model


def create_model(num_rows_df: int,
                 num_output_fields: int,
                 window_size: int = 10,
                 num_neurons: int = 40,
                 weight_decay_kernel: float = 1e-3,
                 weight_decay_recurrent: float = 1e-3,
                 kernel_dropout: float = 0.01,
                 recurrent_dropout: float = 0.2,
                 second_lstm_layer: bool = False):

    input_layer = Input(shape=(window_size, num_rows_df))

    lstm = LSTM(num_neurons,
                recurrent_regularizer=regularizers.l2(weight_decay_recurrent),
                kernel_regularizer=regularizers.l2(weight_decay_kernel),
                recurrent_dropout=recurrent_dropout,
                dropout=kernel_dropout,
                return_sequences=second_lstm_layer)(input_layer)

    if second_lstm_layer:
        lstm = LSTM(num_neurons,
                    recurrent_regularizer=regularizers.l2(
                        weight_decay_recurrent),
                    kernel_regularizer=regularizers.l2(weight_decay_kernel),
                    recurrent_dropout=recurrent_dropout,
                    dropout=kernel_dropout)(lstm)

    pred = Dense(num_output_fields, activation='relu')(lstm)

    model = Model(inputs=input_layer, outputs=pred)
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=[MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)])
    model.summary()
    return model
