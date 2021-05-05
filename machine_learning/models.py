from tensorflow.keras.layers import LSTM, Input, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.recurrent import GRU
from tensorflow.keras import metrics

HYPER_NUM_ROWS_DF = None
HYPER_NUM_OUTPUT_FIELDS = None
HYPER_WINDOW_SIZE = 10
HYPER_LOOK_AHEAD_SIZE = 5


def create_hyperband_model(hp):
    kernel_dropout = hp.Float('kernel_dropout', 0.0, 0.3, sampling='linear')
    recurrent_dropout = hp.Float(
        'recurrent_dropout', 0.0, 0.3, sampling='linear')
    second_layer = hp.Choice('second_lstm_layer', [False, True])
    weight_decay_kernel = hp.Choice(
        'weight_decay_kernel', [0.0, 1e-5, 1e-3])
    weight_decay_recurrent = hp.Choice('weight_decay_recurrent',
                                       [0.0, 1e-5, 1e-3])
    num_neurons = hp.Int('num_neurons', 16, 64, step=8)
    learning_rate = hp.Float(
        'learning_rate', 1e-3, 5e-2, sampling='linear')
    loss = hp.Choice('loss', ['binary_accuracy', 'accuracy'])

    model = create_model(num_rows_df=HYPER_NUM_ROWS_DF,
                         num_output_fields=HYPER_NUM_OUTPUT_FIELDS,
                         window_size=HYPER_WINDOW_SIZE,
                         look_ahead_size=HYPER_LOOK_AHEAD_SIZE,
                         num_neurons=num_neurons,
                         weight_decay_kernel=weight_decay_kernel,
                         weight_decay_recurrent=weight_decay_recurrent,
                         kernel_dropout=kernel_dropout,
                         recurrent_dropout=recurrent_dropout,
                         second_lstm_layer=second_layer,
                         learning_rate=learning_rate,
                         loss=loss)
    return model


def create_model(num_rows_df: int,
                 num_output_fields: int,
                 window_size: int = 10,
                 look_ahead_size: int = 5,
                 num_neurons: int = 40,
                 weight_decay_kernel: float = 1e-4,
                 weight_decay_recurrent: float = 1e-3,
                 kernel_dropout: float = 0.1,
                 recurrent_dropout: float = 0.3,
                 second_lstm_layer: bool = False,
                 learning_rate: float = 0.01,
                 loss: str = 'mse'):

    input_layer = Input(shape=(window_size, num_rows_df))
    norm = keras.layers.LayerNormalization()(input_layer)
    encoder = LSTM(num_neurons,
                   recurrent_regularizer=regularizers.l2(
                       weight_decay_recurrent),
                   kernel_regularizer=regularizers.l2(weight_decay_kernel),
                   recurrent_dropout=recurrent_dropout,
                   dropout=kernel_dropout,
                   return_sequences=second_lstm_layer)(norm)

    if second_lstm_layer:
        encoder = LSTM(num_neurons // 2,
                       recurrent_regularizer=regularizers.l2(
                           weight_decay_recurrent),
                       kernel_regularizer=regularizers.l2(weight_decay_kernel),
                       recurrent_dropout=recurrent_dropout,
                       dropout=kernel_dropout)(encoder)

    repeat = RepeatVector(look_ahead_size)(encoder)
    decoder = LSTM(num_neurons,
                   return_sequences=True)(repeat)
    pred = TimeDistributed(
        Dense(num_output_fields, activation='relu'))(decoder)

    model = Model(inputs=input_layer, outputs=pred)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss,
                  metrics=[MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM),
                  'accuracy'])
    model.summary()
    return model
