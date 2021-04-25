from tensorflow.keras.layers import LSTM, Input, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError

HYPER_NUM_ROWS_DF = None
HYPER_NUM_OUTPUT_FIELDS = None
HYPER_WINDOW_SIZE = 10
HYPER_LOOK_AHEAD_SIZE = 5


def create_hyperband_model(hp):
    kernel_dropout = hp.Float('kernel_dropout', 1e-6, 0.3, sampling='log')
    recurrent_dropout = hp.Float(
        'recurrent_dropout', 1e-6, 0.4, sampling='log')
    #second_layer = hp.Choice('second_lstm_layer', [True, False])
    weight_decay_kernel = hp.Float(
        'weight_decay_kernel', 1e-6, 1e-3, sampling='log')
    weight_decay_recurrent = hp.Float(
        'weight_decay_recurrent', 1e-6, 1e-3, sampling='log')
    num_neurons = hp.Int('num_neurons', 4, 32, step=4)
    learning_rate = hp.Float(
        'learning_rate', 1e-4, 1e-2, sampling='log')
    loss = hp.Choice('loss', ['mse', 'mean_squared_logarithmic_error'])

    model = create_model(num_rows_df=HYPER_NUM_ROWS_DF,
                         num_output_fields=HYPER_NUM_OUTPUT_FIELDS,
                         window_size=HYPER_WINDOW_SIZE,
                         look_ahead_size=HYPER_LOOK_AHEAD_SIZE,
                         num_neurons=num_neurons,
                         weight_decay_kernel=weight_decay_kernel,
                         weight_decay_recurrent=weight_decay_recurrent,
                         kernel_dropout=kernel_dropout,
                         recurrent_dropout=recurrent_dropout,
                         second_lstm_layer=False,
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
    norm = tf.keras.layers.LayerNormalization()(input_layer)
    encoder = LSTM(num_neurons,
                   recurrent_regularizer=regularizers.l2(
                       weight_decay_recurrent),
                   kernel_regularizer=regularizers.l2(weight_decay_kernel),
                   recurrent_dropout=recurrent_dropout,
                   dropout=kernel_dropout,
                   return_sequences=second_lstm_layer)(norm)

    if second_lstm_layer:
        encoder = LSTM(num_neurons,
                       recurrent_regularizer=regularizers.l2(
                           weight_decay_recurrent),
                       kernel_regularizer=regularizers.l2(weight_decay_kernel),
                       recurrent_dropout=recurrent_dropout,
                       dropout=kernel_dropout)(encoder)

    repeat = RepeatVector(look_ahead_size)(encoder)
    decoder = LSTM(num_neurons,
                   recurrent_regularizer=regularizers.l2(
                       weight_decay_recurrent),
                   kernel_regularizer=regularizers.l2(weight_decay_kernel),
                   recurrent_dropout=recurrent_dropout,
                   dropout=kernel_dropout,
                   return_sequences=True)(repeat)
    pred = TimeDistributed(
        Dense(num_output_fields, activation='relu'))(decoder)

    model = Model(inputs=input_layer, outputs=pred)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss,
                  metrics=[MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)])
    model.summary()
    return model
