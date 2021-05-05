from tensorflow.python.keras.layers.core import Dropout, Flatten
from tensorflow.python.keras.optimizer_v2.adam import Adam
from machine_learning.models import create_model
from tensorflow.keras.layers import Input, Dense, Bidirectional
from tensorflow import keras
from tensorflow.python.keras.layers.recurrent import GRU, LSTM

HYPER_NUM_ROWS_DF = None
HYPER_NUM_OUTPUT_FIELDS = None
HYPER_WINDOW_SIZE = 10
HYPER_LOOK_AHEAD_SIZE = 1


def create_bayesian_dummy_classifier(hp):

    dropout = hp.Float('dropout', 0.0, 0.3, sampling='linear')
    neurons_rnn = hp.Int('neurons_rnn', 8, 80, step=8)
    learning_rate = hp.Float('learning_rate', 1e-4, 0.1, sampling='linear')

    #bidirection = hp.Choice('bidirect', [False])

    model = create_dummy_classifier(num_rows_df=HYPER_NUM_ROWS_DF,
                                    num_output_fields=HYPER_NUM_OUTPUT_FIELDS,
                                    window_size=HYPER_WINDOW_SIZE,
                                    neurons_rnn=neurons_rnn,
                                    dropout=dropout,
                                    learning_rate=learning_rate)
    return model


def create_bayesian_classifier(hp):

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
                         learning_rate=learning_rate)
    return model


def create_dummy_classifier(window_size: int,
                            num_rows_df: int,
                            num_output_fields: int,
                            neurons_rnn: int = 10,
                            dropout: float = 0.0,
                            learning_rate: float = 0.01,
                            bidirection: bool = True,
                            return_sequences: bool = False):
                            
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000,
        decay_rate=0.9)

    model = keras.Sequential(name='dummy_classifier')

    model.add(Input(shape=(window_size, num_rows_df), name='input'))

    if bidirection:
        model.add(Bidirectional(
            LSTM(neurons_rnn, return_sequences=return_sequences),
            name='bidirection'))
    else:
        model.add(LSTM(neurons_rnn, name="rnn",
                       return_sequences=return_sequences))
    if return_sequences:
        model.add(Flatten())
    model.add(Dropout(dropout, name='dropout'))
    model.add(Dense(num_output_fields, activation='sigmoid', name='dense_output'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy', 'binary_accuracy'])
    return model
