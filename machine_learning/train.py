import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (CSVLogger, LearningRateScheduler,
                                        ModelCheckpoint)

from machine_learning import ROOT_PATH
from machine_learning.data_generator import BauGenerator
from machine_learning.classfication_models import create_dummy_classifier
from machine_learning.models import create_model
from machine_learning.utils import get_datagen_split
from visualizations.plot_utils import plot_history

BATCH_SIZE = 128
LOOK_AHEAD_SIZE = 1
LOOK_BACK_WINDOW_SIZE = 3


def schedule(epoch, lr):
    if epoch < 50:
        return lr

    else:
        return lr * tf.math.exp(-0.05)


def get_callbacks(logging_path: str):
    callbacks = []

    callbacks.append(CSVLogger(filename=os.path.join(
        logging_path, 'training_curve.csv')))

    callbacks.append(ModelCheckpoint(
        filepath=os.path.join(logging_path, 'lstm.h5'), verbose=1, mode='min'))

    callbacks.append(LearningRateScheduler(schedule=schedule, verbose=1))
    return callbacks


def train_dummy_classifier(df: pd.DataFrame):
    df_train, df_val = get_datagen_split(df)

    datagen_train = BauGenerator(df=df_train, batch_size=BATCH_SIZE,
                                 window_size=LOOK_BACK_WINDOW_SIZE,
                                 look_ahead_steps=LOOK_AHEAD_SIZE)

    datagen_val = BauGenerator(df=df_val, batch_size=BATCH_SIZE,
                               window_size=LOOK_BACK_WINDOW_SIZE,
                               look_ahead_steps=LOOK_AHEAD_SIZE)

    model = create_dummy_classifier(window_size=LOOK_BACK_WINDOW_SIZE,
                                    num_rows_df=datagen_train.X_batches.shape[2],
                                    num_output_fields=datagen_train.Y_batches.shape[1])

    logging_path = os.path.join(ROOT_PATH, 'logs', str(
        datetime.now().strftime("%d%H%M%S")))

    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    callbacks = get_callbacks(logging_path)

    history = model.fit(x=datagen_train,
                        validation_data=datagen_val,
                        epochs=60,
                        callbacks=callbacks,
                        shuffle=True)


def train(df: pd.DataFrame):
    df_train, df_val = get_datagen_split(df)

    datagen_train = BauGenerator(df=df_train, batch_size=BATCH_SIZE,
                                 window_size=LOOK_BACK_WINDOW_SIZE,
                                 look_ahead_steps=LOOK_AHEAD_SIZE)

    datagen_val = BauGenerator(df=df_val, batch_size=BATCH_SIZE,
                               window_size=LOOK_BACK_WINDOW_SIZE,
                               look_ahead_steps=LOOK_AHEAD_SIZE)

    model = create_model(num_rows_df=datagen_train.X_batches.shape[2],
                         num_output_fields=datagen_train.Y_batches.shape[2],
                         look_ahead_size=LOOK_AHEAD_SIZE,
                         window_size=LOOK_BACK_WINDOW_SIZE,
                         weight_decay_recurrent=0,
                         weight_decay_kernel=0,
                         recurrent_dropout=0,
                         kernel_dropout=0,
                         num_neurons=16,
                         learning_rate=0.01,
                         loss='categorical_crossentropy')

    logging_path = os.path.join(ROOT_PATH, 'logs', str(
        datetime.now().strftime("%d%H%M%S")))

    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    callbacks = get_callbacks(logging_path)

    history = model.fit(x=datagen_train.X_batches,
                        y=datagen_train.Y_batches,
                        validation_data=(datagen_val.X_batches,
                                         datagen_val.Y_batches),
                        epochs=100,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks,
                        shuffle=True)

    plot_history(history=history, logging_path=logging_path, show=False)


if __name__ == '__main__':

    DEBUG_SIZE = 0.5

    working_hours = pd.read_csv(
        "./data/preprocessed/df_deep_bau.csv", error_bad_lines=False, sep=',')

    start_index = int(DEBUG_SIZE * working_hours.shape[0])
    working_hours = working_hours.select_dtypes([np.number])[start_index:]

    train_dummy_classifier(working_hours)
    # train(working_hours)
