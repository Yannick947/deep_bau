from datetime import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint, TensorBoard

from machine_learning.utils import get_datagen_split
from machine_learning.models import create_model
from machine_learning import PROJECT_ID_COLUMN, ROOT_PATH
from machine_learning.data_generator import BauGenerator
from visualizations.plot_utils import plot_history

BATCH_SIZE = 32
LOOK_AHEAD_SIZE = 5
LOOK_BACK_WINDOW_SIZE = 10


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
        filepath=os.path.join(logging_path, 'lstm.h5')))

    callbacks.append(LearningRateScheduler(schedule=schedule, verbose=1))
    callbacks.append(TensorBoard(log_dir=logging_path))
    return callbacks


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
                         weight_decay_recurrent=0.0,
                         weight_decay_kernel=0.0,
                         recurrent_dropout=0.0,
                         kernel_dropout=0.0,
                         num_neurons=64)

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

    DEBUG_SIZE = 100

    working_hours = pd.read_csv(
        "./data/preprocessed/df_deep_bau.csv", error_bad_lines=False, sep=',')  # [0:DEBUG_SIZE]

    print(working_hours.shape)

    # df = working_hours[pd.to_numeric(
    #    working_hours[PROJECT_ID_COLUMN], errors='coerce').notna()]

    #df[PROJECT_ID_COLUMN] = pd.to_numeric(df[PROJECT_ID_COLUMN])
    #df = df.astype(np.float64, errors='ignore')
    working_hours = working_hours.select_dtypes([np.number])

    print(working_hours.shape)

    train(working_hours)
