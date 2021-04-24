from datetime import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.losses import MeanAbsoluteError

from machine_learning.utils import get_datagen_split
from machine_learning.create_model import create_model, create_hyperband_model
from machine_learning import PROJECT_ID_COLUMN, ROOT_PATH
from machine_learning.data_generator import BauGenerator
from visualizations.plot_utils import plot_history
import kerastuner as kt


def hyperband_optimization():
    tuner = kt.Hyperband(create_hyperband_model,
                         objective='val_accuracy',
                         max_epochs=80,
                         hyperband_iterations=2)

    tuner.search(tg,
                 validation_data=vg,
                 epochs=50,
                 callbacks=[],
                 workers=16)
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print(best_hyperparameters)


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

    return callbacks


def train(df: pd.DataFrame):
    df_train, df_val = get_datagen_split(df)

    datagen_train = BauGenerator(df=df_train, batch_size=BATCH_SIZE)
    datagen_val = BauGenerator(df=df_val, batch_size=BATCH_SIZE)

    for neurons in [8, 15, 30, 60]:
        model = create_model(num_rows_df=datagen_train.X_batches.shape[2],
                             num_output_fields=datagen_train.Y_batches.shape[1],
                             num_neurons=neurons)

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
                            callbacks=callbacks)

        plot_history(history=history, logging_path=logging_path)


if __name__ == '__main__':

    DEBUG_SIZE = 100
    BATCH_SIZE = 32

    working_hours = pd.read_csv(
        "./data/preprocessed/aggregated_by_day.csv", error_bad_lines=False, sep=',')  # [0:DEBUG_SIZE]
    df = working_hours[pd.to_numeric(
        working_hours[PROJECT_ID_COLUMN], errors='coerce').notna()]

    df[PROJECT_ID_COLUMN] = pd.to_numeric(df[PROJECT_ID_COLUMN])
    df = df.astype(np.float64, errors='ignore')
    df = df.select_dtypes([np.number])
    train(df)
