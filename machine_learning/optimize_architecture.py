from datetime import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint, TensorBoard

from machine_learning.utils import get_datagen_split
import machine_learning.models
from machine_learning.models import create_hyperband_model
from machine_learning import PROJECT_ID_COLUMN
from machine_learning.data_generator import BauGenerator
import kerastuner as kt

BATCH_SIZE = 32
LOOK_AHEAD_SIZE = 5
LOOK_BACK_WINDOW_SIZE = 10


def hyperband_optimization(df: pd.DataFrame):
    df_train, df_val = get_datagen_split(df)

    datagen_train = BauGenerator(df=df_train, batch_size=BATCH_SIZE,
                                 window_size=LOOK_BACK_WINDOW_SIZE,
                                 look_ahead_steps=LOOK_AHEAD_SIZE)

    datagen_val = BauGenerator(df=df_val, batch_size=BATCH_SIZE,
                               window_size=LOOK_BACK_WINDOW_SIZE,
                               look_ahead_steps=LOOK_AHEAD_SIZE)

    machine_learning.models.HYPER_NUM_ROWS_DF = datagen_train.X_batches.shape[2]
    machine_learning.models.HYPER_NUM_OUTPUT_FIELDS = datagen_train.Y_batches.shape[2]
    machine_learning.models.HYPER_WINDOW_SIZE = LOOK_BACK_WINDOW_SIZE
    machine_learning.models.HYPER_LOOK_AHEAD_SIZE = LOOK_AHEAD_SIZE

    tuner = kt.BayesianOptimization(create_hyperband_model,
                                    objective='val_loss',
                                    max_trials=10)

    tuner.search(datagen_train.X_batches,
                 datagen_train.Y_batches,
                 validation_data=(datagen_val.X_batches,
                                  datagen_val.Y_batches),
                 epochs=60,
                 callbacks=[],
                 workers=16)
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print(best_hyperparameters)


if __name__ == '__main__':

    DEBUG_SIZE = 100

    working_hours = pd.read_csv(
        "./data/preprocessed/df_deep_bau.csv", error_bad_lines=False, sep=',', index_col=False)

    START_INDEX = int(0.3 * working_hours.shape[0])

    working_hours = working_hours[START_INDEX:]

    df = working_hours.astype(np.float64, errors='ignore')
    df = df.select_dtypes([np.number])

    hyperband_optimization(df)
