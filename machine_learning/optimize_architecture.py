import datetime

from machine_learning.classfication_models import create_bayesian_classifier, create_bayesian_dummy_classifier
import numpy as np
import pandas as pd
import kerastuner as kt
from machine_learning.utils import get_datagen_split
import machine_learning.models
import machine_learning.classfication_models
from machine_learning.models import create_hyperband_model
from machine_learning.data_generator import BauGenerator

BATCH_SIZE = 256
LOOK_AHEAD_SIZE = 1
LOOK_BACK_WINDOW_SIZE = 10


def dummy_classification(df: pd.DataFrame):
    df_train, df_val = get_datagen_split(df)

    datagen_train = BauGenerator(df=df_train, batch_size=BATCH_SIZE,
                                 window_size=LOOK_BACK_WINDOW_SIZE,
                                 look_ahead_steps=LOOK_AHEAD_SIZE)

    datagen_val = BauGenerator(df=df_val, batch_size=BATCH_SIZE,
                               window_size=LOOK_BACK_WINDOW_SIZE,
                               look_ahead_steps=LOOK_AHEAD_SIZE)

    machine_learning.classfication_models.HYPER_NUM_ROWS_DF = datagen_train.X_batches.shape[2]
    machine_learning.classfication_models.HYPER_NUM_OUTPUT_FIELDS = datagen_train.Y_batches.shape[
        1]
    machine_learning.classfication_models.HYPER_WINDOW_SIZE = LOOK_BACK_WINDOW_SIZE
    machine_learning.classfication_models.HYPER_LOOK_AHEAD_SIZE = LOOK_AHEAD_SIZE

    tuner = kt.BayesianOptimization(create_bayesian_dummy_classifier,
                                    objective='val_accuracy',
                                    max_trials=100,
                                    project_name="arch_opt_")

    tuner.search(datagen_train,
                 validation_data=datagen_val,
                 epochs=60,
                 callbacks=[],
                 workers=16)
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print(best_hyperparameters)


def bayesian_classification_optimization(df: pd.DataFrame):
    df_train, df_val = get_datagen_split(df)

    datagen_train = BauGenerator(df=df_train, batch_size=BATCH_SIZE,
                                 window_size=LOOK_BACK_WINDOW_SIZE,
                                 look_ahead_steps=LOOK_AHEAD_SIZE)

    datagen_val = BauGenerator(df=df_val, batch_size=BATCH_SIZE,
                               window_size=LOOK_BACK_WINDOW_SIZE,
                               look_ahead_steps=LOOK_AHEAD_SIZE)

    machine_learning.classfication_models.HYPER_NUM_ROWS_DF = datagen_train.X_batches.shape[2]
    machine_learning.classfication_models.HYPER_NUM_OUTPUT_FIELDS = datagen_train.Y_batches.shape[
        1]
    machine_learning.classfication_models.HYPER_WINDOW_SIZE = LOOK_BACK_WINDOW_SIZE
    machine_learning.classfication_models.HYPER_LOOK_AHEAD_SIZE = LOOK_AHEAD_SIZE

    tuner = kt.BayesianOptimization(create_bayesian_classifier,
                                    objective='val_loss',
                                    max_trials=200)

    tuner.search(datagen_train,
                 validation_data=datagen_val,
                 epochs=150,
                 callbacks=[],
                 workers=16)
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print(best_hyperparameters)


def hyperband_optimization(df: pd.DataFrame):
    df_train, df_val = get_datagen_split(df)

    datagen_train = BauGenerator(df=df_train, binarize_activity_hours=False, batch_size=BATCH_SIZE,
                                 window_size=LOOK_BACK_WINDOW_SIZE,
                                 look_ahead_steps=LOOK_AHEAD_SIZE)

    datagen_val = BauGenerator(df=df_val, binarize_activity_hours=False, batch_size=BATCH_SIZE,
                               window_size=LOOK_BACK_WINDOW_SIZE,
                               look_ahead_steps=LOOK_AHEAD_SIZE)

    machine_learning.models.HYPER_NUM_ROWS_DF = datagen_train.X_batches.shape[2]
    machine_learning.models.HYPER_NUM_OUTPUT_FIELDS = datagen_train.Y_batches.shape[2]
    machine_learning.models.HYPER_WINDOW_SIZE = LOOK_BACK_WINDOW_SIZE
    machine_learning.models.HYPER_LOOK_AHEAD_SIZE = LOOK_AHEAD_SIZE

    tuner = kt.BayesianOptimization(create_hyperband_model,
                                    objective='val_binary_accuracy',
                                    max_trials=200)

    tuner.search(datagen_train,
                 validation_data=datagen_val,
                 epochs=70,
                 callbacks=[],
                 workers=16)
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print(best_hyperparameters)


if __name__ == '__main__':

    PERCENTAGE_USED_DATA = 0.7

    working_hours = pd.read_csv(
        "./data/preprocessed/df_deep_bau.csv", error_bad_lines=False, sep=',', index_col=False)

    start_index = int((1 - PERCENTAGE_USED_DATA) * working_hours.shape[0])

    working_hours = working_hours[start_index:]

    df = working_hours.select_dtypes([np.number])

    dummy_classification(df)
