from machine_learning import ACTIONS
import pandas as pd
from machine_learning.data_generator import BauGenerator
from machine_learning.train import BATCH_SIZE, LOOK_AHEAD_SIZE, LOOK_BACK_WINDOW_SIZE
from machine_learning.utils import get_datagen_split
import os

from tensorflow import keras
import numpy as np


def load_model_from_file(path: str) -> keras.models.Model:

    model = keras.models.load_model(path, compile=False)
    return model


def predict_future_vals(model: keras.models.Model, window_frame: np.array):

    future_window = model.predict(window_frame)
    future_window_rounded = np.rint(future_window)
    future_window_rounded = np.squeeze(future_window_rounded)
    return future_window_rounded


def reassign_col_names_actions(arr: np.ndarray, df_orig: pd.DataFrame) -> pd.DataFrame:

    cols = [col for col in df_orig.columns if ACTIONS in col]

    reassigned_df = pd.DataFrame(arr, columns=cols)
    return reassigned_df


def get_taetigkeiten_from_arr(arr: np.ndarray, df_orig: pd.DataFrame) -> np.ndarray:

    col_actions_mask = [
        True if ACTIONS in col else False for col in df_orig.columns]
    masked_arr = np.transpose(arr[0, :, col_actions_mask])
    return masked_arr


def reassign_cols(arr: np.ndarray, df_orig: pd.DataFrame) -> pd.DataFrame:

    cols = [col for col in df_orig.columns]
    reassigned_df = pd.DataFrame(arr[0, :, :], columns=cols)
    return reassigned_df


if __name__ == '__main__':

    file_path = "C:/Users/Yannick/Documents/Python/deep_bau/machine_learning/logs/24222831/lstm.h5"
    model = load_model_from_file(file_path)

    DEBUG_SIZE = 100

    working_hours = pd.read_csv(
        "./data/preprocessed/df_deep_bau.csv", error_bad_lines=False, sep=',')  # [0:DEBUG_SIZE]
    working_hours = working_hours.select_dtypes([np.number])

    df_train, df_val = get_datagen_split(working_hours)

    datagen_val = BauGenerator(df=df_val, batch_size=1,
                               window_size=LOOK_BACK_WINDOW_SIZE,
                               look_ahead_steps=LOOK_AHEAD_SIZE)

    for i in range(1000):

        window_x, window_y = datagen_val.get_single_item_by_index(i)
        window_y_hat = predict_future_vals(model, window_x)

        diff = np.sum(np.absolute(window_y_hat - window_y))
        mean = np.sum(window_y)
        print('index: ', i, ' prediction diff: ', diff, ', sum ', mean)

        if diff < 45:

            window_x_actions = get_taetigkeiten_from_arr(
                window_x, working_hours)
            window_pred_merge = np.concatenate(
                [window_x_actions, window_y_hat])

            df_combined = reassign_col_names_actions(
                window_pred_merge, working_hours)
            df_combined.to_csv(f'./df_pred_included_index{i}_diff{diff}.csv')

            df_window_x = reassign_cols(window_x, working_hours)
            df_window_x.to_csv(f'./df_input_window_index{i}_diff{diff}.csv')
