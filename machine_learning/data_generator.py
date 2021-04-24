from typing import Tuple
import pandas as pd
from tensorflow import keras
import numpy as np

from utils.aggregations import ACTIVITIES, HOURS
from machine_learning import PROJECT_ID_COLUMN


class BauGenerator(keras.utils.Sequence):

    def __init__(self, df: pd.DataFrame, window_size: int = 10, batch_size: int = 32, look_ahead_steps: int = 5) -> None:
        """Init datagen

        Args:
            df (pd.DataFrame): Data frame for training, only numeric values allowed !
            window_size (int, optional): [description]. Defaults to 10.
            batch_size (int, optional): [description]. Defaults to 32.
        """
        super().__init__()
        self.window_size = window_size
        self.batch_size = batch_size

        self.df = df
        self.look_ahead_steps = look_ahead_steps

        self.df_activities = self.__get_activities()
        self.X_batches, self.Y_batches = self.__get_batches()

    def __get_activities(self):

        df_activities = pd.DataFrame()
        for col in self.df.columns:
            if ACTIVITIES in col and not 'Id' in col.split():
                df_activities[col] = self.df[col]
        return df_activities

    def __get_batches(self) -> Tuple[np.ndarray, np.ndarray]:

        X_batches = list()
        Y_batches = list()

        for proj_id in self.df[PROJECT_ID_COLUMN].unique():
            df_proj_view = self.df[self.df[PROJECT_ID_COLUMN] == proj_id]
            df_activities_view = self.df_activities[self.df[PROJECT_ID_COLUMN] == proj_id]
            x_batch = np.zeros(
                shape=(self.window_size, self.df.shape[1]))
            y_batch = np.zeros(
                shape=(self.look_ahead_steps, self.df_activities.shape[1]))

            window_index = 0

            # subtract 1 since otherwise there are no labels
            for i in range(df_proj_view.shape[0] - self.look_ahead_steps - self.window_size):
                for ii in range(self.window_size):
                    x_batch[window_index] = df_proj_view.iloc[i + ii].values

                    # TODO: Check if applies
                    if window_index < self.window_size - 1:
                        window_index += 1

                    else:

                        for look_ahead_index in range(self.look_ahead_steps):
                            y_batch[look_ahead_index] = df_activities_view.iloc[i + window_index +
                                                                                look_ahead_index + 1].values

                        X_batches.append(x_batch)
                        Y_batches.append(y_batch)

                        x_batch = np.zeros(
                            shape=(self.window_size, df_proj_view.shape[1]))
                        y_batch = np.zeros(
                            shape=(self.look_ahead_steps, df_activities_view.shape[1]))

                        window_index = 0

        X_batches = np.stack(X_batches, axis=0)
        Y_batches = np.stack(Y_batches, axis=0)

        return X_batches, Y_batches

    def get_single_item_by_index(self, index):
        return self.X_batches[index: index + 1], self.Y_batches[index: index + 1]


if __name__ == '__main__':
    DEBUG_SIZE = 100

    working_hours = pd.read_csv(
        "./data/preprocessed/aggregated_working_hours.csv", index_col=False, error_bad_lines=False, sep=',')[0:DEBUG_SIZE]
    df = working_hours[pd.to_numeric(
        working_hours[PROJECT_ID_COLUMN], errors='coerce').notna()]

    df[PROJECT_ID_COLUMN] = pd.to_numeric(df[PROJECT_ID_COLUMN])
    df = df.astype(np.float64, errors='ignore')
    df = df.select_dtypes([np.number])

    datagen = BauGenerator(df=df, batch_size=32)
    print(datagen.X_batches[0].shape)
    print(len(datagen.X_batches))

    print(datagen.Y_batches[0].shape)
    print(len(datagen.Y_batches))
