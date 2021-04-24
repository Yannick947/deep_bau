from typing import Tuple
import pandas as pd
from tensorflow import keras
import numpy as np

from utils.aggregations import ACTIVITIES, HOURS
from machine_learning import PROJECT_ID_COLUMN


class BauGenerator(keras.utils.Sequence):

    def __init__(self, df: pd.DataFrame, window_size: int = 10, batch_size: int = 32) -> None:
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

            x_batch = np.zeros(
                shape=(self.window_size, self.df.shape[1]))
            y_batch = np.zeros(
                shape=(self.df_activities.shape[1]))

            window_index = 0

            # subtract 1 since otherwise there are no labels
            for i in range(df_proj_view.shape[0]):
                x_batch[window_index] = self.df.iloc[i].values

                # TODO: Check if applies
                if window_index < self.window_size - 1:
                    window_index += 1

                else:
                    window_index = 0

                    y_batch = self.df_activities.iloc[i + 1].values

                    X_batches.append(x_batch)
                    Y_batches.append(y_batch)

                    x_batch = np.zeros(
                        shape=(self.window_size, self.df.shape[1]))
                    y_batch = np.zeros(
                        shape=(self.df_activities.shape[1]))

        X_batches = np.stack(X_batches, axis=0)
        Y_batches = np.stack(Y_batches, axis=0)

        return X_batches, Y_batches

    def __getitem__(self, index):

        batches = self.all_batches[index *
                                   self.batch_size: index * self.batch_size + self.batch_size]
        return batches

    def on_epoch_end():
        pass

    def __len__(self):
        return 10


if __name__ == '__main__':
    DEBUG_SIZE = 100

    working_hours = pd.read_csv(
        "./data/preprocessed/aggregated_working_hours.csv", error_bad_lines=False, sep=',')[0:DEBUG_SIZE]
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
