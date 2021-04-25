from typing import Tuple
import pandas as pd

from machine_learning import PROJECT_ID_COLUMN
from sklearn.model_selection import train_test_split


def get_datagen_split(df: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    project_ids = df[PROJECT_ID_COLUMN].unique()

    project_ids_train, project_ids_test = train_test_split(
        project_ids, train_size=train_size, random_state=42)

    df_train = df[df[PROJECT_ID_COLUMN].isin(project_ids_train)]
    df_test = df[df[PROJECT_ID_COLUMN].isin(project_ids_test)]

    return df_train, df_test
