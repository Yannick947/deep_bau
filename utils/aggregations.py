import os

import pandas as pd

ACTIVITIES = 'Tätigkeit'
HOURS = 'Stunden'


def one_hot_taetigkeiten(df: pd.DataFrame, prefix: str = ACTIVITIES) -> pd.DataFrame:

    one_hot = pd.get_dummies(df[ACTIVITIES], prefix=prefix, drop_first=True)
    df = df.join(one_hot)

    if ACTIVITIES in df.columns:
        df.drop(labels=[ACTIVITIES], inplace=True, axis=1)

    return df


def replace_taetigkeiten_working_hours(df: pd.DataFrame, prefix: str = ACTIVITIES) -> pd.DataFrame:

    for col in df.columns:
        if ACTIVITIES in col and not 'Id' in col.split():
            df = df.apply(replace_taetigkeit, col_name=col, axis=1)
    return df


def replace_taetigkeit(row, col_name: str):
    if row[col_name]:
        row[col_name] = row[HOURS]
    return row


if __name__ == '__main__':
    DEBUG_SIZE = 100

    working_hours = pd.read_csv(
        "./data/working_hours.csv", error_bad_lines=False, sep=';')  # [0:DEBUG_SIZE]
    df = one_hot_taetigkeiten(working_hours)
    df = replace_taetigkeiten_working_hours(df)
    df.to_csv('./aggregated_working_hours.csv', index=False)
    df.to_excel('./aggregated_working_hours.xlsx', index=False)

    print(df.head())
