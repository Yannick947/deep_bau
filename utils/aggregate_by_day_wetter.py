#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:06:40 2021

@author: tquentel
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import aggregations as agg

file = "wetter"

df = pd.read_csv(
    f"/home/tquentel/projects/SDaCathon/erfassungsdaten/{file}.csv", index_col=0)

projekt = "BaustelleID"

columns = df.columns
prev = df.head(20)

df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y")
df = df[pd.to_numeric(df[projekt], errors='coerce').notna()]

df["Projekt Id"] = df[projekt].astype(int)

prev = df.head(20)


def one_hot_col(df_obj, col):
    one_hot = OneHotEncoder()
    obj_enc = one_hot.fit_transform(df_obj.values.reshape(-1, 1)).toarray()
    df_obj_enc = pd.DataFrame(
        obj_enc, columns=one_hot.get_feature_names([col]))

    return df_obj_enc


measure = "Wetter"
value = "Temperatur"

cols = ["Datum", projekt, measure]
df = df[cols]

len(df[measure].unique())
prev = df.head(20)

df_enc = one_hot_col(df[measure], measure)
df_enc.index = df.index

df = pd.concat([df, df_enc], axis=1)

prev = df.head(20)

day = df.groupby(["Datum", projekt]).sum()

# all one-hot columns
one_hot_columns = df.filter(regex=f'{measure}_').columns.to_list()

day = day[one_hot_columns]

day = day.reset_index(drop=False)

prev = day.sample(20)

day.to_csv(f"../data/preprocessed/join/aggregated_by_day_{file}.csv")
