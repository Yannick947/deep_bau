#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:06:40 2021

@author: tquentel
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(
    "/home/tquentel/projects/SDaCathon/deep_bau/data/preprocessed/aggregated_working_hours.csv")

df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y")
df = df[pd.to_numeric(df["Projekt Id"], errors='coerce').notna()]

df["Projekt Id"] = df["Projekt Id"].astype(int)
df.reset_index()

prev = df.head(20)


def one_hot_col(df_obj, col):
    one_hot = OneHotEncoder()
    obj_enc = one_hot.fit_transform(df_obj.values.reshape(-1, 1)).toarray()
    df_obj_enc = pd.DataFrame(
        obj_enc, columns=one_hot.get_feature_names([col]))

    return df_obj_enc


df_enc = one_hot_col(df["Baubereich Id"], "Baubereich")
df_enc.index = df.index

df_enc_per = one_hot_col(df["Person Id"], "Person")
df_enc_per.index = df.index

df = pd.concat([df, df_enc, df_enc_per], axis=1)

prev = df.head(20)

test = df.groupby(["Datum", "Projekt Id"]).sum()

# all one-hot columns
one_hot_columns = df.filter(
    regex='Tätigkeit_|Baubereich_|Person_').columns.to_list()

test = test[one_hot_columns]

test = test.reset_index(drop=False)

test.to_csv("../data/preprocessed/aggregated_by_day.csv")
