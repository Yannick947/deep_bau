#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:07:37 2021

@author: tquentel
"""

#%%
import pandas as pd
from glob import glob

#%%
ROOT_PATH = '/home/tquentel/projects/SDaCathon/deep_bau/data/preprocessed/'

filenames = glob(ROOT_PATH +'join/*.csv')
dfs = [pd.read_csv(f, index_col=0) for f in filenames]

working_hours = pd.read_csv(ROOT_PATH + 'aggregated_by_day.csv', index_col=0)

#%%
nan_value = 0

dfs.insert(0, working_hours)

#%%
# solution 1 (fast)
result = pd.concat(dfs, join='left', axis=1).fillna(nan_value)


prev = result_1.sample(20)

first = dfs[0].merge(dfs[1], how="left", right_on=["Datum", "BaustelleID"], left_on=["Datum", "Projekt Id"])

first = pd.merge(dfs[0], dfs[1], how="left", left_on=["Datum", "Projekt Id"], right_on=["Datum", "BaustelleID"])


