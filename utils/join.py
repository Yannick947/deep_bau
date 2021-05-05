#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:07:37 2021

@author: tquentel
"""

import pandas as pd
from glob import glob

ROOT_PATH = '/home/tquentel/projects/SDaCathon/deep_bau/data/preprocessed/'

filenames = glob(ROOT_PATH + 'join/*.csv')
dfs = [pd.read_csv(f, index_col=0) for f in filenames]

working_hours = pd.read_csv(ROOT_PATH + 'aggregated_by_day.csv', index_col=0)
working_hours.rename(columns={'Projekt Id': 'BaustelleID'}, inplace=True)


first = pd.merge(working_hours, dfs[0].drop_duplicates(
    ["Datum", "BaustelleID"], keep='last'), how="left", on=["Datum", "BaustelleID"], validate="one_to_one")

second = pd.merge(first, dfs[1].drop_duplicates(["Datum", "BaustelleID"], keep='last'), how="left", on=[
                  "Datum", "BaustelleID"], validate="one_to_one")

third = pd.merge(second, dfs[2].drop_duplicates(["Datum", "BaustelleID"], keep='last'), how="left", on=[
                 "Datum", "BaustelleID"], validate="one_to_one")

merged = pd.merge(third, dfs[3].drop_duplicates(["Datum", "BaustelleID"], keep='last'), how="left", on=[
                  "Datum", "BaustelleID"], validate="one_to_one")

# nan temp with mean
merged[['Temperatur_mean', 'Temperatur_min', 'Temperatur_max']] = merged[['Temperatur_mean', 'Temperatur_min',
                                                                          'Temperatur_max']].fillna(merged[['Temperatur_mean', 'Temperatur_min', 'Temperatur_max']].mean())

# fill nan with zero
merged = merged.fillna(0)

merged.to_csv(ROOT_PATH + "df_deep_bau.csv")
