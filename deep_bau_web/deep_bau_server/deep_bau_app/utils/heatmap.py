#%%
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#%%
df = pd.read_csv('/home/tquentel/projects/SDaCathon/deep_bau/data/preprocessed/aggregated_by_day.csv', index_col=0)

#%%

df = df[df["Projekt Id"] == 101227]


#%%
df = df.loc[:, (df != 0).any(axis=0)]

df.set_index(df["Datum"], inplace=True)
df = df.filter(regex='Tätigkeit_')
df.columns = df.columns.str.replace(r'Tätigkeit_', '')


#%%


df = df.transpose()
df.replace(0, np.nan, inplace=True)


#%%
fig, ax = plt.subplots(1, 1)
sns.set_style(style='white')
sns.set(rc={'figure.figsize':(15,15)})

cmap=sns.cm.rocket_r
g1 = sns.heatmap(df, cmap=cmap, square=True, annot=True, linewidths=0, cbar=0)
# g1.set(xticklabels=[])


for i in range(df.shape[1]+1):
    ax.axhline(i, color='white', lw=2)
    
xticks = ax.get_xticks()
    
fig.autofmt_xdate()
    
fig.set

plt.show()