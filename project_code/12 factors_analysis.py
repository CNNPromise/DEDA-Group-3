# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chart drawing
import plotly.io as pio
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
import seaborn as sns
from sklearn import preprocessing

# Mute sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

# Show charts when running kernel
init_notebook_mode(connected=True)

# Change default background color for all visualizations
layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(250,250,250,0.8)')
fig = go.Figure(layout=layout)
templated_fig = pio.to_templated(fig)
pio.templates['my_template'] = templated_fig.layout.template
pio.templates.default = 'my_template'


# Load data
df = pd.read_csv("stockdata_adj.csv")

# Choose Guizhou Moutai (600519) as the target stock
df = df[df['ts_code'] == '600519.SH']
df = df.sort_values('trade_date')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df.index = range(len(df))

df_factors = pd.read_csv("stockdata_factors.csv")

# Shift label column 'adj_close' by 1 to predict the next day's price
df_factors.loc[:, 'label'] = df_factors.loc[:,
                                            "adj_close"].pct_change().shift(-1)
print(df_factors.shape)

df_factors = df_factors.iloc[30:]
df_factors = df_factors[:-1]

df_factors.index = range(len(df_factors))

# Covariance matrix of factors
corr = df_factors.corr()

plt.figure(figsize=(40, 32))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
plt.savefig("CorMtx_whole.png")

# %%

df_factors = pd.read_csv("stockdata_factors.csv")
df_factors.dropna(inplace=True)
df_factors.loc[:, 'label'] = df_factors.loc[:,
                                            "adj_close"].pct_change().shift(-1)
X = np.array(df_factors.drop(columns=['label']))
X = np.array(df_factors.drop(columns=['trade_date']))
X = preprocessing.scale(X)
y = np.array(df_factors['label'])

train_test_split_idx = int(len(X) * 0.8)

train_df = df_factors.loc[:train_test_split_idx].copy()
test_df = df_factors.loc[train_test_split_idx:].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df.trade_date, y=train_df.adj_close,
                         name='Train'))
fig.add_trace(go.Scatter(x=test_df.trade_date, y=test_df.adj_close,
                         name='Test'))
fig.show()

# save the plot
fig.write_image("train_test_split.png")
