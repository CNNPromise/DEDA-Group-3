import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

# Chart drawing
import plotly.io as pio
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode

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

# %%
df_factors = pd.read_csv("stockdata_factors.csv")

df_factors.loc[:, 'label'] = df_factors.loc[:, "adj_close"
                                            ].pct_change().shift(-1)
print(df_factors.shape)

df_factors = df_factors.iloc[30:]
df_factors = df_factors[:-1]

df_factors.index = range(len(df_factors))
df_factors.dropna(inplace=True)
X = np.array(df_factors.drop(columns=['label']))
X = np.array(df_factors.drop(columns=['trade_date']))
X = preprocessing.scale(X)
y = np.array(df_factors['label'])

train_test_split_idx = int(len(X) * 0.8)
X_train = X[:train_test_split_idx]
X_test = X[train_test_split_idx:]
y_train = y[:train_test_split_idx]
y_test = y[train_test_split_idx:]

parameters = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'max_depth': [8, 10, 12, 15],
    'gamma': [0.001, 0.005, 0.01, 0.02],
    'random_state': [42]
}


# %%
model = xgb.XGBRegressor(objective='reg:squarederror')
clf = GridSearchCV(model, parameters)

clf.fit(X_train, y_train)

print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')

model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

y_pred = model.predict(X_test)
print(f'y_true = {np.array(y_test)[:5]}')
print(f'y_pred = {y_pred[:5]}')
print(f'mean_absolute_error = {mean_absolute_error(y_test, y_pred)}')

# %%
df_factors.loc[:, "predict"] = np.nan
df_factors.loc[:, "predict"][:train_test_split_idx] = clf.predict(X_train)
df_factors.loc[:, "predict"][train_test_split_idx:] = clf.predict(X_test)
df_factors.loc[:, "predict_close"] = df_factors["adj_close"
                                                ] * (1 + df_factors["predict"])
df_factors.loc[:, "predict_close"] = df_factors["predict_close"].shift(-1)

df_factors[["adj_close", "predict_close"]].plot()
