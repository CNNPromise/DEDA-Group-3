# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib import style
style.use('ggplot')

df = pd.read_csv('stockdata_factors.csv')
df.loc[:, 'label'] = df.loc[:, "adj_close"].pct_change().shift(-1)
dflabel = df['label'].copy()
df.set_index('trade_date', inplace=True)

scaler = StandardScaler()
# Standardize each column except the 'label' column in the DataFrame
df_standardized = pd.DataFrame(scaler.fit_transform(df),
                               columns=df.columns, index=df.index)
# df_standardized = df_standardized.drop(columns=['label'])

df_standardized['label'] = dflabel

df_standardized.dropna(inplace=True)

df_standardized.head()


# %%
# Setting up factors and labels, X->factor and y->label

X = np.array(df_standardized.drop(columns=['label']))
y = np.array(df_standardized['label'])

# Train and test data set
# Test size=0.2 means we are using 20% data as a testing data

train_test_split_idx = int(len(X) * 0.8)
X_train = X[:train_test_split_idx]
X_test = X[train_test_split_idx:]
y_train = y[:train_test_split_idx]
y_test = y[train_test_split_idx:]
print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

# Use Linear Regression
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy of Linear Regression: ", accuracy)
print("Coefficient of Linear Regression: ", clf.coef_)

df_standardized.loc[:, "predict"] = np.nan
df_standardized.loc[:, "predict"][:train_test_split_idx] = clf.predict(X_train)
df_standardized.loc[:, "predict"][train_test_split_idx:] = clf.predict(X_test)

df_standardized.loc[:, "predict_close"] = df_standardized[
    "adj_close"] * (1 + df_standardized["predict"])
df_standardized.loc[:, "predict_close"] = df_standardized[
    "predict_close"].shift(-1)

df_standardized[["adj_close", "predict_close"]].plot()
plt.title("Linear Regression")
plt.show()

# %%
# Use Support Vector Machine

svr_lin = svm.SVR(kernel='linear', C=1e3).fit(X_train, y_train)
svr_poly = svm.SVR(kernel='poly', C=1e3, degree=2).fit(X_train, y_train)
svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1).fit(X_train, y_train)
accuracy_lin = svr_lin.score(X_test, y_test)
accuracy_poly = svr_poly.score(X_test, y_test)
accuracy_rbf = svr_rbf.score(X_test, y_test)
print(accuracy_lin, accuracy_poly, accuracy_rbf)

df_s = df_standardized.copy()
df_s.loc[:, "predict_svrl"] = np.nan
df_s.loc[:, "predict_svrl"][:train_test_split_idx] = svr_lin.predict(X_train)
df_s.loc[:, "predict_svrl"][train_test_split_idx:] = svr_lin.predict(X_test)
df_s.loc[:, "predict_close_svrl"] = df_s["adj_close"] \
    * (1 + df_s["predict_svrl"])
df_s.loc[:, "predict_close_svrl"] = df_s["predict_close_svrl"].shift(-1)

df_s.loc[:, "predict_svrp"] = np.nan
df_s.loc[:, "predict_svrp"][:train_test_split_idx] = svr_poly.predict(X_train)
df_s.loc[:, "predict_svrp"][train_test_split_idx:] = svr_poly.predict(X_test)
df_s.loc[:, "predict_close_svrp"] = df_s["adj_close"] \
    * (1 + df_s["predict_svrp"])
df_s.loc[:, "predict_close_svrp"] = df_s["predict_close_svrp"].shift(-1)

df_s.loc[:, "predict_svrr"] = np.nan
df_s.loc[:, "predict_svrr"][:train_test_split_idx] = svr_rbf.predict(X_train)
df_s.loc[:, "predict_svrr"][train_test_split_idx:] = svr_rbf.predict(X_test)
df_s.loc[:, "predict_close_svrr"] = df_s["adj_close"] \
    * (1 + df_s["predict_svrr"])
df_s.loc[:, "predict_close_svrr"] = df_s["predict_close_svrr"].shift(-1)

df_s[["adj_close", "predict_close_svrl"]].plot()
plt.title("Support Vector Machine (kernel=linear)")
plt.show()
