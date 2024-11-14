import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import umap
# import umap.umap_ as umap
import random
random.seed(0)

data_ts = pd.read_csv('stockdata_adj.csv')

# choose adj_close, adj_high, adj_low, adj_open, adj_volume

data_ts1 = data_ts[['adj_close', 'adj_high', 'adj_low', 'adj_open',
                    'adj_volume']]

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(data_ts1)
data_ts = StandardScaler().fit_transform(data_ts1)

# UMAP Dimensionality Reduction

reducer = umap.UMAP()
embedding = reducer.fit_transform(data_ts)

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=np.arange(embedding.shape[0]))
