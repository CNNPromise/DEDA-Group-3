
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import networkx as nx

# Load data
df = pd.read_csv('indexdata.csv')
# log returns
logret = np.log(df['close']).diff().dropna()
timeline = pd.to_datetime(df['date'])

# Reshape logret to a 2D array
logret_2d = logret.values.reshape(-1, 1)

# Calculate distance matrix
DM = squareform(pdist(logret_2d))

# MST Clustering
KK = 20
Tree1 = nx.minimum_spanning_tree(nx.Graph(DM), weight='weight')
T1 = pd.DataFrame(Tree1.edges(data='weight'),
                  columns=['Source', 'Target', 'Weight'])
T1 = T1.sort_values('Weight')
Kset_MST = np.concatenate([[1], T1['Target'].iloc[-KK + 1:].unique()])
NodesStart = np.concatenate([[1], T1['Target'].iloc[-KK + 1:].unique()])
NodesEnd = np.concatenate([T1['Source'].iloc[-KK + 1:].unique(),
                           [T1['Target'].max()]])

# Color
cmap = plt.get_cmap('hsv')(np.linspace(0, 1, 20))
cmap[[1, 2, 3, 4, 17, 4, 12], :] = cmap[[4, 17, 12, 3, 4, 1, 2], :]


# Plot MST Clustering
plt.subplot(2, 1, 2)
plt.plot(timeline[NodesStart[0]:NodesEnd[0]],
         logret[NodesStart[0]:NodesEnd[0]],
         color=cmap[0])

for i in range(1, len(NodesStart)):
    if i < len(NodesEnd):
        plt.plot(timeline[NodesEnd[i - 1]:NodesEnd[i]],
                 logret[NodesEnd[i - 1]:NodesEnd[i]],
                 color=cmap[i])
# plt.xlim([timeline[0], timeline[-1]])
plt.xlabel('Time')
plt.ylabel('Log Return')
plt.title('MST Clustering')
plt.tight_layout()
plt.show()
