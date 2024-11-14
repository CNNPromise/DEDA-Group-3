# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx

df = pd.read_csv("stockdata_adj.csv")

plt.figure(figsize=(12, 8))

stocklist = ["600519.SH", "000568.SZ", "000858.SZ", "600559.SH",
             "600809.SH", "002304.SZ", "000596.SZ", "603369.SH", "603198.SH",
             "600702.SH", "600779.SH"]

for stock in stocklist:
    stock_data = df[df.ts_code == stock][::-1].iloc[10:]
    stock_data.set_index('trade_date', inplace=True)
    stock_data['adj_close'].plot(label=stock)

plt.legend(loc='upper left')
plt.title('Close Price of Baijiu Stocks')
plt.show()

plt.figure(figsize=(12, 8))

for stock in stocklist:
    stock_data = df[df.ts_code == stock][::-1].iloc[10:]
    stock_data.set_index('trade_date', inplace=True)
    stock_data['pct_chg'].plot(label=stock)

plt.legend(loc='upper left')
plt.title('Percentage Change of Baijiu Stocks')
plt.show()

# %%
# choose close price of the stocks, and drop other columns.

df = pd.read_csv("stockdata_adj.csv")
df.set_index('trade_date', inplace=True)
stocklist = ["600519.SH", "000568.SZ", "000858.SZ", "600559.SH",
             "600809.SH", "002304.SZ", "000596.SZ", "603369.SH",
             "603198.SH", "600702.SH", "600779.SH"]
df = df[df.ts_code.isin(stocklist)]

prices = df[['ts_code', 'adj_close']]
prices = prices.pivot(columns='ts_code', values='adj_close')
prices = prices.dropna()

# Calculate log returns
returns = np.log(prices / prices.shift(1)).dropna()

# Calculate correlation matrix
correlation_matrix = returns.corr()

# Plot heatmap of correlation matrix
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
# plt.savefig("CorMtx_whole.png")

# Calculate eigenvalues and eigenvectors of correlation matrix
eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
lambda_max = (1 + 1 / len(returns.columns)
              + 2 * np.sqrt(1 / len(returns.columns)))

# Filter eigenvalues and eigenvectors
f_eigenvalues = np.where(eigenvalues < lambda_max, 0, eigenvalues)
f_diagonal = np.diag(f_eigenvalues)
f_eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]

# Calculate filtered correlation matrix
filtered_correlation_matrix = f_eigenvectors @ f_diagonal @ f_eigenvectors.T
np.fill_diagonal(filtered_correlation_matrix, 1)

# Plot heatmap of filtered correlation matrix
plt.figure(figsize=(20, 16))
sns.heatmap(filtered_correlation_matrix, cmap='coolwarm', annot=True)
# plt.savefig("FiltCorMtx_whole.png")

# Calculate distance matrix
distance_matrix = np.sqrt(2 - 2 * filtered_correlation_matrix)
np.fill_diagonal(distance_matrix, 0)

# Plot heatmap of distance matrix
plt.figure(figsize=(20, 16))
sns.heatmap(distance_matrix, cmap='coolwarm', annot=True, fmt='.4f')
plt.savefig("DistMtx_whole.png")

# ——————————————————————————————————————————————————————————————————
# ——————————————————————————————————————————————————————————————————

# Create a graph
G = nx.Graph()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

nodes = ["泸州老窖", "古井贡酒", "五粮液", "贵州茅台", "老白干",
         "舍得酒业", "水井坊", "山西汾酒", "迎驾贡酒", "今世缘"]
G.add_nodes_from(nodes)

edges = [("泸州老窖", "贵州茅台", {'weight': 1/0.4073}),
         ("泸州老窖", "山西汾酒", {'weight': 1/1.2048}),
         ("五粮液", "迎驾贡酒",   {'weight': 1/1.2823}),
         ("五粮液", "水井坊",     {'weight': 1/1.2841}),
         ("泸州老窖", "今世缘",   {'weight': 1/1.3433}),
         ("泸州老窖", "老白干",   {'weight': 1/1.3645}),
         ("泸州老窖", "舍得酒业", {'weight': 1/1.3648}),
         ("古井贡酒", "迎驾贡酒", {'weight': 1/1.3687}),
         ("泸州老窖", "古井贡酒", {'weight': 1/1.4062}),]
G.add_edges_from(edges)

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_size=2000,
        node_color='lightblue', font_size=10)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
plt.show()
