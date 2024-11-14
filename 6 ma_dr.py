import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('stockdata_adj.csv')

# sort the data by date

df['MA100'] = df['adj_close'].rolling(window=100).mean()
df['MA200'] = df['adj_close'].rolling(window=200).mean()
df['MA50'] = df['adj_close'].rolling(window=50).mean()

# Plot The Moving Average

# | 000568 | 泸州老窖 |
# | 600519 | 贵州茅台 |
# | 000858 | 五 粮 液 |
# | 600559 | 老白干酒 |

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

stock_data = df[df.ts_code == '600519.SH'][::-1].iloc[:-200]
stock_data.set_index('trade_date', inplace=True)
stock_data[['adj_close', 'MA100', 'MA200', 'MA50']].plot(ax=axes[0, 0])
axes[0, 0].set_title('GuiZhou Maotai')

stock_data = df[df.ts_code == '000568.SZ'][::-1].iloc[:-200]
stock_data.set_index('trade_date', inplace=True)
stock_data[['adj_close', 'MA100', 'MA200', 'MA50']].plot(ax=axes[0, 1])
axes[0, 1].set_title('Wuliangye')

stock_data = df[df.ts_code == '000858.SZ'][::-1].iloc[:-200]
stock_data.set_index('trade_date', inplace=True)
stock_data[['adj_close', 'MA100', 'MA200', 'MA50']].plot(ax=axes[1, 0])
axes[1, 0].set_title('Luzhou Laojiao')

stock_data = df[df.ts_code == '600559.SH'][::-1].iloc[:-200]
stock_data.set_index('trade_date', inplace=True)
stock_data[['adj_close', 'MA100', 'MA200', 'MA50']].plot(ax=axes[1, 1])
axes[1, 1].set_title('Lao Baigan Jiu')

plt.tight_layout()
plt.show()

# plot pct_chg of the stocks
# | 000568 | 泸州老窖 |
# | 600519 | 贵州茅台 |
# | 000858 | 五 粮 液 |
# | 600559 | 老白干酒 |

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

stock_data = df[df.ts_code == '600519.SH'][::-1].iloc[:-200]
stock_data.set_index('trade_date', inplace=True)
stock_data['pct_chg'].plot(ax=axes[0, 0])
axes[0, 0].set_title('GuiZhou Maotai')

stock_data = df[df.ts_code == '000568.SZ'][::-1].iloc[:-200]
stock_data.set_index('trade_date', inplace=True)
stock_data['pct_chg'].plot(ax=axes[0, 1])
axes[0, 1].set_title('Wuliangye')

stock_data = df[df.ts_code == '000858.SZ'][::-1].iloc[:-200]
stock_data.set_index('trade_date', inplace=True)
stock_data['pct_chg'].plot(ax=axes[1, 0])
axes[1, 0].set_title('Luzhou Laojiao')

stock_data = df[df.ts_code == '600559.SH'][::-1].iloc[:-200]
stock_data.set_index('trade_date', inplace=True)
stock_data['pct_chg'].plot(ax=axes[1, 1])
axes[1, 1].set_title('Lao Baigan Jiu')

plt.tight_layout()
plt.show()
