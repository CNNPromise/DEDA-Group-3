import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts
import baostock as bs

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# This is my personal token, just for the purpose of this project.
# FOR COURSE DEDA.
pro = ts.pro_api(
    token="ee489375697c830bbd5c468d8540c28fa05461f64e87c6da1ff54106")

# get stock data of Baijiu industry from 2014-09-20 to 2024-09-20

# # Example: 贵州茅台
# df1 = pro.daily(ts_code='600519.SH', start_date='20140920',
# end_date='20240920')
# df1.tail()

# All stocks in Baijiu industry

stocklist = ["600696.SH", "600543.SH", "600084.SH", "603198.SH", "600616.SH",
             "600199.SH", "600365.SH", "600573.SH", "600059.SH", "600519.SH",
             "603589.SH", "600809.SH", "600779.SH", "600559.SH", "603369.SH",
             "600132.SH", "600702.SH", "601579.SH", "600238.SH", "600197.SH",
             "603919.SH", "600600.SH", "603779.SH", "000729.SZ", "000995.SZ",
             "000869.SZ", "000568.SZ", "000858.SZ", "000860.SZ", "000799.SZ",
             "000752.SZ", "000929.SZ", "000596.SZ"]

# let df a dataframe with no data, then concate all stock data to df

df = pd.DataFrame()
for stock in stocklist:
    df1 = pro.daily(ts_code=stock, start_date='20140920', end_date='20240920')
    df = pd.concat([df, df1])

df.to_csv("stockdata.csv", index=False)

dfadj = pd.DataFrame()
for stock in stocklist:
    dfadj1 = pro.adj_factor(ts_code=stock,
                            start_date='20140920', end_date='20240920')
    dfadj = pd.concat([dfadj, dfadj1])

dfadj.to_csv("stockadj.csv", index=False)


# concate "stockadj.csv" and "stockdata.csv" by "ts_code" and "trade_date"

df = pd.read_csv("stockdata.csv")
dfadj = pd.read_csv("stockadj.csv")

df1 = pd.merge(df, dfadj, on=["ts_code", "trade_date"])
df1.head()

# calculate the adjusted open, high, low, close, and volume

df1["adj_open"] = df1["open"] * df1["adj_factor"]
df1["adj_high"] = df1["high"] * df1["adj_factor"]
df1["adj_low"] = df1["low"] * df1["adj_factor"]
df1["adj_close"] = df1["close"] * df1["adj_factor"]
df1["adj_volume"] = df1["vol"] / df1["adj_factor"]

df1.to_csv("stockdata_adj.csv", index=False)

# transform the trade_date like "20231221" to datetime 2023-12-21

df1['trade_date'] = pd.to_datetime(df1['trade_date'], format='%Y%m%d')

# df1 = df1.set_index('trade_date')

df1.to_csv("stockdata_adj.csv", index=False)


# get the index data
lg = bs.login()
rs = bs.query_history_k_data_plus("sh.000300",
                                  "date,code,open,high,low,close,volume,amount",
                                  start_date='2014-09-20',
                                  end_date='2024-09-20',
                                  frequency="d")

data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
result.to_csv("indexdata.csv", index=False)
bs.logout()
