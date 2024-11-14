import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('stockdata_adj.csv')

# choose 贵州茅台 600519.SZ from all stocks
stock_data = df[df.ts_code == '600519.SH']

# sort the data by date
stock_data = stock_data.sort_values('trade_date')
stock_data.set_index('trade_date', inplace=True)

# Plot graph
fig = go.Figure(
        data=[
            go.Candlestick(
                x=stock_data.index,
                open=stock_data.adj_open,
                high=stock_data.adj_high,
                low=stock_data.adj_low,
                close=stock_data.adj_close
                )
            ]
        )


fig.update_layout(
    title='Time Series with Range slider for Guizhou Maotai',
    xaxis_title='Date',
    yaxis_title='Price (CNY)',
    xaxis_rangeslider_visible=True
)

fig.show()

# save the plot
fig.write_html("stock_price.html")
