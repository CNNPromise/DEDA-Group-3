# %%
import ta
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Get the stock quote
df_all = pd.read_csv('stockdata_adj.csv')

# Choose Maotai from all stocks
df_all = df_all[df_all['ts_code'] == '600519.SH']

# sort the data and set the date as index
df_all.sort_values('trade_date', inplace=True)
df_all.set_index('trade_date', inplace=True)

df = df_all[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]

# Calculating percent volatility
df.loc[:, 'HIGHLOW_PCT'] = (df['adj_high'] - df['adj_close']
                            ) / df['adj_close'] * 100

# Calculating some factors
df.loc[:, 'PCT_Change'] = (df['adj_close'] - df['adj_open']
                           ) / df['adj_open'] * 100
df.loc[:, 'PCT_Change_overnight'] = df["adj_close"].pct_change()
df.loc[:, "MA5"] = df["adj_close"].rolling(window=5).mean()
df.loc[:, "STD5"] = df["adj_close"].rolling(window=5).std()

# Add momentum indicators

# ta.momentum.AwesomeOscillatorIndicator
indicator_ao = ta.momentum.AwesomeOscillatorIndicator(
    high=df["adj_high"], low=df["adj_low"], window1=5, window2=34)
df.loc[:, 'ao'] = indicator_ao.awesome_oscillator()

# ta.momentum.KAMAIndicato
indicator_kama = ta.momentum.KAMAIndicator(
    close=df["adj_close"], window=10, pow1=2, pow2=30)
df.loc[:, 'kama'] = indicator_kama.kama()

# ta.momentum.PercentagePriceOscillator
indicator_ppo = ta.momentum.PercentagePriceOscillator(
    close=df["adj_close"], window_slow=26, window_fast=12, window_sign=9)
df.loc[:, 'ppo'] = indicator_ppo.ppo()

# ta.momentum.PercentageVolumeOscillator
indicator_pvo = ta.momentum.PercentageVolumeOscillator(
    volume=df["adj_volume"], window_slow=26, window_fast=12, window_sign=9)
df.loc[:, 'pvo'] = indicator_pvo.pvo()

# ta.momentum.ROCIndicator
indicator_roc = ta.momentum.ROCIndicator(
    close=df["adj_close"], window=12)
df.loc[:, 'roc'] = indicator_roc.roc()

# ta.momentum.RSIIndicator
indicator_rsi = ta.momentum.RSIIndicator(
    close=df["adj_close"], window=14)
df.loc[:, 'rsi'] = indicator_rsi.rsi()

# ta.momentum.StochRSIIndicator
indicator_stochrsi = ta.momentum.StochRSIIndicator(
    close=df["adj_close"], window=14, smooth1=3, smooth2=3)
df.loc[:, 'stochrsi'] = indicator_stochrsi.stochrsi()

# ta.momentum.StochasticOscillator
indicator_stoch = ta.momentum.StochasticOscillator(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    window=14, smooth_window=3)
df.loc[:, 'stoch'] = indicator_stoch.stoch()

# ta.momentum.TSIIndicator
indicator_tsi = ta.momentum.TSIIndicator(
    close=df["adj_close"], window_slow=25, window_fast=13)
df.loc[:, 'tsi'] = indicator_tsi.tsi()

# ta.momentum.UltimateOscillator
indicator_uo = ta.momentum.UltimateOscillator(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    window1=7, window2=14, window3=28, weight1=4.0, weight2=2.0, weight3=1.0,
    fillna=True)
df.loc[:, 'uo'] = indicator_uo.ultimate_oscillator()

# ta.momentum.WilliamsRIndicator
indicator_wr = ta.momentum.WilliamsRIndicator(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"], lbp=14)
df.loc[:, 'wr'] = indicator_wr.williams_r()


# Add volume indicators

# ta.volume.AccDistIndexIndicator
indicator_adi = ta.volume.AccDistIndexIndicator(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    volume=df["adj_volume"])
df.loc[:, 'adi'] = indicator_adi.acc_dist_index()

# ta.volume.ChaikinMoneyFlowIndicator
indicator_cmf = ta.volume.ChaikinMoneyFlowIndicator(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    volume=df["adj_volume"], window=20, fillna=True)
df.loc[:, 'cmf'] = indicator_cmf.chaikin_money_flow()

# ta.volume.EaseOfMovementIndicator
indicator_eom = ta.volume.EaseOfMovementIndicator(
    high=df["adj_high"], low=df["adj_low"], volume=df["adj_volume"],
    window=14, fillna=True)
df.loc[:, 'eom'] = indicator_eom.ease_of_movement()

# ta.volume.ForceIndexIndicator
indicator_fi = ta.volume.ForceIndexIndicator(
    close=df["adj_close"], volume=df["adj_volume"], window=13, fillna=True)
df.loc[:, 'fi'] = indicator_fi.force_index()

# ta.volume.MFIIndicator
indicator_mfi = ta.volume.MFIIndicator(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    volume=df["adj_volume"], window=14, fillna=True)
df.loc[:, 'mfi'] = indicator_mfi.money_flow_index()

# ta.volume.NegativeVolumeIndexIndicator
indicator_nvi = ta.volume.NegativeVolumeIndexIndicator(
    close=df["adj_close"], volume=df["adj_volume"], fillna=True)
df.loc[:, 'nvi'] = indicator_nvi.negative_volume_index()

# ta.volume.OnBalanceVolumeIndicator
indicator_obv = ta.volume.OnBalanceVolumeIndicator(
    close=df["adj_close"], volume=df["adj_volume"], fillna=True)
df.loc[:, 'obv'] = indicator_obv.on_balance_volume()

# ta.volume.VolumePriceTrendIndicator
indicator_vpt = ta.volume.VolumePriceTrendIndicator(
    close=df["adj_close"], volume=df["adj_volume"], fillna=True)
df.loc[:, 'vpt'] = indicator_vpt.volume_price_trend()

# ta.volume.VolumeWeightedAveragePrice
indicator_vwap = ta.volume.VolumeWeightedAveragePrice(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    volume=df["adj_volume"], window=14, fillna=True)
df.loc[:, 'vwap'] = indicator_vwap.volume_weighted_average_price()

# Add Volatility Indicators

# ta.volatility.AverageTrueRange
indicator_atr = ta.volatility.AverageTrueRange(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    window=14, fillna=True)
df.loc[:, 'atr'] = indicator_atr.average_true_range()

# ta.volatility.BollingerBands
indicator_bb = ta.volatility.BollingerBands(
    close=df["adj_close"], window=20, window_dev=2)
df.loc[:, 'bb_bbm'] = indicator_bb.bollinger_mavg()
df.loc[:, 'bb_bbh'] = indicator_bb.bollinger_hband()
df.loc[:, 'bb_bbl'] = indicator_bb.bollinger_lband()
df.loc[:, 'bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
df.loc[:, 'bb_bbli'] = indicator_bb.bollinger_lband_indicator()

# ta.volatility.DonchianChannel
indicator_dc = ta.volatility.DonchianChannel(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    window=20, fillna=True)
df.loc[:, 'dc_hband'] = indicator_dc.donchian_channel_hband()
df.loc[:, 'dc_lband'] = indicator_dc.donchian_channel_lband()

# ta.volatility.KeltnerChannel
indicator_kc = ta.volatility.KeltnerChannel(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    window=20, window_atr=10, fillna=True)
df.loc[:, 'kc_hband'] = indicator_kc.keltner_channel_hband()
df.loc[:, 'kc_lband'] = indicator_kc.keltner_channel_lband()
df.loc[:, 'kc_hband_ind'] = indicator_kc.keltner_channel_hband_indicator()
df.loc[:, 'kc_lband_ind'] = indicator_kc.keltner_channel_lband_indicator()

# ta.volatility.UlcerIndex
indicator_ui = ta.volatility.UlcerIndex(close=df["adj_close"], window=14)
df.loc[:, 'ui'] = indicator_ui.ulcer_index()

# Add trend indicators

# ta.trend.ADXIndicator
indicator_adx = ta.trend.ADXIndicator(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    window=14, fillna=True)
df.loc[:, 'adx'] = indicator_adx.adx()
df.loc[:, 'adx_pos'] = indicator_adx.adx_pos()

# ta.trend.AroonIndicator
indicator_aroon = ta.trend.AroonIndicator(
    high=df["adj_high"], low=df["adj_low"], window=25, fillna=True)
df.loc[:, 'aroon_down'] = indicator_aroon.aroon_down()
df.loc[:, 'aroon_ind'] = indicator_aroon.aroon_indicator()
df.loc[:, 'aroon_up'] = indicator_aroon.aroon_up()

# ta.trend.DPOIndicator
indicator_dpo = ta.trend.DPOIndicator(
    close=df["adj_close"], window=20, fillna=True)
df.loc[:, 'dpo'] = indicator_dpo.dpo()

# ta.trend.EMAIndicator
indicator_ema = ta.trend.EMAIndicator(
    close=df["adj_close"], window=12, fillna=True)
df.loc[:, 'ema'] = indicator_ema.ema_indicator()

# ta.trend.IchimokuIndicator
indicator_ichimoku = ta.trend.IchimokuIndicator(
    high=df["adj_high"], low=df["adj_low"],
    window1=9, window2=26, window3=52, visual=False, fillna=True)
df.loc[:, 'ichimoku_a'] = indicator_ichimoku.ichimoku_a()
df.loc[:, 'ichimoku_b'] = indicator_ichimoku.ichimoku_b()
df.loc[:, 'ichimoku_base'] = indicator_ichimoku.ichimoku_base_line()
df.loc[:, 'ichimoku_conversion'] = indicator_ichimoku.ichimoku_conversion_line()

# ta.trend.KSTIndicator
indicator_kst = ta.trend.KSTIndicator(
    close=df["adj_close"], roc1=10, roc2=15, roc3=20, roc4=30,
    nsig=9, fillna=True)
df.loc[:, 'kst'] = indicator_kst.kst()
df.loc[:, 'kst_diff'] = indicator_kst.kst_diff()
df.loc[:, 'kst_sig'] = indicator_kst.kst_sig()

# ta.trend.MACD
indicator_macd = ta.trend.MACD(
    close=df["adj_close"], window_slow=26, window_fast=12,
    window_sign=9, fillna=True)
df.loc[:, 'macd'] = indicator_macd.macd()
df.loc[:, 'macd_diff'] = indicator_macd.macd_diff()
df.loc[:, 'macd_signal'] = indicator_macd.macd_signal()

# ta.trend.MassIndex
indicator_mi = ta.trend.MassIndex(
    high=df["adj_high"], low=df["adj_low"],
    window_fast=9, window_slow=25, fillna=True)
df.loc[:, 'mi'] = indicator_mi.mass_index()

# ta.trend.PSARIndicator
indicator_psar = ta.trend.PSARIndicator(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    step=0.02, max_step=0.2, fillna=True)
df.loc[:, 'psar'] = indicator_psar.psar()
df.loc[:, 'psar_down'] = indicator_psar.psar_down()
df.loc[:, 'psar_down_ind'] = indicator_psar.psar_down_indicator()
df.loc[:, 'psar_up'] = indicator_psar.psar_up()
df.loc[:, 'psar_up_ind'] = indicator_psar.psar_up_indicator()

# ta.trend.SMAIndicator
indicator_sma = ta.trend.SMAIndicator(
    close=df["adj_close"], window=20, fillna=True)
df.loc[:, 'sma'] = indicator_sma.sma_indicator()

# ta.trend.STCIndicator
indicator_stc = ta.trend.STCIndicator(
    close=df["adj_close"], window_slow=50, window_fast=23,
    cycle=10, smooth1=3, smooth2=3, fillna=True)
df.loc[:, 'stc'] = indicator_stc.stc()

# ta.trend.TRIXIndicator
indicator_trix = ta.trend.TRIXIndicator(
    close=df["adj_close"], window=15, fillna=True)
df.loc[:, 'trix'] = indicator_trix.trix()

# ta.trend.VortexIndicator
indicator_vortex = ta.trend.VortexIndicator(
    high=df["adj_high"], low=df["adj_low"], close=df["adj_close"],
    window=14, fillna=True)
df.loc[:, 'vortex_ind_diff'] = indicator_vortex.vortex_indicator_diff()
df.loc[:, 'vortex_ind_neg'] = indicator_vortex.vortex_indicator_neg()
df.loc[:, 'vortex_ind_pos'] = indicator_vortex.vortex_indicator_pos()

# ta.trend.WMAIndicator
indicator_wma = ta.trend.WMAIndicator(
    close=df["adj_close"], window=9, fillna=True)
df.loc[:, 'wma'] = indicator_wma.wma()

# Add Others Indicators

# save factors to csv
df.to_csv('stockdata_factors.csv')


# %%
# characteristics of the factors

df_factors = pd.read_csv("stockdata_factors.csv")


fig = make_subplots(rows=4, cols=1)
fig.add_trace(go.Scatter(x=df_factors['trade_date'], y=df_factors['adj_close'],
                         name='Close Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_factors['trade_date'], y=df_factors['MA5'],
                         name='MA5'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_factors['trade_date'], y=df_factors['macd'],
                         name='MACD'), row=2, col=1)
fig.add_trace(go.Scatter(x=df_factors['trade_date'], y=df_factors['rsi'],
                         name='RSI'), row=3, col=1)
fig.add_trace(go.Scatter(x=df_factors['trade_date'], y=df_factors['vwap'],
                         name='VWAP'), row=4, col=1)
fig.update_layout(title_text='Technical Indicators')
fig.show()

# Save the plot
fig.write_html("technical_indicators.html")

# standardize the factors

df.loc[:, 'label'] = df.loc[:, "adj_close"].pct_change().shift(-1)
dflabel = df['label']

plt.scatter(df['vwap'], df['label'])


# %%
# Standardize the factors

scaler = StandardScaler()

# Standardize each column except the 'label' column in the DataFrame
df_standardized = pd.DataFrame(scaler.fit_transform(df),
                               columns=df.columns, index=df.index)

df_standardized = df_standardized.drop(columns=['label'])
df_standardized['label'] = dflabel

df_standardized.dropna(inplace=True)

# get all trade dates in df_standardized, and select them in df

trade_dates = df_standardized.index
df = df.loc[trade_dates]

plt.scatter(df_standardized['vwap'], df_standardized['label'])

# %%
