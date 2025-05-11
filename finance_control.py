import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
def getter(coin, days):
    if days < 101:
        url = f"https://www.okx.com/api/v5/market/history-mark-price-candles?instId={coin}&bar=1D&limit={days}"
        data = requests.get(url).json()
        df = pd.DataFrame(data['data'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df = df[['datetime', 'close', 'high', 'low']]
        df['datetime'] = pd.to_datetime(df['datetime'].astype(float), unit='ms')
        df['close'] = pd.to_numeric(df['close'], errors="coerce")
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
    else:
        new_days = int(days/100)
        new_days_s = days%100
        new_dataframe = pd.DataFrame()
        url = f"https://www.okx.com/api/v5/market/history-mark-price-candles?instId={coin}&bar=1D&limit={days}"
        data = requests.get(url).json()
        df = pd.DataFrame(data['data'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df = df[['datetime', 'close', 'high', 'low']]
        df['datetime'] = pd.to_datetime(df['datetime'].astype(float), unit='ms')
        df['close'] = pd.to_numeric(df['close'], errors="coerce")
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        new_dataframe = pd.concat([new_dataframe, df], axis=0)
        last_timestamp = data['data'][-1][0]
        for _ in range(1, new_days):
            url = f"https://www.okx.com/api/v5/market/history-mark-price-candles?instId={coin}&bar=1D&limit={days}&after={last_timestamp}"
            data = requests.get(url).json()
            df = pd.DataFrame(data['data'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df = df[['datetime', 'close', 'high', 'low']]
            df['datetime'] = pd.to_datetime(df['datetime'].astype(float), unit='ms')
            df['close'] = pd.to_numeric(df['close'], errors="coerce")
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            new_dataframe = pd.concat([new_dataframe, df], axis=0)
            last_timestamp = data['data'][-1][0]
        if new_days_s > 0:
            last_timestamp = data['data'][-1][0]
            url = f"https://www.okx.com/api/v5/market/history-mark-price-candles?instId={coin}&bar=1D&limit={new_days_s}&after={last_timestamp}"
            data = requests.get(url).json()
            df = pd.DataFrame(data['data'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df = df[['datetime', 'close', 'high', 'low']]
            df['datetime'] = pd.to_datetime(df['datetime'].astype(float), unit='ms')
            df['close'] = pd.to_numeric(df['close'], errors="coerce")
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            new_dataframe = pd.concat([new_dataframe, df], axis=0)
        return new_dataframe
    return df
# getter('BTC-USDT', 200)
# df = getter('BTC-USDT', 16)
# print(df)

def sma(price):
    if len(price) != 0:
        sma_ = 0
        for i in price:
            sma_+=float(i)
        sma_ = sma_/len(price)
        return sma_
    else:
        return f"Price is empty."

# -----------------------------
def checking_trend(coin, s_day, b_day):
    dfs = getter(coin, s_day)
    dfb = getter(coin, b_day)
    smas = sma(dfs['close'])
    smab = sma(dfb['close'])
    diff = smas - smab
    if smas > smab:
        return (f"The trend will go up, a signal to buy! SMA {s_day} = {smas}, SMA {b_day} = {smab}", smas, smab, diff)
    elif smas < smab:
        return (f"The trend will go down, a signal to sell! SMA {s_day} = {smas}, SMA {b_day} = {smab}", smas, smab, diff)
    else:
        return (f"The trend is balanced, time to wait! SMA {s_day} = {smas}, SMA {b_day} = {smab}", smas, smab, diff)

# -------------------------

# response = checking_trend("BTC-USD", 7, 25)
# print(response)

def period_trend(coin, days, period):
    data = getter(coin, days)
    price = data['close']
    date = data['datetime']
    count = 0
    temp_price = []
    result_price = []
    temp_date = []
    result_date = []
    for i, j in zip(price, date):
        temp_price.append(i)
        temp_date.append(j)
        count+=1
        if count == period:
            result_price.append(temp_price[:])
            result_date.append(temp_date[:])
            temp_price.clear()
            temp_date.clear()
            count = 0
    return (result_price, result_date)

def trends_walk_sma(coin, days, period_1, period_2):
    data = getter(coin, days)
    price = data['close']
    date = data['datetime']

    price_s, date_s = period_trend(coin, days, period_1)
    price_b, date_b = period_trend(coin, days, period_2)

    price_s_arr = []
    date_s_ = []
    for i, j in zip(price_s, date_s):
        price_s_arr.append(sma(i))
        date_s_.append(j[0])
    temp = []
    temp.append(price.iloc[-1])
    price_s_arr.append(sma(temp))
    date_s_.append(date.iloc[-1])

    price_b_arr = []
    date_b_ = []
    for i, j in zip(price_b, date_b):
        price_b_arr.append(sma(i))
        date_b_.append(j[0])
    date_b_.append(date.iloc[-1])
    price_b_arr.append(sma(temp))
    
    price_b_arr = np.array(price_b_arr)
    date_b_ = np.array(date_b_).ravel()

    check, _, smab, diff = checking_trend(coin, period_1, period_2)
    plt.plot(date, price, color = "blue", label = f"The price of the coin")
    plt.plot(date_s_, price_s_arr, color = "green", label = f"A signal for a positive trend")
    plt.plot(date_b_, price_b_arr, color = "red", label = f"A signal for a negative trend")
    plt.plot(date, price, color = "white", alpha=0, label = f"{check}")
    plt.plot(date, price, color = "white", alpha=0, label = f"Difference: {diff}")
    plt.grid(True)
    plt.legend()
    plt.show()
    return check
    
# a = trends_walk_sma('BTC-USDT', 50, 10, 20)
# print(a)

def gibrid_ema(price):
    ema_n_1 = price[0]
    k = 2/(len(price)+1)
    ema = []
    for i in price[1:]:
        e = (i*k)+(ema_n_1*(1-k))
        ema.append(e)
        ema_n_1 = e
    return sum(ema)/len(ema)
# resp = ema([10, 20, 30, 20, 10])
# print(resp)

def checking_trend_rolling_ema(emas, emab, day_s, day_b):
    diff = emas-emab
    if emas > emab:
        return (f"The trend will go up, a signal to buy! REMA {day_s} = {emas}, REMA {day_b} = {emab}", emas, emab, diff)
    elif emas < emab:
        return (f"The trend will go down, a signal to sell! REMA {day_s} = {emas}, REMA {day_b} = {emab}", emas, emab, diff)
    else:
        return (f"The trend is balanced, time to wait! REMA {day_s} = {emas}, REMA {day_b} = {emab}", emas, emab, diff)
# trends_walk_('bitcoin', 20, 200)

def real_roll_ema(prices, period, dates):
    rema_n = sum(prices[:period]) / period
    rma_values = []
    for price in prices[period:]:
        rema_n = (rema_n * (period - 1) + price) / period
        rma_values.append(rema_n)
    rma_dates = dates.iloc[:len(dates)-period]
    return rma_values, rma_dates

def trend_walk_rolling_ema(coin, days, period_1, period_2):
    data = getter(coin, days)
    date = data['datetime']
    price = data['close']

    price_s, date_s = period_trend(coin, days, period_1)
    price_b, date_b = period_trend(coin, days, period_2)

    price_s_r, date_s_r = real_roll_ema(price, period_1, date)
    price_b_r, date_b_r = real_roll_ema(price, period_2, date)
    price_s_arr = []
    date_s_ = []

    temp_p = price.iloc[-1]
    temp_b = date.iloc[-1]
    for i, j in zip(price_s, date_s):
        price_s_arr.append(gibrid_ema(i))
        date_s_.append(j[0])
    price_s_arr.append(temp_p)
    date_s_.append(temp_b)
    
    price_b_arr = []
    date_b_ = []
    for i, j in zip(price_b, date_b):
        price_b_arr.append(gibrid_ema(i))
        date_b_.append(j[0])
    price_b_arr.append(temp_p)
    date_b_.append(temp_b)

    check, _, _, diff = checking_trend_rolling_ema(price_s_arr[0], price_b_arr[0], period_1, period_2)
    check_1, _, _, diff_1 = checking_trend_rolling_ema(price_s_r[0], price_b_r[0], period_1, period_2)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(date, price, color = "blue", label = f"Gibrid EMA")
    ax[0].plot(date_s_, price_s_arr, color = "green", label = f"A signal for a positive trend")
    ax[0].plot(date_b_, price_b_arr, color = "red", label = f"A signal for a negative trend")
    # ax[0].plot(date, price, alpha=0, label = f"{check}")
    # ax[0].plot(date, price, alpha=0, label = f"Difference: {diff}")
    ax[1].plot(date, price, color ="blue", label = f"Running(Rolling) EMA")
    ax[1].plot(date_s_r, price_s_r, color = "green", label = f"A signal for a positive trend")
    ax[1].plot(date_b_r, price_b_r, color = "red", label = f"A signal for a negative trend")
    # ax[1].plot(date, price, alpha=0, label = f"{check_1}")
    # ax[1].plot(date, price, alpha=0, label = f"Difference: {diff_1}")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].legend()
    ax[1].legend()
    plt.show()
    print(check, diff, check_1, diff_1)
    if period_1 < 10 or period_2 < 10:
        return (f"Be careful, the forecast is short-term momentum. \n!Check the trend on medium timeframes!", check, diff, check_1, diff_1)
    elif period_1 < 26 or period_2 < 26:
        return("A real trend. For long-term trends, use longer timeframes(10-25).", check, diff, check_1, diff_1)
    else:
        return("Long-term trend.", check, diff, check_1, diff_1)

# resp = trend_walk_rolling_ema('BTC-USDT', 100, 10, 15)
# print(resp)

def rsi(coin, days):
    df = getter(coin, days+1)
    price = df['close']
    lesion = []
    profit = []
    i1 = price.iloc[0]
    print(i1)
    if len(price) < 2:
        return ("Not enough data!", 0)
    else:
        for i in price.iloc[1:]:
            diff = i1-i
            if diff < 0:
                lesion.append(diff)
                profit.append(0)
            else:
                profit.append(diff)
                lesion.append(0)
            i1 = i
        print(lesion, profit)
        if (len(profit) != 0 and len(lesion) != 0) and (len(profit) == days and len(lesion) == days):
            rs = np.mean(profit)/np.mean(np.abs(lesion))    
            rsi_ = 100-(100/(1+rs))
            if rsi_ < 30:
                return (f"The trend will be oversold, a signal to buy! RSI - {rsi_}", rsi_)
            if rsi_ > 70:
                return (f"The trend will go overbought, a signal to sell! RSI - {rsi_}", rsi_)
            else:
                return (f"The trend is balanced, time to wait! RSI - {rsi_}", rsi_)
        else:
            return (f"The *Loss* or *Profit* array is empty.", 0)
# a, b = rsi("BTC-USDT", 10)
# print(a, b)
def checking(coin, days, period_1, period_2):
    _, _, smab, _ = checking_trend(coin, period_1, period_2)
    data = getter(coin, days)
    if data['close'][0] > smab:
        print("The trend, most of all, will go up (following a smaller SMA).")
        return 1
    else:
        print("The trend, most of all, will go down (by a larger SMA).")
        return 0

def rsi_sma(coin, days, period_1, period_2):
    _, rsi_in = rsi(coin, days)
    check = checking(coin, days, period_1, period_2)
    print(rsi_in, check)
    if rsi_in <= 40 and check == 1:
        return("Potential to buy from a pullback.")
    elif rsi_in >= 60 and check == 0:
        return("Sales potential.")
    else:
        return("For the time being, wait patiently.")
# resp = rsi('BTC-USDT', 300)
# print(resp)

def ichimoku_line(minimum, maximum, date, period):
    result = []
    date_in = []
    for i in range(period, len(minimum)):
        date_in.append(date.iloc[i])
        res_ = (min(minimum[i-period:i]) + max(maximum[i-period:i])) / 2
        result.append(res_)
    return result, date_in

def ichimoku_cloud(coin, days):
    data = getter(coin, days)
    date = data['datetime']
    price = data['close']
    minimum = data['low']
    maximum = data['high']
    if days >= 9:
        tenkan, date_in = ichimoku_line(minimum, maximum, date, 9)
    else:
        tenkan, date_in = [], []

    if days >= 26:
        kijun, date_in_ = ichimoku_line(minimum, maximum, date, 26)
    else:
        kijun, date_in_ = [], []

    senkou_span_a = [(t + k) / 2 for t, k in zip(tenkan, kijun)] if tenkan and kijun else []

    if days >= 52:
        senkou_span_b, date_in_b = ichimoku_line(minimum, maximum, date, 52)
    else:
        senkou_span_b, date_in_b = [], []

    chikou_span = price.iloc[:-26]
    date_chikou = date.iloc[26:len(date)]
    min_span = senkou_span_a[0]
    max_span = senkou_span_b[0]
    print(min_span, max_span)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5))
    ax[0].plot(date, price, color = "blue", label = "The price of the coin")
    ax[0].plot(date_in, tenkan, color = "green", label = "Tekan-sen")
    ax[0].plot(date_in_, kijun, color = "red", label = "Kijun-sen")
    ax[0].plot(date_chikou, chikou_span, color = "gray", label = "Chikou-sen")
    ax[1].plot(date, price, color = "blue", label = "The price of the coin")
    ax[1].plot(date_in_, senkou_span_a, color = "purple", label = "Senkou Span A")
    ax[1].plot(date_in_b, senkou_span_b, color = "black", label = "Senkou Span B")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    text = "Current situation:\n"
    if price.iloc[0] > min_span and price.iloc[0] > max_span:
        text+="The trend is bullish.\n"
    elif price.iloc[0] < min_span and price.iloc[0] < max_span:
        text+="The trend is bearish.\n"
    else:
        text+="Now is the time of the flute.\n"
    text += "Market entry signals:\n"
    if tenkan[0] > kijun[0]:
        if tenkan[0] > max_span:
            text+="A strong buy signal.\n"
        elif tenkan[0] < max_span and tenkan[0] > min_span:
            text+='Neutral signal. It is better to wait.\n'
        else:
            text+='Weak signal. It is better to wait.\n'
    text += "Where the market will go:\n"
    if chikou_span.iloc[0] > price.iloc[26]:
        text+='The trend will go up.'
    else:
        text+='The trend will go down.'
    return text
# text = ichimoku_cloud("BTC-USDT", 200)
# print(text)
def classic_ema(price, limit, days):
    ema_t_1_price = price.iloc[:limit]
    ema_t_1 = sma(ema_t_1_price)
    k = 2/(limit+1)
    ema_new = [ema_t_1]
    for i in range(limit, days):
        ema_ = (price.iloc[i]*k)+(ema_t_1*(1-k))
        ema_new.append(ema_)
        ema_t_1 = ema_
    return ema_new

def trend_walk_classic_ema(coin, period_1, period_2, days):
    data = getter(coin, days)
    price = data['close']
    date = data['datetime']
    ema_p_1 = classic_ema(price, period_1, days)
    ema_p_1D = date.iloc[:days-(period_1-1)].reset_index(drop=True)
    ema_p_2 = classic_ema(price, period_2, days)
    ema_p_2D = date.iloc[:days-(period_2-1)].reset_index(drop=True)
    plt.plot(date, price, color = "blue", label = "The price of the coin")
    plt.plot(ema_p_1D, ema_p_1, color = "green", label = "A signal for a positive trend")
    plt.plot(ema_p_2D, ema_p_2, color = "red", label = "A signal for a negative trend")
    plt.grid(True)
    plt.legend()
    plt.show()
# trend_walk_classic_ema("BTC-USDT", 10, 20, 100)

def check_ema(coin, period, days):
    data = getter(coin, days)
    ema_ = classic_ema(data['close'], period, days)
    if data['close'][0] > ema_[0]:
        return("A bullish trend dominates now.")
    elif data['close'][0] == ema_[0]:
        return("The trend is uncertain, the price is on par with the EMA!")
    else:
        return("A bearish trend dominates now.")
# trend_walk_classic_ema("BTC-USDT", 10, 20, 100)
# check_ema("BTC-USDT", 20, 100)
def macd(coin, days):
    if days >= 35:
        data = getter(coin, days)
        price = data['close']
        date = data['datetime']
        print(date)
        ema_12 = classic_ema(price, 12, days)
        ema_26 = classic_ema(price, 26, days)
        macd_line = []
        date_macd = []
        minimum = min(len(ema_12), len(ema_26))
        ema_diff = np.array(ema_12[:minimum]) - np.array(ema_26[:minimum])
        std_dev = np.std(ema_diff)
        text = "What is the trend at the moment: \n"
        if ema_diff[0] > std_dev:
            text += "Bullish trend, strong momentum.\n"
        elif ema_diff[0] < -std_dev:
            text += "Bearish trend, strong momentum.\n"
        else:
            text += "Flat, weak pulse.\n"
        
        text += "What is the trend at the moment (check 2): \n"
        for i, j, d in zip(ema_12, ema_26, date.iloc[12:]):
            macd_line.append(i-j)
            date_macd.append(d)
        if macd_line[0] > 0:
            text += "The bullish trend dominates.\n"
        else:
            text += "The bearish trend dominates.\n"
        signal_date = date_macd[:len(date_macd)-8]
        signal_line = classic_ema(pd.Series(macd_line), 9, len(macd_line))
        text += "Stronger signal: \n"
        if macd_line[0] > signal_line[0]:
            text += "Buy signal!\n"
        elif macd_line[0] < signal_line[0]:
            text += "Signal for sale!\n"
        else:
            text += "Time to wait!\n"
        histogram = []
        histogram_date = []
        for i, j, d in zip(macd_line, signal_line, date_macd):
            histogram.append(i-j)
            histogram_date.append(d)
        text+="Determination of impulse by histogram:\n"
        if histogram[0] > 0 and histogram[0] > histogram[1]:
            text += "Bullish momentum.\n"
        elif histogram[0] < 0 and histogram[0] < histogram[1]:
            text += "Bearish momentum.\n"
        elif histogram[0] > 0 and histogram[0] < histogram[1]:
            text += "There is a slowdown, a reversal to a bearish trend.\n"
        elif histogram[0] < 0 and histogram[0] > histogram[1]:
            text += "There is a fading, reversal to a bullish trend.\n"
        else:
            text += "Time flies, wait a while.\n"

        plt.plot(date_macd, macd_line, color = "green", label = "MACD line")
        plt.plot(signal_date, signal_line, color = "red", label = "Signal line")
        plt.plot(histogram_date, histogram, color = "black", label = "Histogram")
        plt.legend()
        plt.grid(True)
        plt.show()

        return text
    else:
        return "Too little data!"
# resp = macd("BTC-USDT", 70)
# print(resp)
def stochastic_oscillator(coin, period, days):
    data = getter(coin, days)
    
    close_price = data['close'][:period].to_list()
    low_price = min(data['low'][:period])
    high_price = max(data['high'][:period])
    k_percent = []
    k_price = []
    k_date = data['datetime'][period-1:]
    low = []
    high = []
    for i in range(period-1, days):
        close_price = data['close'].iloc[i]
        start_idx = max(0, i - period + 1)
        low_price = min(data['low'][start_idx:i+1])
        high_price = max(data['high'][start_idx:i+1])
        temp = ((close_price - low_price) / (high_price - low_price)) * 100
        k_percent.append(temp)
        low.append(low_price)
        high.append(close_price)
        k_price.append(close_price-low_price)
    d_percent = []

    d_date = k_date[2:]
    for i in range(2, len(k_percent)):
        d_percent.append(sum(k_percent[i-2:i+1]) / 3)
    k_, d_ = [], []
    d_price = k_price[:days-period-1]
    for i, j, l, h in zip(k_percent, k_price, low, high):
        i/=100
        temp = l+(j*i)
        k_.append(temp)
    for i, j, l in zip(d_percent, d_price, low):
        i/=100
        temp = l+(j*i)
        d_.append(temp)
    fig, ax = plt.subplots()
    ax.plot(data['datetime'], data['close'], color = "blue")
    ax.plot(k_date, k_, color = "green", label = "%K")
    ax.plot(d_date, d_, color = "red", label = "%D")
    plt.grid(True)
    plt.legend()
    plt.show()
    text = ""
    if k_percent[0] > 80 and k_percent[1] < k_percent[0]:
        text+="Overbought, recession time.\n"
    elif k_percent[0] < 20 and k_percent[1] > k_percent[0]:
        text+="Oversold, time to grow.\n"
    else:
        text+="Flat dominates, time to wait.\n"
    if k_percent[0] > d_percent[0] and k_percent[1] <= d_percent[1]:
        text += "Buy signal! %K crossed %D from bottom to top."
    elif k_percent[0] < d_percent[0] and k_percent[1] >= d_percent[1]:
        text += "Signal for sale! %K crossed %D from top to bottom."
    else:
        text += "Flat or no clear signal yet."
    text+="\n%K: "
    text+=str(k_percent[0])
    text+="\n%D: "
    text+=str(d_percent[0])
    return text
# resp = stochastic_oscillator("BTC-USDT", 3, 50)
import numpy as np

def test_Dickey_Fuller(price):
    x_t = price[:len(price)-1]
    x_delta = []
    for i, j in zip(price, price[1:]):
        x_delta.append(j-i)
    alpha = 0
    for i, j in zip(x_t, x_delta):
        alpha+=(i*j)/(i)**2
    dispersion = 0
    e = 0
    for i, j in zip(x_t, x_delta):
        e += (j-alpha*i)**2
    dispersion = e/(len(price)-2)
    se_zn = 0
    for i in x_t:
        se_zn+= i**2
    se = np.sqrt(dispersion/se_zn)
    t_statistic = alpha/se
    levels = [-3.75, -2.99, -2.64]
    if t_statistic < -3.75:
        return f"Stac! Very high!{t_statistic}", 0
    elif t_statistic < -2.99 and t_statistic > -3.75:
        return f"Stac! High!{t_statistic}", 0
    elif t_statistic < -2.64 and t_statistic > -2.99:
        return f"Stac! Medium!{t_statistic}", 0
    elif t_statistic < 0 and t_statistic > -2.64:
        return f"Stac! Low!{t_statistic}", 0
    else:
        return f"Not stac! {t_statistic}", 1

def test_engel_granger(price, price_1):
    x_av = sum(price)/len(price)
    y_av = sum(price_1)/len(price_1)
    covariation_chis = 0
    for i, j in zip(price, price_1):
        covariation_chis+=(i-x_av)*(j-y_av)
    covariation = covariation_chis/len(price)-1
    disp_sum = 0
    for i in price:
        disp_sum = (i-x_av)**2
    dispersion = disp_sum/len(price)-1
    b1 = covariation/dispersion
    b0 = y_av-b1*x_av
    e = []
    for i, j in zip(price, price_1):
        e.append(j-(b0+b1*i))
    return e

def cointegration(coin_1, coin_2, days): 
    data, data_1 = getter(coin_1, days), getter(coin_2, days)
    _, resp = test_Dickey_Fuller(data['close'])
    _, resp_1 = test_Dickey_Fuller(data_1['close'])
    if resp == 1 and resp_1 == 1:
        e = test_engel_granger(data['close'], data_1['close'])
        _, resp = test_Dickey_Fuller(e)
        if resp == 1 and resp_1 == 1:
            return "Rows cointegrated!"
        else:
            return "Rows not cointegrated!"
    else:
        return "Rows not stacionared and not cointegrated"
# resp = cointegration("BTC-USDT", "DOGE-USDT", 1500)
# print(resp)
def zig_zag(coin, days, depth, dev):
    data = getter(coin, days)
    price = list(data['close'])
    date = list(data['datetime'])
    date_array = [data['datetime'][0]]
    date_array_1 = [data['datetime'][0]]
    price = np.flip(price)
    count_d = 0
    check = []
    array_1 = [price[0]]
    count_perc = 0
    diff_perc_1 = 0
    count_not = 0
    date_array = [date[0]]
    for i, j in zip(price, date):
        count_d+=1
        check.append(i)
        if count_d >= depth:
            max_el = max(check)
            diff_perc = ((max_el*100)/array_1[count_perc])-100
            if diff_perc < dev:
                count_not+=1
            else:
                array_1.append(max_el)
                count_perc+=1
                check.clear()
                count_d = 0
                date_array.append(j)
                continue
            min_el = min(check)
            diff_perc_1 = ((min_el*100)/array_1[count_perc])-100
            if diff_perc_1 > -dev:
                count_not+=1
            else:
                array_1.append(min_el)
                count_perc+=1
                check.clear()
                count_d = 0
                date_array.append(j)
                continue
    return price, date, array_1, date_array
# _, _, arr, _ = zig_zag("BTC-USDT", 50, 3, 5)
# print(arr)

def fibonacci_levels(coin, days, depth, div):
    price, date_or, extremum, date = zig_zag(coin, days, depth, div)
    maximum = max(extremum)
    minimum = min(extremum)
    levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    up = []
    down = []
    length = len(extremum)/len(levels)
    if length < 1:
        length = 1
    date_ = []
    for i, j in zip(levels, date[::int(length)]):
        up.append(maximum-(maximum - minimum)*i)
        down.append(minimum+(maximum - minimum)*i)
        date_.append(j)
    levels_date = [date[0]]
    levels_date.append(date[-1])
    some_l = [levels[0]]
    some_l.append(levels[0])

    plt.plot(date_or, price, color = "blue", label = "Price")
    plt.plot(date_, up, color = "green", marker = 'o', label = "Up")
    plt.plot(date_, down, color = "red", marker = "o", label = "Down")
    plt.plot(date, extremum, color = "purple", label = "ZigZag extremums")
    plt.grid(True)
    plt.show()
    return extremum
def detect_trend(extremums):
    ups = 0
    downs = 0
    for i in range(1, len(extremums)):
        if extremums[i] > extremums[i-1]:
            ups += 1
        elif extremums[i] < extremums[i-1]:
            downs += 1
    if ups > downs:
        return "Now dominates uptrend."
    elif downs > ups:
        return "Now dominates downtrend."
    else:
        return "Now dominates flat trend."


def elliot_waves(coin, days, depth, div):
    ext = fibonacci_levels(coin, days, depth, div)
    response = detect_trend(ext)
    return response

# resp = elliot_waves("BTC-USDT", 50, 3, 5)
# print(resp)