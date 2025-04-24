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
        # value = True
        return (f"The trend will go up, a signal to buy! SMA {s_day} = {smas}, SMA {b_day} = {smab}", smas, smab, diff)
    elif smas < smab:
        # value = False
        return (f"The trend will go down, a signal to sell! SMA {s_day} = {smas}, SMA {b_day} = {smab}", smas, smab, diff)
    else:
        # value = None
        return (f"The trend is balanced, time to wait! SMA {s_day} = {smas}, SMA {b_day} = {smab}", smas, smab, diff)

# -------------------------

# response = checking_trend("BTC-USDT", 7, 25)
# print(response)
# def making_array(coin, days, diapason):
#     df = getter(coin, diapason)
#     # print(df.head(10))
#     df['datetime'] = pd.to_datetime(df['datetime'])
#     df.set_index('datetime', inplace=True)
#     result = df.resample(f"{days}D").apply(sma)
#     return result

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
    # print(price_s_arr)
    # print(date_s_)
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
    
# a = trends_walk_sma('BTC-USDT', 50, 10, 20)
# print(a)

def ema(price, days):
    if len(price) >= days:
        ema_n_1 = sma(price)
        k = 2/(days+1)
        p = price[-1]
        ema = (p*k)+(ema_n_1* (1-k))
        return ema
    else:
        return f"Not enough data!"
# ema(coin, 5)

def checking_trend_rolling_ema(emas, emab, day_s, day_b):
    diff = emas-emab
    if emas > emab:
        return (f"The trend will go up, a signal to buy! EMA {day_s} = {emas}, EMA {day_b} = {emab}", emas, emab, diff)
    elif emas < emab:
        return (f"The trend will go down, a signal to sell! EMA {day_s} = {emas}, EMA {day_b} = {emab}", emas, emab, diff)
    else:
        return (f"The trend is balanced, time to wait! EMA {day_s} = {emas}, EMA {day_b} = {emab}", emas, emab, diff)
# trends_walk_('bitcoin', 20, 200)

def trend_walk_rolling_ema(coin, days, period_1, period_2):
    if period_1 > 0 and period_2 > 0:
        data = getter(coin, days)
        date = data['datetime']
        price = data['close']

        price_s, date_s = period_trend(coin, days, period_1)
        price_b, date_b = period_trend(coin, days, period_2)
        price_s_arr = []
        date_s_ = []

        temp_p = price.iloc[-1]
        temp_b = date.iloc[-1]
        for i, j in zip(price_s, date_s):
            price_s_arr.append(ema(i, period_1))
            date_s_.append(j[0])
        price_s_arr.append(temp_p)
        date_s_.append(temp_b)
        
        price_b_arr = []
        date_b_ = []
        for i, j in zip(price_b, date_b):
            price_b_arr.append(ema(i, period_2))
            date_b_.append(j[0])
        price_b_arr.append(temp_p)
        date_b_.append(temp_b)

        check, _, _, diff = checking_trend_rolling_ema(price_s_arr[0], price_b_arr[0], period_1, period_2)
        plt.plot(date, price, color = "blue", label = f"The price of the coin")
        plt.plot(date_s_, price_s_arr, color = "green", label = f"A signal for a positive trend")
        plt.plot(date_b_, price_b_arr, color = "red", label = f"A signal for a negative trend")
        plt.plot(date, price, alpha=0, label = f"{check}")
        plt.plot(date, price, alpha=0, label = f"Difference: {diff}")
        plt.grid(True)
        plt.legend()
        plt.show()
        if period_1 < 10 or period_2 < 10:
            return (f"Be careful, the forecast is short-term momentum. \n!Check the trend on medium timeframes!", check, diff)
        elif period_1 < 26 or period_2 < 26:
            return("A real trend. For long-term trends, use longer timeframes(10-25).", check, diff)
        else:
            return("Long-term trend.", check, diff)
    else:
        return("Periods must be positive numbers.", 0, 0)

# trend_walk_rolling_ema('BTC-USDT', 100, 5, 20)

def rsi(coin, days):
    df = getter(coin, days)
    price = df['close']
    lesion = []
    profit = []
    i1 = price.iloc[0]
    if len(price) < 2:
        return ("Not enough data!", 0)
    else:
        for i in price.iloc[1:]:
            diff = i-i1
            if diff < 0:
                lesion.append(diff)
            else:
                profit.append(diff)
            i1 = i
        if len(profit) != 0 and len(lesion) != 0:
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
# rsi_sma("BTC-USDT", 20, 2, 5)
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
    # Tenkan-sen
    if days >= 9:
        tenkan, date_in = ichimoku_line(minimum, maximum, date, 9)
    else:
        tenkan, date_in = [], []

    # Kijun-sen
    if days >= 26:
        kijun, date_in_ = ichimoku_line(minimum, maximum, date, 26)
    else:
        kijun, date_in_ = [], []

    # Senkou Span A
    senkou_span_a = [(t + k) / 2 for t, k in zip(tenkan, kijun)] if tenkan and kijun else []

    # Senkou Span B
    if days >= 52:
        senkou_span_b, date_in_b = ichimoku_line(minimum, maximum, date, 52)
    else:
        senkou_span_b, date_in_b = [], []

    # Chikou Span
    chikou_span = price.shift(-26)
    # print(chikou_span)
    date_chikou = date.shift(-26)
    min_span = min(senkou_span_a+senkou_span_b)
    # print(min_span)
    max_span = max(senkou_span_a+senkou_span_b)
    # print(max_span)
    # print(price[0])
    fig, ax = plt.subplots(2, 1, figsize=(9, 5))
    ax[0].plot(date, price, color = "blue", label = "The price of the coin")
    ax[0].plot(date_in, tenkan, color = "green", label = "Tekan-sen")
    ax[0].plot(date_in_, kijun, color = "red", label = "Kijun-sen")
    ax[0].plot(date_chikou, chikou_span, color = "aqua", label = "Chikou-sen")
    ax[1].plot(date, price, color = "blue", label = "The price of the coin")
    ax[1].plot(date_in_, senkou_span_a, color = "purple", label = "Senkou Span A")
    ax[1].plot(date_in_b, senkou_span_b, color = "black", label = "Senkou Span B")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    # print("Теперішня ситуація:")
    text = "Current situation:\n"
    if price.iloc[0] > min_span and price.iloc[0] > max_span:
        text+="The trend is bullish.\n"
    elif price.iloc[0] < min_span and price.iloc[0] < max_span:
        text+="The trend is bearish.\n"
    else:
        text+="Now is the time of the flute.\n"
    # print("Сигнали на вхід в ринок:")
    text += "Market entry signals:\n"
    if tenkan[0] > kijun[0]:
        if tenkan[0] > max_span:
            text+="A strong buy signal.\n"
        elif tenkan[0] < max_span and tenkan[0] > min_span:
            text+='Neutral signal. It is better to wait.\n'
        else:
            text+='Weak signal. It is better to wait.\n'
    # print("Куди ринок піде:")
    text += "Where the market will go: Where the market will go:\n"
    if chikou_span.iloc[0] > price.iloc[0]:
        text+='The trend will go up.'
    else:
        text+='The trend will go down.'
    return text
# ichimoku_cloud("BTC-USDT", 53)
# tekan-sen
# Что показывает: Эта линия показывает краткосрочный тренд.

# Когда используется:
# Если Tenkan-sen выше Kijun-sen — это сигнал к покупке (бычий тренд).
# Если Tenkan-sen ниже Kijun-sen — это сигнал к продаже (медвежий тренд)
# Когда линии Tenkan и Kijun пересекаются, это может быть сигналом на вход в рынок.

# -----------------------------------------------
# kijun-sen
# Что показывает: Эта линия служит ориентиром для долгосрочного тренда. 
# Она может также использоваться как уровень поддержки или сопротивления.

# Когда используется:
# Если Tenkan-сен выше Kijun-сен, это сигнал на покупку.
# Если Tenkan-сен ниже Kijun-сен, это сигнал на продажу.

# Если цена пересекает Kijun-сен (снизу вверх — сигнал на покупку, 
# сверху вниз — сигнал на продажу), это может быть хорошим сигналом для входа в рынок.

# Senkou Span A
# Что показывает: Это линия, которая образует верхнюю границу облака Ичимоку.

# Когда используется:
# Если цена выше Senkou Span A, это сигнализирует о бычьем тренде.
# Если цена ниже Senkou Span A, это сигнализирует о медвежьем тренде.

# Senkou Span B
# Что показывает: Это нижняя граница облака.

# Когда используется:
# Если цена выше облака (между Senkou Span A и Senkou Span B), это сигнализирует о бычьем тренде.
# Если цена ниже облака, это сигнализирует о медвежьем тренде.
# Если цена находится внутри облака, это может быть сигналом для бокового тренда или неопределённости.

# Senkou Span B
# Что показывает: Эта линия отображает текущую цену по отношению к более ранним периодам.

# Когда используется:
# Если Chikou Span выше ценового графика, это сигнализирует о бычьем тренде.
# Если Chikou Span ниже ценового графика, это сигнализирует о медвежьем тренде.
# Если Chikou Span пересекает цену снизу вверх, это может быть сигналом на покупку.
# Если Chikou Span пересекает цену сверху вниз, это может быть сигналом на продажу.

# Если цена находится выше облака, это сигнализирует о бычьем тренде.

# Если цена находится ниже облака, это сигнализирует о медвежьем тренде.

# Если цена находится внутри облака, это может означать боковой тренд, и 
# не стоит делать торговых решений без дальнейшего анализа.


# MACD (Moving Average Convergence Divergence)

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
    ema_p_1D = date.iloc[period_1-1:].reset_index(drop=True)
    ema_p_2 = classic_ema(price, period_2, days)
    ema_p_2D = date.iloc[period_2-1:].reset_index(drop=True)
    # print(ema_p_1D)
    # print(len(ema_p_1), len(ema_p_1D), len(ema_p_2), len(ema_p_2D))
    plt.plot(date, price, color = "blue", label = "The price of the coin")
    plt.plot(ema_p_1D, ema_p_1, color = "green", label = "A signal for a positive trend")
    plt.plot(ema_p_2D, ema_p_2, color = "red", label = "A signal for a negative trend")
    plt.grid(True)
    plt.legend()
    plt.show()

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
    from math import isclose
    if days >= 35:
        data = getter(coin, days)
        price = data['close']
        date = data['datetime']
        ema_12 = classic_ema(price, 12, days)
        ema_26 = classic_ema(price, 26, days)
        macd_line = []
        date_macd = []
        # print(ema_12[0], ema_12[1], ema_26[0], ema_26[1])
        # print(int(ema_12[1]), int(ema_26[1]))
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
        signal_date = date_macd[8:]
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


# macd("BTC-USDT", 50)


def stochastic_oscillator(coin, period, days):
    data = getter(coin, days)
    
    close_price = data['close'][:period].to_list()
    low_price = min(data['low'][:period])
    high_price = max(data['high'][:period])
    
    # k_percent = [((close_price[-1] - low_price) / (high_price - low_price)) * 100]
    k_percent = []
    k_price = []
    k_date = data['datetime'][period-1:]
    low = []
    high = []
    for i in range(period-1, days):
        close_price = data['close'].iloc[i]
        low_price = min(data['low'][i-period+1:i+1])
        high_price = max(data['high'][i-period+1:i+1])
        temp = ((close_price - low_price) / (high_price - low_price)) * 100
        k_percent.append(temp)
        low.append(low_price)
        high.append(close_price)
        k_price.append(close_price-low_price)
    
    # d_percent = [sma(k_percent[:3])]
    d_percent = []

    d_date = k_date[2:]
    # print(data['close'])
    for i in range(2, len(k_percent)):
        temp = k_percent[i-2:i+1]
        d_percent.append(sma(temp))
    k_, d_ = [], []
    d_price = k_price[:days-period-1]
    for i, j, l, h in zip(k_percent, k_price, low, high):
        # 1000*80 = 80
        i/=100
        temp = l+(j*i)
        k_.append(temp)
        # print(f"{l}+{j}*{i} = {l+(j*i)}, h = {h}")
    for i, j, l in zip(d_percent, d_price, low):
        i/=100
        temp = l+(j*i)
        d_.append(temp)
        # print(f"{l}+{j}*{i} = {l+(j*i)}")
    # print(k_)
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
    # print(k_percent[0], d_percent[0])
    if k_percent[0] > d_percent[0] and k_percent[1] <= d_percent[1]:
        text += "Buy signal! %K crossed %D from bottom to top."
    elif k_percent[0] < d_percent[0] and k_percent[1] >= d_percent[1]:
        text += "Signal for sale! %K crossed %D from top to bottom."
    else:
        text += "Flat or no clear signal yet."
    return text

# stochastic_oscillator("BTC-USDT", 14, 200)

# Depth (глубина) = 2
# Бар считается экстремумом, если он выше (или ниже) 2 свечей слева и 2 свечей справа.

# Backstep = 3
# Между двумя соседними экстремумами должно быть не менее 3 баров.

# Threshold (порог) = 5%
# Разница в цене между экстремумами должна быть ≥ 5%.

        
import numpy as np
from scipy.stats import norm

def test_Dickey_Fuller(price, p_value):
    alpha = 1
    betta = 1
    hamma = 1
    delta = 1
    t = 2
    result_X = []
    y_delta = []
    y = []
    
    for i, k in zip(price[:-1], price[1:]):
        if len(y_delta) == 2:
            y_delta.pop(0)
        
        if len(y_delta) != 0:
            temp = [alpha, betta * t, hamma * i, delta * y_delta[0]]
            result_X.append(temp)
            y_delta.append(k - i)
            y.append(y_delta[-1])
        else:
            y_delta.append(k - i)
        
        t += 1
    
    result_X = np.array(result_X)
    x_t = result_X.T
    subt_XT_X = np.dot(x_t, result_X)
    subt_XT_Y = np.dot(x_t, y)
    
    x_inv = np.linalg.inv(subt_XT_X)
    res = np.dot(x_inv, subt_XT_Y)
    
    y_hat = np.dot(result_X, res)
    e = np.array(y) - y_hat
    e_sum_sq = np.sum(e ** 2)
    
    var_betta = e_sum_sq / (len(price) - len(result_X[0]))
    st_err_betta = np.sqrt(var_betta * np.linalg.inv(subt_XT_X).diagonal())
    t_stat = res[2] / st_err_betta[2]
    
    p = 2 * (1 - norm.cdf(abs(t_stat)))
    
    # print(f"t-statistic: {t_stat}")
    # print(f"p-value: {p}")

    if p < p_value:
        return True
    else:
        return False

def test_engel_granger(coin_1, coin_2, days, p_value):
    data, data_1 = getter(coin_1, days), getter(coin_2, days)
    data.dropna()
    data_1.dropna()
    resp_1 = test_Dickey_Fuller(data['close'], p_value)
    resp_2 = test_Dickey_Fuller(data_1['close'], p_value)
    
    if resp_1 and resp_2:
        xy = 0
        x = 0
        y = 0
        x2_in_sum = 0
        
        for i, j in zip(data['close'], data_1['close']):
            xy += i * j
            x += i
            y += j
            x2_in_sum += i ** 2
        
        x2 = x ** 2
        betta = (len(data['close']) * xy - x * y) / (len(data['close']) * x2_in_sum - x2)
        alpha = (y - betta * x) / len(data['close'])
        
        u = []
        for i, j in zip(data['close'], data_1['close']):
            y_in = alpha + betta * i
            u.append(j - y_in)
        
        return u
    else:
        return False

def cointegration(coin_1, coin_2, days, p_value):
    u = test_engel_granger(coin_1, coin_2, days, p_value)
    text = ""
    if u == False:
        text+= "The series are not cointegrated."
    else:
        final = test_Dickey_Fuller(u, p_value)
    
        if final:
            text +="All ok! The series are cointegrated."
        else:
            text+="Not ok... The series are not cointegrated."
    return text
# response = cointegration("BTC-USDT", "TRX-USDT", 10, 0.05)
# print(response)
def zig_zag(coin, days, depth):
    data = getter(coin, days)
    price = list(data['close'])
    extremum = list([price[0]])
    date = list(data['datetime'])
    date_array = [data['datetime'][0]]
    count = 0
    for i, k in zip(price[1:], date[1:]):
        temp_depth = (i - extremum[count])/extremum[count]
        if abs(temp_depth) >= depth:
            extremum.append(i)
            count+=1
            date_array.append(k)
    return price, date, extremum, date_array

def fibonacci_levels(coin, days, p):
    price, date_or, extremum, date = zig_zag(coin, days, p)
    maximum = max(extremum)
    minimum = min(extremum)
    levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    up = []
    down = []
    length = len(extremum)/len(levels)
    date_ = []
    for i, j in zip(levels, date[::int(length)]):
        up.append(maximum-(maximum - minimum)*i)
        down.append(minimum+(maximum - minimum)*i)
        date_.append(j)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(date_or, price, color = "blue")
    ax[1].plot(date_, up, color = "green", marker = 'o')
    ax[1].plot(date_, down, color = "red", marker = "o")
    ax[0].plot(date, extremum, color = "purple")
    ax[0].grid(True)
    ax[1].grid(True)
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


def elliot_waves(coin, days, p):
    ext = fibonacci_levels(coin, days, p)
    response = detect_trend(ext)
    return response

# elliot_waves("BTC-USDT", 50, 0.03)