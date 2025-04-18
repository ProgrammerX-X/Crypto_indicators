import requests
# from fastapi import FastAPI
# import threading
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
# app = FastAPI()
# @app.get("/getter")
def getter(coin, days):
    # https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}
    # url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
    # data = requests.get(url).json()
    # df = pd.DataFrame(data['prices'], columns=['datetime', 'price'])
    # df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    # return df\
    # days = days*1440
    
    # last_timestamp = data['data'][-1][0]
    if days < 101:
        url = f"https://www.okx.com/api/v5/market/history-mark-price-candles?instId={coin}&bar=1D&limit={days}"
        data = requests.get(url).json()
        df = pd.DataFrame(data['data'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df = df[['datetime', 'close', 'high', 'low']]
        df['datetime'] = pd.to_datetime(df['datetime'].astype(float), unit='ms')
        # df['close'] = df['close'].sorted
        df['close'] = pd.to_numeric(df['close'], errors="coerce")
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        # print((df['close'][0]).dtype)
    else:
        new_days = int(days/100)
        new_days_s = days%100
        new_dataframe = pd.DataFrame()
        url = f"https://www.okx.com/api/v5/market/history-mark-price-candles?instId={coin}&bar=1D&limit={days}"
        data = requests.get(url).json()
        df = pd.DataFrame(data['data'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df = df[['datetime', 'close', 'high', 'low']]
        df['datetime'] = pd.to_datetime(df['datetime'].astype(float), unit='ms')
        # df['close'] = df['close'].sorted
        df['close'] = pd.to_numeric(df['close'], errors="coerce")
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        new_dataframe = pd.concat([new_dataframe, df], axis=0)
        last_timestamp = data['data'][-1][0]
        # print(last_timestamp)
        for _ in range(1, new_days):
            url = f"https://www.okx.com/api/v5/market/history-mark-price-candles?instId={coin}&bar=1D&limit={days}&after={last_timestamp}"
            data = requests.get(url).json()
            # print(data)
            df = pd.DataFrame(data['data'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df = df[['datetime', 'close', 'high', 'low']]
            df['datetime'] = pd.to_datetime(df['datetime'].astype(float), unit='ms')
            # df['close'] = df['close'].sorted
            df['close'] = pd.to_numeric(df['close'], errors="coerce")
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            new_dataframe = pd.concat([new_dataframe, df], axis=0)
            last_timestamp = data['data'][-1][0]
            # print(last_timestamp)
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
        # new_dataframe.to_csv('file.csv', index=False)
        # print((new_dataframe['close'][0]).dtype)
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
        return (f"Тренд піде вгору, сигнал для покупки! SMA {s_day} = {smas}, SMA {b_day} = {smab}", smas, smab, diff)
    elif smas < smab:
        # value = False
        return (f"Тренд піде вниз, сигнал для продажу! SMA {s_day} = {smas}, SMA {b_day} = {smab}", smas, smab, diff)
    else:
        # value = None
        return (f"Тренд збалансований, час зачекати! SMA {s_day} = {smas}, SMA {b_day} = {smab}", smas, smab, diff)

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
    # print(date_b_)
    # print(len(price_b_arr))
    date_b_.append(date.iloc[-1])
    price_b_arr.append(sma(temp))
    
    price_b_arr = np.array(price_b_arr)
    date_b_ = np.array(date_b_).ravel()

    check, _, _, diff = checking_trend(coin, period_1, period_2)

    plt.plot(date, price, color = "blue", label = f"Ціна монети")
    plt.plot(date_s_, price_s_arr, color = "green", label = f"Сигнал на позитивний тренд")
    plt.plot(date_b_, price_b_arr, color = "red", label = f"Сигнал на негативний тренд")
    plt.plot(date, price, color = "white", alpha=0, label = f"{check}")
    plt.plot(date, price, color = "white", alpha=0, label = f"Різниця: {diff}")
    plt.grid(True)
    plt.legend()
    plt.show()
    
# trends_walk_sma('BTC-USDT', 2000, 20, 50)

def ema(price, days):
    if len(price) >= days:
        ema_n_1 = sma(price)
        k = 2/(days+1)
        p = price[-1]
        ema = (p*k)+(ema_n_1* (1-k))
        return ema
    else:
        return f"Недостатньо даних!"
# ema(coin, 5)

def checking_trend_ema(price, day_s, day_b):
    emas = ema(price, day_s)
    emab = ema(price, day_b)
    diff = emas-emab
    if emas > emab:
        return (f"Тренд піде вгору, сигнал для покупки! EMA {day_s} = {emas}, EMA {day_b} = {emab}", emas, emab, diff)
    elif emas < emab:
        return (f"Тренд піде вниз, сигнал для продажу! EMA {day_s} = {emas}, EMA {day_b} = {emab}", emas, emab, diff)
    else:
        return (f"Тренд збалансований, час зачекати! EMA {day_s} = {emas}, EMA {day_b} = {emab}", emas, emab, diff)
# trends_walk_('bitcoin', 20, 200)

def trend_walk_ema(coin, days, period_1, period_2):
    data = getter(coin, days)
    date = data['datetime']
    price = data['close']
    check, _, _, diff = checking_trend_ema(np.array(price), period_1, period_2)

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

    plt.plot(date, price, color = "blue", label = f"Ціна монети")
    plt.plot(date_s_, price_s_arr, color = "green", label = f"Сигнал на позитивний тренд")
    plt.plot(date_b_, price_b_arr, color = "red", label = f"Сигнал на негативний тренд")
    plt.plot(date, price, alpha=0, label = f"{check}")
    plt.plot(date, price, alpha=0, label = f"Різниця: {diff}")
    plt.grid(True)
    plt.legend()
    plt.show()

# trend_walk_ema('BTC-USDT', 300, 10, 30)

def rsi(coin, days):
    df = getter(coin, days)
    price = df['close']
    lesion = []
    profit = []
    i1 = price.iloc[0]
    if len(price) < 2:
        return "Недостатньо даних!"
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
                return f"Тренд піде перепроданий, сигнал для покупки! RSI - {rsi_}"
            if rsi_ > 70:
                return f"Тренд піде перекуплений, сигнал для продажу! RSI - {rsi_}"
            else:
                return f"Тренд збалансований, час зачекати! RSI - {rsi_}"
        else:
            return f"Масив *Збитки* або *Прибутки* пустий."

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
    chikou_span = price.iloc[26:].tolist()
    date_chikou = date.iloc[26:].tolist()

    fig, ax = plt.subplots(2, 1, figsize=(9, 5))
    ax[0].plot(date, price, color = "blue", label = "Ціна монети")
    ax[0].plot(date_in, tenkan, color = "green", label = "Tekan-sen")
    ax[0].plot(date_in_, kijun, color = "red", label = "Kijun-sen")
    ax[0].plot(date_chikou, chikou_span, color = "black", label = "Chikou-sen")
    ax[1].plot(date, price, color = "blue", label = "Ціна монети")
    ax[1].plot(date_in_, senkou_span_a, color = "aqua", label = "Senkou Span A")
    ax[1].plot(date_in_b, senkou_span_b, color = "aqua", label = "Senkou Span B")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()
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

# ichimoku_cloud("BTC-USDT", 300)


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
    plt.plot(date, price, color = "blue")
    plt.plot(ema_p_1D, ema_p_1, color = "green")
    plt.plot(ema_p_2D, ema_p_2, color = "red")
    plt.grid(True)
    plt.show()

# trend_walk_classic_ema("BTC-USDT", 5, 10, 120)

def macd(coin, days):
    if days >= 35:
        data = getter(coin, days)
        price = data['close']
        date = data['datetime']
        ema_12 = classic_ema(price, 12, days)
        ema_26 = classic_ema(price, 26, days)
        macd_line = []
        date_macd = []
        for i, j, d in zip(ema_12, ema_26, date.iloc[12:]):
            macd_line.append(i-j)
            date_macd.append(d)
        signal_date = date_macd[8:]
        signal_line = classic_ema(pd.Series(macd_line), 9, len(macd_line))
        histogram = []
        histogram_date = []
        for i, j, d in zip(macd_line, signal_line, date_macd):
            histogram.append(i-j)
            histogram_date.append(d)

        plt.plot(date_macd, macd_line, color = "green", label = "MACD line")
        plt.plot(signal_date, signal_line, color = "red", label = "Signal line")
        plt.plot(histogram_date, histogram, color = "black", label = "Histogram")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Занадто мало даних!")


# macd("BTC-USDT", 35)


def stochastic_oscillator(coin, period, days):
    data = getter(coin, days)
    
    close_price = data['close'][:period].to_list()
    low_price = min(data['low'][:period])
    high_price = max(data['high'][:period])
    
    k_percent = [((close_price[-1] - low_price) / (high_price - low_price)) * 100]

    k_date = data['datetime'][period-1:]

    for i in range(period, days):
        close_price = data['close'][i]
        low_price = min(data['low'][i-period+1:i+1])
        high_price = max(data['high'][i-period+1:i+1])
        temp = ((close_price - low_price) / (high_price - low_price)) * 100
        k_percent.append(temp)
    
    d_percent = [sma(k_percent[:3])]

    d_date = k_date[2:]

    for i in range(3, len(k_percent)):
        temp = k_percent[i-2:i+1]
        d_percent.append(sma(temp))

    plt.plot(k_date, k_percent, color="green", label="%K")
    plt.plot(d_date, d_percent, color="red", label="%D")
    plt.grid(True)
    plt.legend()
    plt.show()


# stochastic_oscillator("BTC-USDT", 20, 100)

# def fibbonaci_levels(price, date, min_, max_):
#     fib_levels = [] 
#     trend = []
#     for i, j in zip(min_, max_):
#         if i < 0  and j < 0:
#             trend.append(False)
#         else:
#             trend.append(True)
#     for i, j, k in zip(price, date, trend):
#         # x.append()
#         high = np.max(i)
#         low = np.min(i)
#         last_price = i[-1]
        
#         if high == low:
#             fib_levels.append(last_price)
#             continue
        
#         x = (high - last_price) / (high - low)
#         if k == False:
#             fib_levels.append(high+(high - low)*x)
#         else:
#             fib_levels.append(high-(high - low)*x)
#     # print(fib_levels)
#     return fib_levels
#         # fib_levels.append(np.min(i))
#     # print(price)



# Depth (глубина) = 2
# Бар считается экстремумом, если он выше (или ниже) 2 свечей слева и 2 свечей справа.

# Backstep = 3
# Между двумя соседними экстремумами должно быть не менее 3 баров.

# Threshold (порог) = 5%
# Разница в цене между экстремумами должна быть ≥ 5%.



# def zig_zag(coin, days, depth, backstep, threshold):
#     data = getter(coin, days)
#     price = data['close']
#     date = data['datetime']
#     depth = (depth*2)+1
#     depth_arg = depth/2
#     count = 0
#     temp_max = []
#     count_1 = 0
#     count_2 = 0
#     while True:
#         pass
    # for i in price:
    #     if count != depth:
    #         count+=1
    #         temp_max.append(i)
    #     else:
    #         count = 0
    #         if temp_max[depth_arg] > np.max(temp_max[depth_arg:] and temp_max[:depth_arg]):
    #             print(temp_max, temp_max[depth_arg])
    #         else:
    #             temp_max.clear()
    #         temp_max.clear()
        

# def elliott_waves(coin, days, depth, backstep, threshold):
#     zig_zag(coin, days, depth, backstep, threshold)
#     data = getter(coin, days)
#     price = data['close']
#     date = data['datetime']

#     plt.plot(date, price, color = "blue")
#     plt.grid(True)
#     plt.tight_layout()
#     # plt.show()

# elliott_waves("BTC-USDT", 10, 2, 3, 0.05)

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
    
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p}")

    if p < p_value:
        return True
    else:
        return False

def test_engel_granger(coin_1, coin_2, days, p_value):
    data, data_1 = getter(coin_1, days), getter(coin_2, days)
    # print(data_1)
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
    if u == False:
        print("The series are not cointegrated.")
    else:
        final = test_Dickey_Fuller(u, p_value)
    
        if final:
            print("All ok! The series are cointegrated.")
        else:
            print("Not ok... The series are not cointegrated.")
# cointegration("BTC-USDT", "TRX-USDT", 10, 0.05)