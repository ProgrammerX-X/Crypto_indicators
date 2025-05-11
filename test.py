# import matplotlib.pyplot as plt

# # Дані
# x = [1, 2, 3, 4, 5]
# y1 = [10, 20, 25, 30, 40]
# y2 = [1, 1.5, 1.7, 2, 2.2]

# fig, ax1 = plt.subplots()

# # Перша вісь (ліва)
# ax1.plot(x, y1, 'g-')
# ax1.set_ylabel('Дані 1', color='g')

# # Друга вісь (права)
# ax2 = ax1.twinx()  # Створює другу вісь по Y
# ax2.set_ylabel('Підписи до Даних 2', color='b')
# ax2.set_yticks(y2)
# ax2.set_yticklabels(['A', 'B', 'C', 'D', 'E'])  # Просто надписи

# plt.show()


# import numpy as np

# def test_Dickey_Fuller(price):
#     alpha = 1  # Константа
#     betta = 1  # Коэффициент для t (временной индекс)
#     hamma = 1  # Коэффициент для значений ряда (например, lag)
#     delta = 1  # Коэффициент для разности
#     t = 2  # Начальное значение времени
#     result_X = []  # Матрица независимых переменных (регрессоры)
#     y_delta = []  # Список для изменений (первых разностей)
#     y = []  # Массив для зависимой переменной (цели)
#     e = []  # Остатки
#     y_delta_ = []  # Заполняется для вычисления изменений

#     # Строим X и y
#     for i, k in zip(price[0:], price[1:]):
#         if len(y_delta) == 2:
#             y_delta.pop(0)
#         if len(y_delta) != 0:
#             temp = [alpha, betta * t, hamma * i, delta * y_delta[0]]  # Модель
#             result_X.append(temp)
#             y_delta.append(k - i)  # Сохраняем изменения
#             y.append(y_delta[-1])  # Зависимая переменная — это разности
#         else:
#             y_delta.append(k - i)
#         y_delta_.append(k - i)  # Для других вычислений
#         t += 1

#     # Преобразуем result_X в NumPy массив и транспонируем
#     x_t = np.array(result_X).T
#     # Подсчитываем X^T * X
#     subt_XT_X = np.dot(x_t, result_X)
#     # Подсчитываем X^T * y
#     subt_XT_Y = np.dot(x_t, y)
#     # Находим обратную матрицу (X^T * X)^(-1)
#     x_inv = np.linalg.inv(subt_XT_X)
#     # Оценки коэффициентов (beta)
#     res = np.dot(x_inv, subt_XT_Y)

#     # Выводим коэффициенты (beta)
#     print("Оценки коэффициентов:", res)

#     # Теперь вычисляем остатки
#     y_hat = np.dot(result_X, res)  # Предсказания
#     e = np.array(y) - y_hat  # Остатки: фактические значения минус предсказания

#     # Выводим остатки для анализа
#     print("Остатки:", e)

#     # Оценка статистической значимости
#     # Статистика t для коэффициента beta:
#     residual_sum_squares = np.sum(e**2)
#     variance_beta = residual_sum_squares / (len(price) - len(result_X[0]))
#     std_err_beta = np.sqrt(variance_beta * np.linalg.inv(subt_XT_X).diagonal())  # Стандартная ошибка для коэффициентов

#     t_stat = res[2] / std_err_beta[2]  # Статистика t для beta_2 (третий коэффициент)
#     print(f"t-статистика для beta_2: {t_stat}")

#     # Определение уровня значимости через p-значение
#     p_value = 2 * (1 - norm.cdf(abs(t_stat)))  # Для нормального распределения
#     print(f"p-значение: {p_value}")

#     # Проверка на стационарность (пороговые значения для p-value)
#     if p_value < 0.05:  # Если p-value меньше 0.05, то можем отклонить гипотезу о наличии единичного корня
#         return True  # Ряд коинтегрирован, стационарен
#     else:
#         return False  # Ряд не коинтегрирован, не стационарен
# import requests
# import pandas as pd
# def getter(coin, days):
#     if days < 101:
#         # mark-price
#         url = f"https://www.okx.com/api/v5/market/history-mark-price-candles?instId={coin}&bar=1D&limit={days}"
#         data = requests.get(url).json()
#         df = pd.DataFrame(data['data'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
#         df = df[['datetime', 'close', 'high', 'low']]
#         df['datetime'] = pd.to_datetime(df['datetime'].astype(float), unit='ms')
#         df['close'] = pd.to_numeric(df['close'], errors="coerce")
#         df['low'] = pd.to_numeric(df['low'], errors='coerce')
#         df['high'] = pd.to_numeric(df['high'], errors='coerce')
#         print(df['datetime'])
# getter("BTC-USDT", 5)









array = [1, 2, 3, 4,9, 10, 11, 12, 5 ,6 ,-1 ,8 , 13, 14 ,15 ,16 ,17, 18]
depth = 3
count_d = 0
check = []
array_1 = [array[0]]
count_perc = 0
dev = 10
diff_perc_1 = 0
count_perc_1 = 0
count_not = 0
for i in array:
    count_d+=1
    check.append(i)
    if count_d >= depth:
        max_el = max(check)
        diff_perc = ((max_el*100)/array_1[count_perc])-100
        print(f"{max_el}*100/{array_1[count_perc]} = {diff_perc}")
        if diff_perc < dev:
            count_not+=1
        else:
            print(diff_perc)
            array_1.append(max_el)
            count_perc+=1
            check.clear()
            count_d = 0
            continue
        min_el = min(check)
        diff_perc_1 = ((min_el*100)/array_1[count_perc])-100
        print(f"{min_el}*100/{array_1[count_perc]} = {diff_perc_1}")
        if diff_perc_1 > -dev:
            count_not+=1
        else:
            array_1.append(min_el)
            count_perc+=1
            check.clear()
            count_d = 0
            continue
print(array)
print(array_1)
# print(array_2)
        # if diff_perc < 