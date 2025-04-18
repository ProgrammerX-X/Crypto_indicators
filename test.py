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
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Пример данных
data = [1.5, 2.5, 3.5, 2.0, 4.0, 3.0, 3.5, 4.5, 5.0, 4.5]
# Преобразуем в Series (можно использовать и массив)
ts = pd.Series(data)

# Применяем ADF-тест
result = adfuller(ts)

# Печатаем результат
print("ADF-статистика:", result[0])
print("p-значение:", result[1])
print("Число лагов:", result[2])
print("Число наблюдений:", result[3])
print("Критические значения:", result[4])
print("Соответствующие p-значения для критических значений:", result[5])
