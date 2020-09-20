import numpy as np
import pandas as pd
import json, codecs


with open('models.json') as f:
    file = json.load(f)


'''Вычислим ожидаемую доходность за один переход при выходе из i состояния
 и при выборе стратегии k
 :param prob {int[][]} - вероятность перехода из i состояния в j
 :param profitableness {int[][]} - доходность 
 :param num_state_out - номер состояния, из которого выходим '''


def calc_income(num_state_out, num_strategy):
   return np.sum(matrix_transition_probability[num_strategy][num_state_out] * matrix_profitableness[num_strategy][num_state_out])


count_state_system = int(file['states']) #2  # У П
count_strategy_system = int(file['strategy'])  #2  L M N
count_step_modeling = int(file['modelCount'])

matrix_transition_probability = np.array(file['transition_probability'], dtype=float)
matrix_profitableness = np.array(file['profitableness'], dtype=float)


matrix_waiting_profit = np.zeros(shape=(count_state_system, count_strategy_system))

matrix_full_waitng_profitableness = np.zeros(shape=(count_step_modeling+1, count_state_system))
matrix_select_num_strategy_by_step = np.zeros(shape=(count_step_modeling+1, count_state_system))


# Заполняем матрицу ожидаемого дохода
for i in range(count_state_system):
    for j in range(count_strategy_system):
        matrix_waiting_profit[i, j] = calc_income(i, j)



# Шаг 2. Вычисляем полный ожидаемый доход
for i in range(count_step_modeling):
    for j in range(count_state_system):
        q = matrix_waiting_profit[j]
        a = matrix_transition_probability[0:count_strategy_system, j]
        b = np.reshape(np.asarray(matrix_full_waitng_profitableness[i]), (count_state_system, 1))
        k = a @ b
        matrix_full_waitng_profitableness[i + 1][j] = max(q + np.reshape(k.T, (count_strategy_system,)))
        matrix_select_num_strategy_by_step[i+1][j] = np.argmax(q + np.reshape(k.T, (count_strategy_system,)))


# вывод результатов моделирования для каждого шага: итоговая доходность и оптимальная стратегия;
print(matrix_select_num_strategy_by_step)
print(matrix_full_waitng_profitableness)


to_json = {'Strategy': matrix_select_num_strategy_by_step.tolist(), 'Profitableness': matrix_full_waitng_profitableness.tolist()}
json.dump(to_json,
          codecs.open('test.json', 'w', encoding='utf-8'),
          separators=(',', ':'), sort_keys=True, indent=4)


'''

− вывод графа состояний;
− сохранение и загрузка данных в файл.

'''
