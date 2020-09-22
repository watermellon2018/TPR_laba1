import numpy as np
import pandas as pd
import json, codecs


class Controller:
    count_state_system = 0
    count_strategy_system = 0
    count_step_modeling = 0

    def readFile(self):
        with open('models.json') as f:
            self.file = json.load(f)
            self.count_state_system = int(self.file['states'])  # 2  # У П
            self.count_strategy_system = int(self.file['strategy'])  # 2  L M N
            self.count_step_modeling = int(self.file['modelCount'])
            self.matrix_transition_probability = np.array(self.file['transition_probability'], dtype=float)
            self.matrix_profitableness = np.array(self.file['profitableness'], dtype=float)

            self.matrix_waiting_profit = np.zeros(shape=(self.count_state_system, self.count_strategy_system))

            self.matrix_full_waitng_profitableness = np.zeros(shape=(self.count_step_modeling + 1, self.count_state_system))
            self.matrix_select_num_strategy_by_step = np.zeros(shape=(self.count_step_modeling + 1, self.count_state_system))


    def setParam(self, count_state, count_stratagy, count_step):
        self.count_state_system = count_state
        self.count_strategy_system = count_stratagy
        self.count_step_modeling = count_step


    '''Вычислим ожидаемую доходность за один переход при выходе из i состояния
    и при выборе стратегии k
    :param prob {int[][]} - вероятность перехода из i состояния в j
    :param profitableness {int[][]} - доходность 
    :param num_state_out - номер состояния, из которого выходим '''


    def calc_income(self, num_state_out, num_strategy):
        return np.sum(self.matrix_transition_probability[num_strategy][num_state_out]
                      * self.matrix_profitableness[num_strategy][num_state_out])


# count_state_system = int(file['states']) #2  # У П
# count_strategy_system = int(file['strategy'])  #2  L M N
# count_step_modeling = int(file['modelCount'])


    #
    # self.matrix_transition_probability = np.array(self.file['transition_probability'], dtype=float)
    # self.matrix_profitableness = np.array(self.file['profitableness'], dtype=float)

#
# matrix_waiting_profit = np.zeros(shape=(count_state_system, count_strategy_system))
#
# matrix_full_waitng_profitableness = np.zeros(shape=(count_step_modeling+1, count_state_system))
# matrix_select_num_strategy_by_step = np.zeros(shape=(count_step_modeling+1, count_state_system))


    # Заполняем матрицу ожидаемого дохода
    def fill_matrix_waiting_profit(self):
        for i in range(self.count_state_system):
            for j in range(self.count_strategy_system):
                self.matrix_waiting_profit[i, j] = self.calc_income(i, j)



    # Шаг 2. Вычисляем полный ожидаемый доход
    def calc_full_matrix_profit(self):
        for i in range(self.count_step_modeling):
            for j in range(self.count_state_system):
                q = self.matrix_waiting_profit[j]
                a = self.matrix_transition_probability[0:self.count_strategy_system, j]
                b = np.reshape(np.asarray(self.matrix_full_waitng_profitableness[i]), (self.count_state_system, 1))
                k = a @ b
                res = q + np.reshape(k.T, (self.count_strategy_system,))
                self.matrix_full_waitng_profitableness[i + 1][j] = max(res)
                self.matrix_select_num_strategy_by_step[i+1][j] = np.argmax(res)

    #  само решение
    def soluction(self):
        self.fill_matrix_waiting_profit()
        self.calc_full_matrix_profit()


    # вывод результатов моделирования для каждого шага: итоговая доходность и оптимальная стратегия;
    ''' Столбики - этап моделирования, ряды:
    для матрицы итоговой годовности - итоговая готовность для соответсвующего состояния
    для матрицы стратегии - какую лучше стратегию взять для соответсвующего состояния
    P.S. Таблица как в пособии'''

    def output(self):
        self.matrix_select_num_strategy_by_step = self.matrix_select_num_strategy_by_step.T
        self.matrix_full_waitng_profitableness = self.matrix_full_waitng_profitableness.T

        increment_matrix_for_strategy = np.ones(shape=self.matrix_select_num_strategy_by_step.shape)
        increment_matrix_for_strategy[:, 0] = 0
        print(self.matrix_select_num_strategy_by_step + increment_matrix_for_strategy)
        print(self.matrix_full_waitng_profitableness)


    '''Запись результатов в файл '''
    def write_into_file(self):
        to_json = {'Strategy': matrix_select_num_strategy_by_step.tolist(),
                   'Profitableness': matrix_full_waitng_profitableness.tolist()}
        json.dump(to_json,
                  codecs.open('test.json', 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True, indent=4)


c = Controller()
c.readFile()
c.soluction()
c.output()