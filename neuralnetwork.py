from math import exp
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def sigmoid(x):
    '''
        Вычисление сигмоида по формуле 1/(1+exp(-x))
        In: x - число
        Out: результат операции - число
    '''
    return 1/(1+exp(-x))

def sigmoidDerivative(x):
    '''
        Вычисление производной сигмоида по формуле x * (1 - x)
        In: x - число
        Out: результат операции - число
    '''
    return x * (1 - x)

def dot(x, y):
    '''
        Вычисление поэлементного произведения векторов с последующим сложением
        In: x, y - вектор-строки одинаковой размерности (1xN)
        Out: результат операции - число
        
    '''
    res = 0
    for i, j in zip(x, y):
        res += i*j
    return res

def mult(x, layer):
    '''
        Перемножение входного вектора сигналов на слой нейронов
        In: 
            x - входной вектор-сигнал размерности 1xN
            layer - веса для одного слоя нейронов. Матрица рамерностью MxN, где M - количество нейронов в слое
        Out:
            результат операции - вектор 1xM
    '''
    res = [dot(x, inputs) for inputs in layer]
    return res

def mult_sigm(x, layer):
    '''
        Перемножение входного вектора сигналов на слой нейронов с применением функци сигмоида
        In: 
            x - входной вектор-сигнал размерности 1xN
            layer - веса для одного слоя нейронов. Матрица рамерностью MxN, где M - количество нейронов в слое
        Out:
            результат операции - вектор 1xM
    '''
    res = [sigmoid(out) for out in mult(x, layer)]
    return res

def feed_forward(x, nn):
    '''
        Перемножение входного вектора на нейросеть
        In:
            x - входной вектор-сигнал размерности 1xN
            nn - нейросеть, состоящая из списка слоев
        Out:
            результат операции - матрица с послойными результатами перемножения
    '''
    outs = []
    for layer in nn:
        x = mult(x, layer)
        outs.append(x)
    return outs

def feed_forward_sigm(signal, nn, bias = False):
    '''
        Перемножение входного вектора на нейросеть с применением функции сигмоида
        In:
            x - входной вектор-сигнал размерности 1xN
            nn - нейросеть, состоящая из списка слоев
			bias - признак наличия смещения (у каждого нейрона)
        Out:
            результат операции - матрица с послойными результатами перемножения
    '''
    outs = []
    x = [i for i in signal]
    for layer in nn:
        #Если у нас есть входы смещений, то вектор сигнала должен быть расширен на 1 элемент, равный 1
        x = mult_sigm(x + [1] if bias else x, layer)
        outs.append(x)
    return outs

class NeuralNetwork:	
	
    def __init__(self, nn, bias_enable = False):
        '''
			Инициализация структуры сети
			In:
				self - указатель на экземпляр объекта
				nn - матрица, описывающая структуру сети
			Out:
				None
		'''
        self.nn = nn
        self.bias_enable = bias_enable
        self.outs = []               #выходы нейронов после распространения сигнала
        self.delta0 = []             #ошибка выходного слоя нейронов
        self.deltas = []             #ошибки на выходах нейронов
		
    def add_bias(self):
        '''
			Добавление смещения
			In:
				self - указатель на экземпляр объекта
			Out:
				self.nn со смещением
        '''
        tmp_nn = []
        for layer in self.nn:
            layer = [inputs + [1] for inputs in layer]
            tmp_nn.append(layer)
        self.nn = tmp_nn
        self.bias_enable = True
        
    def del_bias(self):
        '''
            Удаление смещения
			In:
				self - указатель на экземпляр объекта
			Out:
				self.nn без смещением
        '''
        if not self.bias_enable:
            return
        tmp_nn = []
        for layer in self.nn:
            layer = [inputs[:-1] for inputs in layer]
            tmp_nn.append(layer)
        self.nn = tmp_nn
        self.bias_enable = False

    def get_bias_enable(self):
        '''
            Получение признака наличия смещения
            In:
                self - указатель на экземпляр объекта
            Out:
                True - смещение есть
                False - смещение отсутствует
        '''
        return self.bias_enable


    def feed_forward_sigm(self, signal):
        '''
            Перемножение входного вектора на нейросеть с применением функции сигмоида
            In:
                self - указатель на экземпляр объекта
                x - входной вектор-сигнал размерности 1xN
                nn - нейросеть, состоящая из списка слоев
			    bias - признак наличия смещения (у каждого нейрона)
            Out:
                результат операции - матрица с послойными результатами перемножения
        '''
        x = [i for i in signal]
        outs = []
        for layer in self.nn:
            #Если у нас есть входы смещений, то вектор сигнала должен быть расширен на 1 элемент, равный 1
            x = mult_sigm(x + [1] if self.bias_enable else x, layer)
            outs.append(x)
        self.outs = outs

    def get_errors(self, taget):
        '''
            Ошибка вычисления
        '''
        self.errors = [(t - o)**2 for t,o in zip(target, self.outs[-1])]

    def get_delta0(self, target):
        '''
            Вычисление ошибки на выходе сети
            In:
                self - указатель на экземпляр объекта
			    target - целевой показатель выхода (список)
            Out:
                ошибка сети (список)
        '''
        self.delta0 = [(t - o) * sigmoidDerivative(o) for o, t in zip(self.outs[-1], target)]

    def get_delta_hidden_step(self, layer, out, delta):
        temp = [0 for _ in range(len(layer[0]))]
        for i in range(len(layer)):
            for j in range(len(layer[i])):
                temp[j] += layer[i][j]*delta[i]
        res_delta = [i for i in out]
        for i in range(len(res_delta)):
            res_delta[i] *= sigmoidDerivative(res_delta[i])*temp[i]
        return res_delta



    def back_propagation_deltas(self, target):
        '''
            Вычисление обратного распространения ошибки по сети
            In:
                self - указатель на экземпляр объекта`
                target - целевой показатель выхода (список)
            Out:
                deltas - ошибки на выходах нейросети
        '''
        self.get_delta0(target)
        tmp_delta = self.delta0
        loc_deltas = []
        loc_deltas.insert(0, tmp_delta)
        for layer,out in zip(self.nn[-1:0:-1], self.outs[-2::-1]):
            tmp_delta = self.get_delta_hidden_step(layer, out, tmp_delta)
            loc_deltas.insert(0, tmp_delta)
        self.deltas = loc_deltas

    def back_propagation_weights(self, signal, target, rho):
        '''
            Корректировка структуры сети
            In:
                self - указатель на экземпляр объекта
                signal - входной сигнал
                target - целевой показатель выхода (список)
                rho - шаг обучения
            Out:
                self.nn с измененными параметрами
        '''
        tmp_signal = [i for i in signal]
        for i in range(len(self.nn[0])):
            if self.bias_enable:
                tmp_signal += [1]
            for j in range(len(self.nn[0][i])):
                self.nn[0][i][j] = self.nn[0][i][j] + (rho*self.deltas[0][i]*tmp_signal[j])
        
        tmp_outs = self.outs
        for i in range(1, len(self.nn)):
            for j in range(len(self.nn[i])):
                if self.bias_enable:
                    tmp_outs[j] += tmp_outs[j] + [1]
                for k in range(len(self.nn[i][j])):
                    self.nn[i][j][k] = self.nn[i][j][k] + (rho*self.deltas[i][j]*self.outs[j][k])

    def learn_nn(self, signal, target, rho, count):
        i = 0
        #self.add_bias()
        while (i < count):
            self.feed_forward_sigm(signal)
            self.back_propagation_deltas(target)
            self.back_propagation_weights(signal, target, rho)
            i += 1



		

if __name__ == "__main__":

#    nn = [[[0.45, -0.12], [0.78, 0.13]],[[1.5, -2.3]]]
#    signal = [1, 0]
#    target = [1]

#    my_nn = NeuralNetwork(nn)
#    my_nn.feed_forward_sigm(signal)
#    print(my_nn.outs)
#    my_nn.get_errors(target)
#    print(my_nn.errors)
#    my_nn.back_propagation_deltas(target)
#    print(my_nn.deltas)
#    my_nn.back_propagation_weights(signal, target, 0.7)
#    print(my_nn.nn)

#    my_nn.feed_forward_sigm(signal)
#    print(my_nn.outs)
#    my_nn.get_errors(target)
#    print(my_nn.errors)

    nn = [[[20,20], [20,20]],[[-60,60]]]
    signal = [[0,0],[0,1],[1,0],[1,1]]
    targets = [[0],[1],[1],[0]]

    my_nn = NeuralNetwork(nn)
    my_nn.add_bias()

    for s,t in zip(signal,targets):
        my_nn.feed_forward_sigm(s)
        print(s, my_nn.outs[-1])
    
    j = 0
    while(j < 10000):
        for s,t in zip(signal,targets):
            my_nn.learn_nn(s, t, 0.5, 1)
        j += 1
    
    for s,t in zip(signal,targets):
        my_nn.feed_forward_sigm(s)
        print(s, my_nn.outs[-1])