from math import exp

def sigmoid(x):
    '''
        Вычисление сигмоида по формуле 1/(1+exp(-x))
        In: x - число
        Out: результат операции - число
    '''
    return 1/(1+exp(-x))

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

def feed_forward_sigm(x, nn, bias = False):
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
    for layer in nn:
        #Если у нас есть входы смещений, то вектор сигнала должен быть расширен на 1 элемент, равный 1
        x = mult_sigm(x + [1] if bias else x, layer)
        outs.append(x)
    return outs

class NeuralNetwork:	

    nn = []
    bias_enable = False
    outs = []
	
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


    def feed_forward_sigm(self, x):#, nn, bias = False):
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
        for layer in self.nn:
            #Если у нас есть входы смещений, то вектор сигнала должен быть расширен на 1 элемент, равный 1
            x = mult_sigm(x + [1] if self.bias_enable else x, layer)
            outs.append(x)
        self.outs = outs
		

if __name__ == "__main__":
    nn = [[[1, 0.5],[-1,2]],[[1.5, -1]]]
    inp = [0, 1]

    my_nn = NeuralNetwork(nn)
    my_nn.add_bias()
    my_nn.feed_forward_sigm(inp)
    print(my_nn.outs)

