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
        tmp_layer = [inputs + [1] if bias else inputs for inputs in layer]
        x = mult_sigm(x + [1] if bias else x, tmp_layer)
        outs.append(x)
    return outs

if __name__ == "__main__":
    nn = [[[1, 0.5],[-1,2]],[[1.5, -1]]]
    inp = [0, 1]

    print(feed_forward_sigm(inp, nn, bias = True))
    print(nn)
