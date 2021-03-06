import math
import random

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def sigmoidDerivative(x):
    return x * (1 - x)

class NeuralNetwork:

    __instance = None

    @classmethod
    def getInstance(cls, *args):
        '''
            Возвращаем объект NeuralNetworks
            на вход передаются информация о сети в виде i, h1, h1, ..., o, где
            i - количество входов
            hN - количество нейронов в скрытом N слое
            o - количество выходов
        '''
        if len(args) < 2:
            raise Exception('Ошибка создания. Нейронная сеть должна содержать хотя бы 2 слоя')
        if cls.__instance is None:
            cls.__instance = cls(*args)
        return cls.__instance

    def __init__(self, *args):
        self.layers = list(args)
        
        self.network = []
        for i in range(len(self.layers) - 1):
            layer = [[random.random() for _ in range(self.layers[i + 1])] for _ in range(self.layers[i] + 1)]
            self.network.append(layer)
            
        self.hiddens = []
        self.errh = []
        for i in range(1, len(self.layers) - 1):
            hidden = [0 for _ in range(self.layers[i])]
            errhidden = [0 for _ in range(self.layers[i])]
            self.hiddens.append(hidden)
            self.errh.append(errhidden)
            
        self.out = [0 for _ in range(self.layers[-1])]
        self.erro = [0 for _ in range(self.layers[-1])]

    def printLayer(self, layerNumber):
        '''
            Вывод на экран структуры слоя с номером layerNumber
            layerNumber - номер слоя, который необходимо вывести на экран
        '''
        try:
            layer = self.network[layerNumber]
            print('Слой %d'%layerNumber)
            for x in range(len(layer)):
                for y in range(len(layer[x])):
                    print(layer[x][y], end = ' ')
                print()
            print()
        except Exception:
            print('Указан некорректный номер слоя')

    def printLayers(self):
        '''
            Вывод на экран структуры всех слоев сети
        '''
        for i in range(len(self.network)):
            self.printLayer(i)
    
    def printOut(self):
        '''
            Вывод на экран выходов сети
        '''
        print('Значения выходного слоя:')
        for o in self.out:
            print(o, end = ' ')
        print('\n')
    
    def printHiddenOut(self, hiddenLayerNum):
        '''
            Вывод на экран значений слоя hiddenLayerNum
            hiddenLayerNum - номер слоя, котрый необходимо вывести на экран
        '''
        try:
            hiddenLayer = self.hiddens[hiddenLayerNum]
            print('Значения на выходах скрытого слоя %d'%hiddenLayerNum)
            for o in hiddenLayer:
                print(o, end = ' ')
            print('\n')
        except Exception:
            print('Указан некорректный номер скрытого слоя')
    
    def printHiddenOuts(self):
        '''
            Вывод на экран значений всех скрытых слоев
        '''
        for hOuts in range(len(self.hiddens)):
            self.printHiddenOut(hOuts)

    def printNetwork(self):
        '''
            Вывод на экран структуры сети и значений выходов
        '''
        self.printLayers()
        self.printHiddenOuts()
        self.printOut()
        

if __name__ == '__main__':
    nn = NeuralNetwork.getInstance(2, 2, 1)
    nn.printNetwork()