import math
import random

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def sigmoidDerivative(x):
    return x * (1 - x)

class NeuralNetwork:

    __instance = None;

    @classmethod
    def getInstance(cls, *args):
        '''
            Получаем объект NeuralNetworks
            на вход передаются информация о сети в виде i, h1, h1, ..., o, где
            i - количество входов
            hN - количество нейронов в скрытом N слое
            o - количество выходов
        '''
        if len(args) < 2:
            raise Exception('Ошибка создания. Нейронная сеть должна содержать хотя бы 2 слоя')
            return None;
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

    def printLayers(self):
        print(self.layers)

    def printNetwork(self):
        print(self.network)

if __name__ == '__main__':
    nn = NeuralNetwork.getInstance(2, 4, 5, 1)
    nn.printNetwork()
            
