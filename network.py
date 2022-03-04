import numpy as np

class Network(object):
    def __init__(self, layers):
        """
        Initializing the weights and biases of the network. Parameter
        "layers" is a list of numbrs whose length represents the number
        of layers and the value of individual elements represent the
        number of neurons. 
        """
        self.weights = [np.random.randn(j, k) for j, k in zip(layers[1:], layers[0:-1])]
        self.biases = [np.random.randn(j, 1) for j in layers[1:]]