import numpy as np

class Network(object):
    def __init__(self, layers):
        """
        Initializing the weights and biases of the network. Parameter
        "layers" is a list of numbrs whose length represents the number
        of layers and the value of individual elements represent the
        number of neurons. 
        """
        self.numlayers = len(layers)
        self.weights = [np.random.randn(j, k) for j, k in zip(layers[1:], layers[0:-1])]
        self.biases = [np.random.randn(j, 1) for j in layers[1:]]

    def feedforward(self, batch):
        """
        Performing forward propagation on the given batch. Saving the
        weighted inputs of each layer in a list "zlist" and activations
        in a list "alist" that is going to be used in backpropagation.
        """

        a = batch
        zlist = []
        alist = [batch]

        for step in range(self.numlayers - 1):
            z = np.dot(self.weights[step], a) + self.biases[step]
            zlist.append(z)
            a = sigmoid(z)
            alist.append(a)
        
        return zlist, alist


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
