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

    def backprop(self, batchsize, ybatch, zlist, alist):
        """
        Calculating the gradients of weights and biases with respect to
        the cost in a batch. Returning weightders - list containing all
        weight derivatives starting with the end - and biasders - list
        containing all bias derivatices starting with the end. All of them
        are divided by the number of examples in a batch
        """
        errors = []
        weightders = []
        biasders = []

        error = ybatch - alist[-1]
        errors.append(error)
        weightder = np.dot(error, np.transpose(alist[-2]))
        weightders.append(weightder / batchsize)
        biasder = np.sum(error, axis=1, keepdims=True) / batchsize
        biasders.append(biasder)

        for i in range(2, self.numlayers):
            error = np.multiply(np.dot(np.transpose(self.weights[-i + 1]), error), sigmoidprime(zlist[-i]))
            errors.append(error)
            weightder = np.dot(error, np.transpose(alist[-i-1]))
            weightders.append(weightder / batchsize)
            biasder = np.sum(error, axis=1, keepdims=True) / batchsize
            biasders.append(biasder)

        return weightders, biasders




def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidprime(z):
    return sigmoid(z) * (1 - sigmoid(z))
