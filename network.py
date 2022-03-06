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

    def backprop(self, ybatch, zlist, alist):
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
        batchsize = ybatch.shape[1]

        error = (alist[-1] - ybatch) * sigmoidprime(zlist[-1])
        errors.insert(0, error)
        weightder = np.dot(error, np.transpose(alist[-2]))
        weightders.insert(0, weightder / batchsize)
        biasder = np.sum(error, axis=1, keepdims=True)
        biasders.insert(0, biasder / batchsize)

        for i in range(2, self.numlayers):
            error = np.multiply(np.dot(np.transpose(self.weights[-i + 1]), error), sigmoidprime(zlist[-i]))
            errors.insert(0, error)
            weightder = np.dot(error, np.transpose(alist[-i-1]))
            weightders.insert(0, weightder / batchsize)
            biasder = np.sum(error, axis=1, keepdims=True) / batchsize
            biasders.insert(0, biasder)

        return weightders, biasders

    def SGD(self, data, epochs, valdata = False, batchsize = 20, lr = 0.01):
        """
        First we split the features and labels from the data (data
        comes in form of a tuple (features, labels)). Then we create
        a list of indices and shuffle them. Next we create a list of
        minibatches with those shuffled indices. 
        """
        features = data[0]
        labels = data[1]

        #well shuffle the indices to be sure we get random values
        #when calculating the derivatives using batches
        indices = [i for i in range(features.shape[1])]

        for epoch in range(epochs):
            np.random.shuffle(indices)

            nrexamples = len(indices)
            tail = 0
            indexlist = []

            for i in range(int(np.floor(nrexamples/batchsize)) + 1):
                head = int(nrexamples - np.floor(nrexamples/batchsize) * batchsize + batchsize * i)
                minibatch = indices[tail:head]
                tail = head
                indexlist.append(minibatch)

            if nrexamples % batchsize == 0:
                indexlist = indexlist[1:]

            for index in indexlist:
                #feedforward step
                zlist, alist = self.feedforward(features[:, index])

                #backpropagation step
                weightders, biasders = self.backprop(labels[:, index], zlist, alist)

                for i in range(len(self.weights)):
                    self.weights[i] = self.weights[i] - lr * weightders[i]
                    self.biases[i] = self.biases[i] - lr * biasders[i]
            
            if valdata == False:
                print(f"End of epoch {epoch}")
            else:
                accuracy = self.accuracy(valdata)
                print(f"End of epoch {epoch}: {accuracy} / {valdata[0].shape[1]}")
    
    def predict(self, features):
        "Given the features of examples predict the output"

        nrexamples = features.shape[1]
        zlist, alist = self.feedforward(features)
        maxindex = np.reshape(np.argmax(alist[-1], axis = 0),(1,nrexamples))

        return maxindex

    def accuracy(self, data):
        "Gives the accuracy of model given data"

        features = data[0]
        labels = data[1]
        #nrexamples = features.shape[1]

        predictions = self.predict(features)

        accuracy = np.sum(labels == predictions)

        return accuracy
            






            




def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidprime(z):
    return sigmoid(z) * (1 - sigmoid(z))
