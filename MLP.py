import numpy as np
import abc
from tqdm import tqdm

class BaseActivationFunction(metaclass=abc.ABCMeta):
    """ This is the base class for all activation functions """

    def __init__(self, type):
        self.type = type

        # this function keeps the current value as well as derivative
        self.value = 0
        self.derivative = 0


    @abc.abstractmethod
    def getValue(self, inputs):
        pass

    def getDerivative(self):
        return self.derivative

    def getType(self):
        return self.type


class SigmoidActivationFunction(BaseActivationFunction):
    """ This is the sigmoid activation function """

    def __init__(self):
        super().__init__('Sigmoid')

    def getValue(self, inputs):
        self.value = 1.0 / (1.0 + np.exp(-inputs))

        # value of derivative is updated immediately
        # self.derivative = np.exp(-inputs) / (1 + np.exp(-inputs)) ** 2
        self.derivative = self.value * (1 - self.value)

        return self.value


class ReLUActivationFunction(BaseActivationFunction):
    """ This is the ReLU activation function """

    def __init__(self):
        super().__init__('ReLU')

    def getValue(self, inputs):
        self.value = np.maximum(0, inputs)

        # value of derivative is updated immediately
        if inputs > 0.0:
            self.derivative = 1
        else:
            self.derivative = 0.001

        return self.value


class LinearActivationFunction(BaseActivationFunction):
    """ This is the Linear activation function """

    def __init__(self):
        super().__init__('Linear')

    def getValue(self, inputs):
        self.value = inputs

        # value of derivative is updated immediately
        self.derivative = 1

        return self.value


class BaseLossFunction(metaclass=abc.ABCMeta):
    """ This is the base class for all Lossfunctions """

    def __init__(self, type, outputLayer):
        self.outputLayer = outputLayer
        self.type = type
        self.value = 0
        self.derivatives = np.empty(outputLayer.numberOfPerceptrons())

    def getValue(self):
        return self.value

    @abc.abstractmethod
    def update(self, output):
        pass

    def getType(self):
        return self.type

    def getDerivative(self, index):
        return self.derivatives[index]


class MeanSquaredLossFunction(BaseLossFunction):
    """ This is the mean squared loss function """

    def __init__(self, outputLayer):
        super().__init__('MeanSquaredLossFunction', outputLayer)

    def update(self, output):
        num = output.shape[0]
        outmlp = self.outputLayer.getPerceptrons()
        val = np.sum((outmlp - output) ** 2)
        self.derivatives = 2.0 / num * (outmlp - output)
        # for i in range(num):
        #     outmlp = self.outputLayer.getPerceptron(i).getValue()
        #     val += (outmlp - output[i]) ** 2
        #     self.derivatives[i] = 2.0 / num * (outmlp - output[i])

        self.value = val/num


class CrossEntropyLossFunction(BaseLossFunction):
    """ This is the cross entropy loss function """

    def __init__(self, outputLayer):
        super().__init__('CrossEntropyLossFunction', outputLayer)

    def update(self, output):
        outmlp = self.outputLayer.getPerceptrons().astype(float)
        outmlp = np.exp(outmlp)
        den = np.sum(outmlp)
        softmax = outmlp/den
        lnSoftmax = np.log(softmax)
        val = -np.sum(output*lnSoftmax)
        self.derivatives = softmax - output
        self.value = val

        # val = 0
        # den = 0
        # num = output.shape[0]

        # for i in range(num):
        #     outmlp = self.outputLayer.getPerceptron(i).getValue()
        #     den += np.exp(outmlp)

        # for j in range(num):
        #     outmlp = self.outputLayer.getPerceptron(j).getValue()
        #     val += -output[j]*(outmlp-np.log(den))
        #     self.derivatives[j] = np.exp(outmlp)/den - output[j]
        # self.value = val

class Perceptron:
    """ This class encodes a single perceptron """

    def __init__(self, mlp, layer, num_inputs=0, activationFunction='sigmoid', isInputPerceptron=False):
        self.mlp = mlp
        self.num_inputs = num_inputs
        self.layer = layer
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
        # print("weights: ", self.weights, "bias: ", self.bias)
        self.derivatives = np.empty(num_inputs)
        self.isInputPerceptron = isInputPerceptron
        self.value = 0
        self.da = {'sigmoid': SigmoidActivationFunction, 'relu': ReLUActivationFunction,
                   'linear': LinearActivationFunction}
        self.activationFunction = self.da[activationFunction]()

    def update(self, precedingLayer, index=None):
        # create new input by multiplying weights with predecessors' corresponding output
        pPerceptrons = precedingLayer.getPerceptrons().astype(np.float)
        val = self.weights@pPerceptrons
        # for i in range(0, self.num_inputs):
        #     val += self.weights[i] * precedingLayer.getPerceptron(i).getValue()

        # the activation function is supposed to update it's derivative within function getValue()
        self.value = self.activationFunction.getValue(val + self.bias)

        # update derivatives
        self.derivatives = self.activationFunction.getDerivative() * self.weights


    def backprop(self, precedingLayer, succeedingLayer, index, learningRate):
        # do nothing if this is inputlayer
        if precedingLayer is None:
            return
        # get derivative of lossfucntion if this is outputlayer
        if succeedingLayer is None:
            d = self.mlp.lossFunction.getDerivative(index)
        # else eval and sum over all derivitaves with respect to weights
        else:
            d = 0
            for i in range(succeedingLayer.numberOfPerceptrons()):
                d += succeedingLayer.getPerceptron(i).getDerivative(index)

        # do sgd for bias
        self.bias -= learningRate * d * self.activationFunction.getDerivative()
        # get value of perceptrons of preceding layer
        pPerceptrons = precedingLayer.getPerceptrons().astype(np.float)
        # do sgd for weights
        self.weights -= learningRate * d * self.activationFunction.getDerivative() * pPerceptrons
        # update derivivates of perceptron
        self.derivatives *= d

    def getValue(self):
        return self.value

    def setValue(self, val):
        self.value = val

    def getDerivative(self, index):
        return self.derivatives[index]

    def setActivationFunction(self, activationFunction):
        self.activationFunction = self.da[activationFunction]()

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def __mul__(self, val: float) -> float:
        return self.value * val

    def __rmul__(self, val: float) -> float:
        return self * val

    def __add__(self, val: float) -> float:
        return self.value + val

    def __sub__(self, val: float) -> float:
        return self.value - val

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

class Layer:
    """ This class encodes a single layer """

    def __init__(self, mlp, num_perceptrons, activationFunction, precedingLayer):
        self.mlp = mlp
        self.numPerceptrons = num_perceptrons
        self.precedingLayer = precedingLayer
        self.perceptrons = np.empty(num_perceptrons, dtype=Perceptron)
        self.isInputLayer = None

        if precedingLayer is None:
            self.isInputLayer = True

            for i in range(num_perceptrons):
                self.perceptrons[i] = Perceptron(mlp, layer=self, isInputPerceptron=True)

        else:
            self.isInputLayer = False
            for i in range(num_perceptrons):
                self.perceptrons[i] = Perceptron(mlp, layer=self, num_inputs=precedingLayer.numberOfPerceptrons(),
                                                 activationFunction=activationFunction)

    def numberOfPerceptrons(self):
        return self.numPerceptrons

    def update(self):
        for i in range(self.numPerceptrons):
            self.perceptrons[i].update(self.precedingLayer, i)

    def backprop(self, succeedingLayer, learningRate):
        if succeedingLayer is None:
            # this layer is actually the output layer
            for i in range(self.numPerceptrons):
                self.getPerceptron(i).backprop(precedingLayer=self.precedingLayer, succeedingLayer=None, index=i,
                                               learningRate=learningRate)
        else:
            # this is obviously a hidden layer
            for i in range(self.numPerceptrons):
                self.getPerceptron(i).backprop(precedingLayer=self.precedingLayer, succeedingLayer=succeedingLayer,
                                               index=i, learningRate=learningRate)

    def getPerceptron(self, index):
        return self.perceptrons[index]

    def setActivationFunction(self, activationFunction):
        for i in range(self.numberOfPerceptrons()):
            self.perceptrons[i].setActivationFunction(activationFunction)

    def showPerceptrons(self):
        print(self.perceptrons)

    def getPerceptrons(self):
        return self.perceptrons


class MultiLayerPerceptron:
    """ This class encodes the actual multilayer perceptron """

    def __init__(self, topology, lossFunction, activationFunction='sigmoid'):
        self.topology = topology
        self.layers = np.empty(topology.size, dtype=Layer)
        self.numLayers = topology.size
        self.dl = {'meansquared': MeanSquaredLossFunction, 'crossentropy': CrossEntropyLossFunction}

        self.layers[0] = Layer(self, self.topology[0], activationFunction, precedingLayer=None)
        for i in range(1, topology.size):
            self.layers[i] = Layer(self, self.topology[i], activationFunction, precedingLayer=self.layers[i - 1])
        if lossFunction == 'crossentropy':
            self.layers[-1].setActivationFunction('linear')

        self.lossFunction = self.dl[lossFunction](self.layers[self.numberOfLayers() - 1])

    #########################
    # Some helper functions #
    #########################
    def numberOfLayers(self):
        return self.numLayers

    def getLayer(self, index):
        return self.layers[index]

    def setValuesOfInputLayer(self, inputs):
        # check whether the input fits the number of input perceptrons
        if inputs.size != self.layers[0].numberOfPerceptrons():
            print('size of data does not fit')
            return False
        else:
            # fill in values of input perceptrons
            for i in range(self.layers[0].numberOfPerceptrons()):
                self.layers[0].getPerceptron(i).setValue(inputs[i])

            return True

    def getLossFunction(self):
        return self.lossFunction

    ########################
    # show MLPs structure  #
    ########################
    def summary(self, learningRate=None, numberOfEpochs=None, globalError=None, globalAccuracy=None):
        print('---------------------------------')
        print('Structure: ')
        print('Layer 0:')
        print('  Number of Neurons: {}\n'.format(self.layers[0].numberOfPerceptrons()))
        for i in range(1, self.numberOfLayers()):
            print('Layer {}:'.format(i))
            print('  Number of Neurons: {}'.format(self.layers[i].numberOfPerceptrons()))
            print('  Activation Function: {}'.format(self.layers[i].getPerceptron(0).activationFunction.getType()))
            print()

        print('Loss Function: {}'.format(self.lossFunction.getType()))
        if learningRate is not None:
            print('Learningrate: {}'.format(learningRate))
        if numberOfEpochs is not None:
            print('Epochs: {}'.format(numberOfEpochs))
        if globalError is not None:
            print('Loss: {}'.format(globalError))
        if globalAccuracy is not None:
            print('Accuracy: {}'.format(globalAccuracy))
        print('---------------------------------')

    ########################################################
    # features are supposed to be stored in 1d numpy array #
    ########################################################
    def predict(self, inputs, softmax=False):
        # check whether the input fits the number of input perceptrons
        if self.setValuesOfInputLayer(inputs):
            for j in range(1, self.numberOfLayers()):
                self.getLayer(j).update()
            # output = np.empty(self.getLayer(self.numberOfLayers() - 1).numberOfPerceptrons())

            # for k in range(output.size):
            #     output[k] = self.getLayer(self.numberOfLayers() - 1).getPerceptron(k).getValue()
            # print(self.getLayer(self.numberOfLayers() - 1).getPerceptrons().astype(np.double).dtype)
            output = self.getLayer(self.numberOfLayers() - 1).getPerceptrons().astype(np.double)
            if softmax:
                # den = 0
                # for i in range(output.size):
                #     den += np.exp(output[i])
                #     output[i] = np.exp(output[i])

                output = np.exp(output)
                den = np.sum(output)
                output /= den
            return output
        return None

    ################################################################
    # features-vectors are supposed to be stored in 2d numpy array #
    # number of samples x number of features                       #
    ################################################################
    def predictAll(self, inputs, softmax=False):
        # check if dimensions match
        if inputs.shape[1] != self.layers[0].numberOfPerceptrons():
            print('size of data does not fit')
            return

        results = np.empty((inputs.shape[0], self.layers[self.numberOfLayers() - 1].numberOfPerceptrons()))
        for i in range(inputs.shape[0]):
            results[i, :] = self.predict(inputs[i], softmax)

        return results

    ################################################################
    # input data is supposed to be stored in 2d numpy array        #
    # number of samples x number of input neurons                  #
    ################################################################
    # input data is supposed to be stored as (numberOfSample, sizeOfInputLayer)
    def learn(self, inputs, outputs, learningRate, numberOfEpochs, batch_size=None, output_epochs=None, usetqdm=True, printSummary=False):
        # check whether the input and out fits the number of input and output perceptrons
        if inputs.shape[1] != self.layers[0].numberOfPerceptrons() or outputs.shape[1] != self.layers[
            self.numberOfLayers() - 1].numberOfPerceptrons() or inputs.shape[0] != outputs.shape[0]:
            print('size of data does not fit')
            return

        bar = range(numberOfEpochs)
        if usetqdm:
            bar = tqdm(bar)
        if outputs.shape[1] > 1:
            softmax = True
        else:
            softmax = False
        globalError = 0
        globalAccuracy = 0
        for i in bar:
            err = 0
            # shuffle data first inputs and outputs
            if batch_size is None or type(batch_size) != int or batch_size > inputs.shape[0]:
                batch_size = inputs.shape[0]

            p = np.random.choice(inputs.shape[0], batch_size, replace=False)
            inputs_after_choice = inputs[p]
            outputs_after_choice = outputs[p]
            nCorrects = 0

            for k in range(inputs_after_choice.shape[0]):
                output = self.predict(inputs_after_choice[k], softmax=softmax)
                if output.shape[0] > 1:
                    output = (output == output.max(axis=0)).astype(int)
                else:
                    output = np.round(output).astype(int)

                if np.allclose(output, outputs_after_choice[k]):
                    nCorrects += 1
                self.getLossFunction().update(outputs_after_choice[k])
                err += self.getLossFunction().getValue()

                # loop over all but the first layer in reverse order
                self.getLayer(self.numberOfLayers() - 1).backprop(None, learningRate)
                for l in range(self.numberOfLayers() - 2, 0, -1):
                    self.getLayer(l).backprop(self.getLayer(l + 1), learningRate)
            # return err/inputs_after_choice.shape[0]
            globalError = err/inputs_after_choice.shape[0]
            globalAccuracy = nCorrects/inputs_after_choice.shape[0]
            if usetqdm:
                bar.set_postfix({'loss': globalError, 'accuracy': globalAccuracy})
            elif output_epochs is None or (i + 1) % output_epochs == 0 or i == 0:
                print("Mean error in epoch {} : {}".format(i + 1, err / inputs_after_choice.shape[0]))
        if printSummary:
            self.summary(learningRate, numberOfEpochs, globalError, globalAccuracy)


if __name__ == "__main__":
    # MLP = MultiLayerPerceptron(np.array([2, 2, 1]), 'meansquared')
    # MLP.summary()
    # data = np.array([
    #   [-2, -1],  # Alice
    #   [25, 6],   # Bob
    #   [17, 4],   # Charlie
    #   [-15, -6], # Diana
    # ])
    # all_y_trues = np.array([
    #   [1], # Alice
    #   [0], # Bob
    #   [0], # Charlie
    #   [1], # Diana
    # ])
    # MLP.learn(data, all_y_trues, 1, 100000)
    # MLP = MultiLayerPerceptron(np.array([30, 5, 5, 1]), 'meansquared')

    # MLP.summary()
    from sklearn.model_selection import train_test_split
    from sklearn import metrics, datasets
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    data = datasets.load_iris()
    X = data.data
    y = data.target
    yy = np.zeros((y.size, 3))
    yy[np.argwhere(y==2), 2] = 1
    yy[np.argwhere(y==0), 0] = 1
    yy[np.argwhere(y==1), 1] = 1
    y = yy
    print("{} {}".format(X.shape, y.shape))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    mlp_iris = MultiLayerPerceptron(np.array([4, 5, 3]), 'crossentropy', 'relu')
    mlp_iris.learn(sc.fit_transform(x_train), y_train, 0.1, 1000, printSummary=True)
    y_pred = mlp_iris.predictAll(sc.fit_transform(x_test))
    print(x_train.shape, y_train.shape)
    y_pred = (y_pred == y_pred.max(axis=1)[:, None]).astype(int)
    print(metrics.accuracy_score(y_test, y_pred)*100)

    # from sklearn.datasets import load_breast_cancer
    # data = load_breast_cancer()
    # X = data.data
    # y = data.target.reshape((len(data.data), 1))
    # print("{} {}".format(X.shape, y.shape))
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    # MLP.learn(x_train, y_train, 0.1, 10000)
    # y_pred = MLP.predictAll(x_test)

    # print(metrics.accuracy_score(y_test, y_pred)*100)

    # import pandas as pd

    # # Importing the dataset
    # dataset = pd.read_csv('Churn_Modelling.csv')
    # X = dataset.iloc[:, 3:13].values
    # y = dataset.iloc[:, 13].values

    # from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    # from sklearn.compose import ColumnTransformer

    # ct = ColumnTransformer([("OneHotEncoding", OneHotEncoder(),[1, 2])], remainder="passthrough")

    # X = ct.fit_transform(X)
    # X = X[:, 3:]
    # print(X.shape)
    # # Splitting the dataset into the Training set and Test set
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)
    # # Feature Scaling
    # t = 0
    # f = 0
    # for i in y_train:
    #     if i:
    #         t += 1
    #     else:
    #         f +=1
    # print(t, f)
    # y_train = y_train.reshape((2000, 1))
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    # MLP = MultiLayerPerceptron(np.array([10, 7, 1]), 'meansquared', 'sigmoid')
    # MLP.learn(X_train, y_train, .1, 100, softmax=False)

    # y_pred = MLP.predictAll(X_test)
    # y_pred = (y_pred > 0.5)
    # t = 0
    # f = 0
    # for i in y_pred:
    #     if i:
    #         t += 1
    #     else:
    #         f +=1
    # print(t, f)
    # # Making the Confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # print(metrics.accuracy_score(y_test, y_pred)*100)


