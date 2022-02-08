from strategies.AbstractMnistStrategy import AbstractMnistStrategy

from keras.models import Sequential
from keras.layers import Dense

class LinearClassifier(AbstractMnistStrategy):
    """
    x_train : 60000 x 28 x 28 -> 60000 x 784
    x_test  : 10000 x 28 x 28 -> 10000 x 784
    """
    def reshapeDataset(self, data):
        return data.reshape(data.shape[0], -1)

    """
    Total params: 7,850
    Trainable params: 7,850
    Non-trainable params: 0
    """
    def getModel(self):
        inputShape = 28 * 28
        model = Sequential()
        model.add(Dense(self.numClasses, input_dim = inputShape, activation = 'softmax'))

        return model
