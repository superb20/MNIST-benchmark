from enum import Enum

from strategies.AbstractMnistStrategy import AbstractMnistStrategy
from strategies.LinearClassifier import LinearClassifier
from strategies.LeNet1Classifier import LeNet1Classifier
from strategies.LeNet4Classifier import LeNet4Classifier
from strategies.LeNet5Classifier import LeNet5Classifier
from strategies.McdnnClassifier import McdnnClassifier

class MnistStrategy(Enum):
    LINEAR = 0
    LENET1 = 1
    LENET4 = 2
    LENET5 = 3
    ALEXNET = 4
    MCDNN = 5

    def getStrategy(MODEL):
        if MODEL == MnistStrategy.LINEAR:
            return LinearClassifier()

        elif MODEL == MnistStrategy.LENET1:
            return LeNet1Classifier()

        elif MODEL == MnistStrategy.LENET4:
            return LeNet4Classifier()

        elif MODEL == MnistStrategy.LENET5:
            return LeNet5Classifier()

        elif MODEL == MnistStrategy.MCDNN:
            return McdnnClassifier()

        else:
            return AbstractMnistStrategy()
