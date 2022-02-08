from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils import plot_model

from MnistStrategy import MnistStrategy

# Select strategy
"""
MnistStrategy.LINEAR  : Simple Linear Classifier
MnistStrategy.LENET1  : LeNet-1
MnistStrategy.LENET4  : LeNet-4
MnistStrategy.LENET5  : LeNet-5
MnistStrategy.MCDNN   : MCDNN
"""
strategy = MnistStrategy.getStrategy(MnistStrategy.MCDNN)
# Load dataset
"""
x_train : 60000 x 28 x 28 <class 'numpy.ndarray'>
y_train : 60000 x 1 <class 'numpy.ndarray'>
x_test  : 10000 x 28 x 28 <class 'numpy.ndarray'>
y_test  : 10000 x 1 <class 'numpy.ndarray'>
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('Shape of x_train dataset: {}'.format(x_train.shape))
print('Type of x_train dataset : {}'.format(type(x_train)))
print('Shape of y_train dataset: {}'.format(y_train.shape))
print('Type of y_train dataset : {}'.format(type(y_train)))
print('Shape of x_test dataset : {}'.format(x_test.shape))
print('Type of x_test dataset  : {}'.format(type(x_test)))
print('Shape of y_test dataset : {}'.format(y_test.shape))
print('Type of y_test dataset  : {}\n'.format(type(y_test)))

# Reshape the train and test dataset
"""
MnistStrategy.Linear
x_train : 60000 x 28 x 28 -> 60000 x 784
x_test  : 10000 x 28 x 28 -> 10000 x 784

MnistStrategy.LENET1
x_train : 60000 x 28 x 28 -> 60000 x 28 x 28 x 1
x_test  : 10000 x 28 x 28 -> 10000 x 28 x 28 x 1

MnistStrategy.LENET4
MnistStrategy.LENET5
x_train : 60000 x 28 x 28 -> 60000 x 32 x 32 x 1
x_test  : 10000 x 28 x 28 -> 10000 x 32 x 32 x 1

MnistStrategy.MCDNN : MCDNN(SINGLE)
x_train : 60000 x 28 x 28 -> 60000 x 29 x 29 x 1
x_test  : 10000 x 28 x 28 -> 10000 x 29 x 29 x 1
"""
x_train = strategy.reshapeDataset(x_train)
x_test  = strategy.reshapeDataset(x_test)
print('Shape of x_train dataset after reshape: {}'.format(x_train.shape))
print('Shape of x_test dataset after reshape : {}\n'.format(x_test.shape))

# Range of vlues
print('Range of values ​​of x_train: %d ~ %d' % (x_train.min(), x_train.max()))
print('Range of values ​​of x_test : %d ~ %d' % (x_test.min(), x_test.max()))

# Normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print('Range of values ​​of x_train after normalization : %d ~ %d' % (x_train.min(), x_train.max()))
print('Range of values ​​of x_test after normalization  : %d ~ %d\n' % (x_test.min(), x_test.max()))

# One-hot-Encoding
"""
y_train : 60000 x 1 -> 60000 x 10
y_test  : 10000 x 1 -> 10000 x 10
"""
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print('Shape of y_train dataset after One hot encoding: {}'.format(y_train.shape))
print('Shape of y_test dataset after One hot encoding : {}\n'.format(y_test.shape))

# Get model
"""
MnistStrategy.Linear
Total params: 7,850
Trainable params: 7,850
Non-trainable params: 0

MnistStrategy.LENET1
Total params: 3,246
Trainable params: 3,246
Non-trainable params: 0

MnistStrategy.LENET4
Total params: 51,050
Trainable params: 51,050
Non-trainable params: 0

MnistStrategy.LENET5
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0

MnistStrategy.MCDNN(SINGLE)
Total params: 76,040
Trainable params: 76,040
Non-trainable params: 0
"""
model = strategy.getModel()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
plot_model(model, show_shapes=True)

# Train model
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 30, batch_size = 200)

# Print train result
"""
MnistStrategy.LINEAR
train: 93.42%
test : 92.86%

MnistStrategy.LENET1
train: 98.57%
test : 98.44%

MnistStrategy.LENET4
train: 99.80%
test : 99.01%

MnistStrategy.LENET5
train: 99.82%
test : 98.97%

MnistStrategy.MCDNN(SINGLE)
train: 99.93%
test : 99.14%
"""
print("train accuracy: {:5f}%".format(model.evaluate(x_train, y_train, verbose = 0)[1] * 100))
print("test  accuracy: {:5f}%".format(model.evaluate(x_test, y_test, verbose = 0)[1] * 100))
