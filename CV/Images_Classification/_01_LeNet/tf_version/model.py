from tensorflow.keras.layers import Dense, Flatten, Conv2D ,MaxPool2D
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(6, 5, activation='relu',padding='same')
        self.pool1 = MaxPool2D()
        self.conv2 = Conv2D(16,5,activation='relu')
        self.pool2 = MaxPool2D()
        self.flatten = Flatten()
        self.d1 = Dense(84, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)      # input[batch, 28, 28, 1] output[batch, 28, 28, 6]
        x = self.pool1(x)      # output [batch, 14, 14, 6]
        x = self.conv2(x)      # output [batch, 10, 10, 16]
        x = self.pool2(x)      # output [batch, 5, 5, 16]
        x = self.flatten(x)    # output [batch, 120]
        x = self.d1(x)         # output [batch, 84]
        return self.d2(x)      # output [batch, 10]
