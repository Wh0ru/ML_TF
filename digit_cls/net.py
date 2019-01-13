from keras.layers import MaxPooling2D,Conv2D,BatchNormalization,Dense
from keras.layers import Activation,Add,Multiply,GlobalAveragePooling2D
from keras.models import Model,Input
from resnet18 import resnet18
from ConvOffset2D import ConvOffset2D

def conv_block(inp, filters, kernel_size, act, strides=(1, 1)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', strides=strides)(inp)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    return x

def Net(shape,numclass):
    inp = Input(shape=shape)

    x = resnet18(inp)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(numclass, activation='softmax')(x)

    model = Model(inp,outputs)
    return model

# model=Net((28,28,1),10)
# print(model.summary())