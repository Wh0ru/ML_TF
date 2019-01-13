from keras.layers import MaxPooling2D,Conv2D,BatchNormalization,Activation,Reshape,Dense
from keras.layers import ZeroPadding2D,Add,SpatialDropout2D,GlobalAveragePooling2D,Multiply
from keras.models import Input,Model
from ConvOffset2D import ConvOffset2D


def senblock(inp,filters):
    x = GlobalAveragePooling2D()(inp)
    _filter = filters//16
    x = Reshape((1, 1, filters))(x)
    x = Dense(_filter, use_bias=False, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dense(filters, use_bias=False, kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    x = Multiply()([inp, x])
    return x



def conv_block(inp,kernel_size,filters,strides=(2,2)):
    x=Conv2D(filters,kernel_size=kernel_size,strides=strides,padding='same')(inp)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    x=Conv2D(filters,kernel_size=kernel_size,padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    # x = senblock(x, filters)

    shortcut=Conv2D(filters,kernel_size=(1,1),strides=strides,padding='same')(inp)
    shortcut=BatchNormalization()(shortcut)


    x=Add()([x,shortcut])
    x=Activation('relu')(x)
    return x

def identity_block(inp,kernel_size,filters):
    x=Conv2D(filters,kernel_size=kernel_size,padding='same')(inp)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    x=Conv2D(filters,kernel_size=kernel_size,padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    # x = senblock(x, filters)

    x=Add()([x,inp])
    x=Activation('relu')(x)
    return x

def dfconv_block(inp,kernel_size,filters,strides=(2,2)):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inp)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    x=ConvOffset2D(filters)(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    # x = senblock(x, filters)

    shortcut=Conv2D(filters,kernel_size=(1,1),strides=strides,padding='same')(inp)
    shortcut=BatchNormalization()(shortcut)


    x=Add()([x,shortcut])
    x=Activation('relu')(x)
    return x

def dfidentity_block(inp,kernel_size,filters):
    x=ConvOffset2D(filters)(inp)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    x=ConvOffset2D(filters)(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    # x = senblock(x, filters)

    x=Add()([x,inp])
    x=Activation('relu')(x)
    return x

def resnet18(inp,dropout=0.1):
    x_1=ZeroPadding2D((3,3))(inp)
    x_1=Conv2D(64,(7,7),strides=(2,2))(x_1)
    x_1=BatchNormalization()(x_1)
    x_1=Activation('relu')(x_1)
    # x_1 = SpatialDropout2D(dropout)(x_1)

    x_2=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x_1)
    x_2=identity_block(x_2,kernel_size=(3,3),filters=64)
    x_2=identity_block(x_2,kernel_size=(3,3),filters=64)
    # x_2 = SpatialDropout2D(dropout)(x_2)

    x_3=conv_block(x_2,kernel_size=(3,3),filters=128)
    x_3=identity_block(x_3,kernel_size=(3,3),filters=128)
    # x_3 = SpatialDropout2D(dropout)(x_3)

    x_4=conv_block(x_3,kernel_size=(3,3),filters=256)
    x_4=identity_block(x_4,kernel_size=(3,3),filters=256)
    # x_4 = SpatialDropout2D(dropout)(x_4)

    x_5=dfconv_block(x_4,kernel_size=(3,3),filters=512)
    x_5=dfidentity_block(x_5,kernel_size=(3,3),filters=512)
    # x_5 = SpatialDropout2D(dropout)(x_5)

    return x_5
