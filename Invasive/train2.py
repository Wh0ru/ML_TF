import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
import keras
from keras.layers import Concatenate
from pre_data import load_train,load_test
from keras.models import Model
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
import keras.backend as K
from keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pre_data import centering_image
import matplotlib.pyplot as plt

train_path='train/'
test_path='test/'
train_files,train_set,train_label=load_train(train_path)
IMAGE_SIZE=224
CHANNELS=3
EPOCHES=60

X,y=shuffle(train_files,train_label)
scale=int(len(X)*0.8)
X_train_files,y_train,X_valid_files,y_valid=X[:scale],y[:scale],X[scale:],y[scale:]

X_train=[]
X_valid=[]
BATCH_SIZE=16

for i in X_train_files:
    img=cv2.imread(i)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if(img.shape[0]>img.shape[1]):
        title_size=(int(img.shape[1]*256/img.shape[0]),256)
    else:
        title_size=(256,int(img.shape[0]*256/img.shape[1]))
    img=cv2.resize(img,title_size)
    img=centering_image(img)
    img=img[16:240,16:240]
    X_train.append(img)
X_train=np.array(X_train)

train_datagen=ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255.,
)

train_generator=train_datagen.flow(X_train,y_train,batch_size=BATCH_SIZE,shuffle=False)

for i in X_valid_files:
    img=cv2.imread(i)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if(img.shape[0]>img.shape[1]):
        title_size=(int(img.shape[1]*256/img.shape[0]),256)
    else:
        title_size=(256,int(img.shape[0]*256/img.shape[1]))
    img=cv2.resize(img,title_size)
    img=centering_image(img)
    img=img[16:240,16:240]
    img=cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
    X_valid.append(img)
X_valid=np.array(X_valid)

valid_datagen=ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255.
)

valid_generator=valid_datagen.flow(X_valid,y_valid,batch_size=BATCH_SIZE,shuffle=False)

input_tensor=Input(shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS))
base_model_1=Xception(include_top=False,weights='imagenet',input_tensor=input_tensor)
base_model_2=InceptionV3(include_top=False,weights='imagenet',input_tensor=input_tensor)
# base_model_3=ResNet50(include_top=False,weights='imagenet',input_tensor=input_tensor)
x_1=base_model_1.output
x_1=GlobalAveragePooling2D()(x_1)

x_2=base_model_2.output
x_2=GlobalAveragePooling2D()(x_2)

# x_3=base_model_3.output
# x_3=GlobalAveragePooling2D()(x_3)

x=Concatenate()([x_1,x_2])
x=Dropout(0.5)(x)
output=Dense(1,activation='sigmoid')(x)

for layer in base_model_1.layers:
    layer.trainable=False

for layer in base_model_2.layers:
    layer.trainable = False

# for layer in base_model_3.layers:
#     layer.trainable = False

model=Model(inputs=input_tensor,outputs=output)

adam=Adam(lr=2*1e-3)
sgd=SGD(lr=1e-3,momentum=0.9)

model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

checkpoints=ModelCheckpoint('weights_best_{epoch:02d}.hdf5',monitor='val_acc',verbose=1,save_best_only=True)
lr_reducer=ReduceLROnPlateau(monitor='val_acc',factor=0.4,patience=3,verbose=1,min_lr=1e-6)

callbacks=[checkpoints,lr_reducer]

model.fit_generator(train_generator,steps_per_epoch=train_generator.n/BATCH_SIZE,epochs=EPOCHES,validation_data=valid_generator,
                    validation_steps=valid_generator.n/BATCH_SIZE,shuffle=False,callbacks=callbacks)