import numpy as np
from pre_data import get_data,preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import GlobalAveragePooling2D,Dropout,Dense,Average,Conv2D,MaxPooling2D,BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Input
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,CSVLogger
from keras.preprocessing.image import ImageDataGenerator


batch_size=64
train,train_label,vid,vid_label=get_data()

train_datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True)
    # preprocessing_function=preprocess_input)

train_generator=train_datagen.flow(train,train_label,batch_size=batch_size,
                                   shuffle=False)

vid_datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True)
    # preprocessing_function=preprocess_input

vid_generator=vid_datagen.flow(vid,vid_label,batch_size=batch_size,
                                   shuffle=False)
input_tensor=Input(shape=(70,70,3))
x=input_tensor
x=Conv2D(filters=64,kernel_size=(3,3),activation='relu')(x)
x=BatchNormalization()(x)
x=Conv2D(filters=64,kernel_size=(3,3),activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=BatchNormalization()(x)
x=Dropout(0.1)(x)

x=Conv2D(filters=128,kernel_size=(3,3),activation='relu')(x)
x=BatchNormalization()(x)
x=Conv2D(filters=128,kernel_size=(3,3),activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=BatchNormalization()(x)
x=Dropout(0.1)(x)


x=Conv2D(filters=256,kernel_size=(3,3),activation='relu')(x)
x=BatchNormalization()(x)
x=Conv2D(filters=256,kernel_size=(3,3),activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=BatchNormalization()(x)
x=Dropout(0.1)(x)

x=GlobalAveragePooling2D()(x)
x=Dense(256,activation='relu')(x)
x=BatchNormalization()(x)
x=Dropout(0.5)(x)

x=Dense(256,activation='relu')(x)
x=BatchNormalization()(x)
x=Dropout(0.5)(x)

prediction=Dense(12,activation='softmax')(x)

model=Model(inputs=input_tensor,outputs=prediction)

lr_reducer=ReduceLROnPlateau(monitor='val_acc',
                             patience=3,verbose=1,factor=0.4,min_lr=1e-5)
filepath='weights.best_{epoch:02d}-{val_acc:2f}.hdf5'
checkpoint=ModelCheckpoint(filepath,monitor='val_acc',
                           verbose=1,save_best_only=True,mode='auto')

callbacks=[checkpoint,lr_reducer]

adam=Adam(lr=2*1e-3)
model.compile(optimizer=adam,loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_generator,train_generator.n/batch_size,
                    epochs=30,validation_data=vid_generator,validation_steps=10,
                    callbacks=callbacks)