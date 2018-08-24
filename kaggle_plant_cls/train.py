import numpy as np
from pre_data import get_data,preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import GlobalAveragePooling2D,Dropout,Dense,Average
from keras.models import Model
from keras.models import Input
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

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
    # preprocessing_function=preprocess_input)

vid_generator=vid_datagen.flow(vid,vid_label,batch_size=batch_size,
                                   shuffle=False)

x=Input(shape=(224,224,3))
base_model_1=ResNet50(include_top=False,input_tensor=x,weights='imagenet')
# base_model_2=InceptionV3(include_top=False,input_tensor=x,weights='imagenet')
# base_model_3=VGG19(include_top=False,input_tensor=x,weights='imagenet')

x_1=base_model_1.output
x_1=GlobalAveragePooling2D()(x_1)
x_1=Dense(1024,activation='relu')(x_1)
x_1=Dropout(0.5)(x_1)

# x_2=base_model_2.output
# x_2=GlobalAveragePooling2D()(x_2)
# x_2=Dense(1024,activation='relu')(x_2)
# x_2=Dropout(0.5)(x_2)

# x_3=base_model_3.output
# x_3=GlobalAveragePooling2D()(x_3)
# x_3=Dense(1024,activation='relu')(x_3)
# x_3=Dropout(0.5)(x_3)

predictions_1=Dense(12,activation='softmax')(x_1)
# predictions_2=Dense(12,activation='softmax')(x_2)
# predictions_3=Dense(12,activation='softmax')(x_3)

# predictions=Average()([predictions_1,predictions_2,predictions_3])

for layer in base_model_1.layers:
    layer.trainable=False
#
# for layer in base_model_2.layers:
#     layer.trainable=False

# for layer in base_model_3.layers:
#     layer.trainable=False

lr_reducer=ReduceLROnPlateau(monitor='val_acc',
                             patience=3,verbose=1,factor=0.4,min_lr=1e-5)
filepath='weights.best_{epoch:02d}-{val_acc:2f}.hdf5'
checkpoint=ModelCheckpoint(filepath,monitor='val_acc',
                           verbose=1,save_best_only=True,mode='auto')
filepath='weights,last_auto4.hdf5'
# checkpoint_all=ModelCheckpoint(filepath,monitor='val_acc',verbose=1,
#                                save_best_only=False,mode='auto')

callbacks=[checkpoint,lr_reducer]


model=Model(inputs=x,outputs=predictions_1)
adam=Adam(lr=2*1e-3)
model.compile(optimizer=adam,loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_generator,train_generator.n/batch_size,
                    epochs=30,validation_data=vid_generator,validation_steps=10,
                    callbacks=callbacks)