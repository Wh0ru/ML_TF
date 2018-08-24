from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from dof_cls_pre_data import get_files
from keras.models import Model,Input
import keras.backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import cv2
import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from sklearn.model_selection import KFold
KFold.split()
import h5py

NB_CLASS=120
IM_WIDTH=224
IM_HEIGHT=224
batch_size=16
EPOCH=60
train_list=[]
vid_list=[]

train,label,vid,vid_label=get_files()

for i in train:
    img=cv2.imread(i)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(IM_WIDTH,IM_HEIGHT))
    train_list.append(img)
train_list=np.stack(train_list,axis=0)
train_label=keras.utils.np_utils.to_categorical(label,NB_CLASS)

for i in vid:
    img=cv2.imread(i)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(IM_WIDTH,IM_HEIGHT))
    vid_list.append(img)
vid_list=np.stack(vid_list,axis=0)
vid_label=keras.utils.np_utils.to_categorical(vid_label,NB_CLASS)


train_datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255,
)
train_generator=train_datagen.flow(train_list,train_label,batch_size=16,
                                   shuffle=False)

vid_datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255,
)
vid_generator=train_datagen.flow(vid_list,vid_label,batch_size=16,
                                   shuffle=False)


input_tensor=Input((IM_WIDTH,IM_HEIGHT,3))
base_model_1=InceptionV3(input_tensor=input_tensor,weights='imagenet',include_top=False)
base_model_2=ResNet50(input_tensor=input_tensor,weights='imagenet',include_top=False)
base_model_3=Xception(input_tensor=input_tensor,weights='imagenet',include_top=False)

x_1=base_model_1.output
x_1=GlobalAveragePooling2D()(x_1)

x_2=base_model_2.output
x_2=GlobalAveragePooling2D()(x_2)

x_3=base_model_3.output
x_3=GlobalAveragePooling2D()(x_3)


x=Concatenate()([x_1,x_2,x_3])
x=Dense(1024,activation='relu')(x)
x=Dropout(0.5)(x)
predictions=Dense(120,activation='softmax')(x)


for layer in base_model_1.layers:
    layer.trainable=False

for layer in base_model_2.layers:
    layer.trainable=False

for layer in base_model_3.layers:
    layer.trainable=False

filepath='weights.best_{epoch:02d}-{val_acc:2f}.h5'
checkpoint=ModelCheckpoint(filepath,monitor='val_acc',verbose=1,
                           save_best_only=True,mode='auto')
lr_reducer=ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,min_lr=1e-5)
callbacks=[lr_reducer,checkpoint]

model=Model(inputs=input_tensor,outputs=predictions)

adam=Adam(lr=2*1e-3)
model.compile(optimizer=adam,loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_generator,steps_per_epoch=train_generator.n/batch_size,
                    epochs=EPOCH,shuffle=False,callbacks=callbacks,validation_data=vid_generator,
                    validation_steps=10)