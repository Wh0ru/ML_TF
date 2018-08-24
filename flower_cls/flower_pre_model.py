from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.resnet50 import ResNet50
import cv2
import keras
from keras.optimizers import Adam
from keras.layers import Dense,GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from keras import backend as K
import numpy as np
from keras.models import Input
from flower_pre_images import get_file

NB_CLASS=5
IM_WIDTH=224
IM_HEIGHT=224
batch_size=16
EPOCH=60
train_list=[]
vid_list=[]


train,train_label,vid,vid_label=get_file(r'C:\xlxz\Tensorflow_practise\flwoer\flower_photos')

for i in train:
    img=cv2.imread(i)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(IM_WIDTH,IM_HEIGHT))
    train_list.append(img)
train_list=np.stack(train_list,axis=0)
train_label=keras.utils.np_utils.to_categorical(train_label,5)


for i in vid:
    img=cv2.imread(i)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(IM_WIDTH,IM_HEIGHT))
    vid_list.append(img)
vid_list=np.stack(vid_list,axis=0)
vid_label=keras.utils.np_utils.to_categorical(vid_label,5)


train_datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255
)
train_generator=train_datagen.flow(train_list,train_label,batch_size=16,
                                   shuffle=False)


vid_datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255
)

vid_generator=train_datagen.flow(vid_list,vid_label,batch_size=16,
                                   shuffle=False)
input_tensor=Input((IM_WIDTH,IM_HEIGHT,3))
x=input_tensor

base_model_1=InceptionV3(input_tensor=x,weights='imagenet',include_top=False)
base_model_2=ResNet50(input_tensor=x,weights='imagenet',include_top=False)

x_1=base_model_1.output
x_1=GlobalAveragePooling2D()(x_1)
x_1=Dense(1024,activation='relu')(x_1)

x_2=base_model_2.output
x_2=GlobalAveragePooling2D()(x_2)
x_2=Dense(1024,activation='relu')(x_2)

predictions_1=Dense(5,activation='softmax')(x_1)
predictions_2=Dense(5,activation='softmax')(x_2)
predictions=keras.layers.Average()([predictions_1,predictions_2])

model=Model(inputs=x,outputs=predictions)

for layer in base_model_1.layers:
    layer.trainable=False

for layer in base_model_2.layers:
    layer.trainable=False

adam=Adam()

def lr_sch(epoch):
    if epoch<5:
        return 1e-3
    if 10<=epoch<15:
        return 1e-4

lr_scheduler=LearningRateScheduler(lr_sch)
lr_reducer=ReduceLROnPlateau(monitor='val_acc',factor=0.2,patience=5,
                             mode='auto',min_lr=1e-4)

callbacks=[lr_scheduler,lr_reducer]

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_generator,steps_per_epoch=train_generator.n/batch_size,epochs=EPOCH,
                    validation_data=vid_generator,validation_steps=10,
                    callbacks=callbacks)

# for i,layer in enumerate(base_model_1.layers):
#     print(i,layer.input)
#
# for i,layer in enumerate(base_model_2.layers):
#     print(i,layer.input)