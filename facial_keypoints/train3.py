from keras.models import Model
from keras.models import Input
import cv2
import numpy as np
from skimage import transform
from skimage.transform import SimilarityTransform, AffineTransform

from sklearn import metrics
from keras.optimizers import Adam
from guassian import load2d
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D,Conv2D,MaxPooling2D,Conv2DTranspose,Reshape
from keras.regularizers import l2
from weights import flatten_except_1dim,find_weight
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,TensorBoard
from pre_data import get_data
from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE=96
kernel_size=24
BATCH_SIZE=64
CHANNELS=1
EPOCHS=3000
n=160
constant=10
sigma=5
X_train, y_train, y_train0, nm_landmarks = load2d(test=False,sigma=sigma)
prop_train=0.9
Ntrain=int(X_train.shape[0]*prop_train)
X_tra, y_tra, X_val,y_val = X_train[:Ntrain],y_train[:Ntrain],X_train[Ntrain:],y_train[Ntrain:]
del X_train,y_train

def transform_img(data,
                  max_rotation=0.01,
                  max_shift=2,
                  max_shear=0,
                  max_scale=0.01, mode="edge"):

    scale = (np.random.uniform(1 - max_scale, 1 + max_scale),
             np.random.uniform(1 - max_scale, 1 + max_scale))
    rotation_tmp = np.random.uniform(-1 * max_rotation, max_rotation)
    translation = (np.random.uniform(-1 * max_shift, max_shift),
                   np.random.uniform(-1 * max_shift, max_shift))
    shear = np.random.uniform(-1 * max_shear, max_shear)
    tform = AffineTransform(
        scale=scale,
        rotation=np.deg2rad(rotation_tmp),
        translation=translation,
        shear=np.deg2rad(shear)
    )

    data=transform.warp(data, tform, mode=mode)
    return data

def get_batch(X_tra, y_tra):
    while True:
        for start in range(0,len(X_tra),BATCH_SIZE):
            x_batch=[]
            y_batch=[]
            w_batch=[]
            end=min(start+BATCH_SIZE,len(X_tra))
            train_batch=X_tra[start:end]
            label_batch=y_tra[start:end]
            for i in train_batch:
                x_batch.append(transform_img(i))
            for i in label_batch:
                y_batch.append(transform_img(i))
            w_batch.append(find_weight(y_tra[start:end]))
            x_batch=np.array(x_batch)
            y_batch=np.array(y_batch)
            w_batch=np.vstack(w_batch)
            yield x_batch,y_batch,w_batch

img_input=Input(shape=(IMAGE_SIZE,IMAGE_SIZE,1))

x=Conv2D(64,(3,3),activation='relu',padding='same')(img_input)
x=Conv2D(64,(3,3),activation='relu',padding='same')(x)
x=BatchNormalization()(x)
x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)


x=Conv2D(128,(3,3),activation='relu',padding='same')(x)
x=Conv2D(128,(3,3),activation='relu',padding='same')(x)
x=BatchNormalization()(x)
x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)


x=Conv2D(n,(kernel_size,kernel_size),activation='relu',padding='same')(x)
x=Conv2D(n,(1,1),activation='relu',padding='same')(x)
x=BatchNormalization()(x)


x=Conv2DTranspose(15,kernel_size=(4,4),strides=(4,4),use_bias=False)(x)
output=Reshape((-1,1))(x)


model=Model(inputs=img_input,outputs=output)

# print(model.summary())

checkpoints=ModelCheckpoint('weights.best_{epoch:02d}.hdf5',monitor='val_acc',
                        verbose=1,save_best_only=True)

lr_reducer=ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,
                             factor=0.4,min_lr=1e-7)

callbacks=[checkpoints,lr_reducer]

# adam=Adam(lr=2*1e-3)
model.compile(optimizer='rmsprop',loss='mse',sample_weight_mode='temporal',metrics=['accuracy'])

for i in range(EPOCHS):
    xs,ys,ws=next(get_batch(X_tra,y_tra))
    xs_val,ys_val,ws_val = next(get_batch(X_val, y_val))
    w_batch_fla=flatten_except_1dim(ws,ndim=2)
    y_batch_fla=flatten_except_1dim(ys,ndim=3)
    w_val_fla=flatten_except_1dim(ws_val,ndim=2)
    y_val_fla=flatten_except_1dim(ys_val,ndim=3)
    model.fit(xs,
              y_batch_fla*constant,
              sample_weight=w_batch_fla,
              validation_data=(xs_val,y_val_fla*constant,w_val_fla),
              batch_size=BATCH_SIZE,
              epochs=1,callbacks=callbacks)