import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import keras
from pre_data import get_data,augment,get_test
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.utils import np_utils
from keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D,MaxPooling2D,Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

IMAGE_SIZE=28
EPOCHS=30
CHANNELS=1
BATCH_SIZE=128
nClasses=10
N_FLOD=5


def model():
    input_tensor=Input(shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS))

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='elu')(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='elu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), activation='elu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='elu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output=Dense(nClasses,activation='softmax')(x)

    model=Model(inputs=input_tensor,outputs=output)

    return model

def train_generator(train,label):
    while True:
        for start in range(0,len(train),BATCH_SIZE):
            end=min(start+BATCH_SIZE,len(train))
            X_tra=[]
            train_batch=train[start:end]
            for i in train_batch:
                i=augment(i,np.random.randint(6))
                X_tra.append(i)
            X_tra=np.array(X_tra,np.float32)/255.
            y_tra=np_utils.to_categorical(label[start:end],num_classes=10)
            yield X_tra,y_tra

def test_generator(test):
    while True:
        for start in range(0,len(test),BATCH_SIZE):
            end=min(start+BATCH_SIZE,len(test))
            test_batch=test[start:end]/255.
            yield test_batch

def train(x,y,kf,n_flod,model,test):
    preds_test=np.zeros(test.shape[0],dtype=np.float32)

    i=1
    for train_index, valid_index in kf.split(x):
        X_tra = x[train_index]
        X_val = x[valid_index]
        y_tra = y[train_index]
        y_val = y[valid_index]

        checkpoints=ModelCheckpoint('weights_best_'+str(i)+'.hdf5',monitor='val_acc',
                                    save_best_only=True,save_weights_only=True)
        lr_reducer=ReduceLROnPlateau(monitor='val_acc',factor=0.2,patience=3,
                                     verbose=1,min_lr=1e-7)
        callbacks=[checkpoints,lr_reducer]

        train_steps = len(X_tra)/BATCH_SIZE
        valid_steps = len(X_val) / BATCH_SIZE
        test_steps = len(test) / BATCH_SIZE

        model=model
        sgd=SGD(lr=2*1e-3,momentum=0.9,nesterov=True)
        model.compile(optimizer=sgd,loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit_generator(train_generator(X_tra,y_tra),
                            train_steps,epochs=EPOCHS,verbose=1,
                            validation_data=train_generator(X_val,y_val),
                            validation_steps=valid_steps,callbacks=callbacks)

        preds=model.predict_generator(test_generator(test),steps=test_steps,
                                      verbose=1)
        preds=np.argmax(preds,axis=1)

        preds_test+=preds
        i+=1

    preds_test/=n_flod
    return preds_test

kf=KFold(n_splits=N_FLOD,shuffle=True)


model=model()
xs,ys=get_data()
test=get_test()
test_pred=train(xs,ys,kf,N_FLOD,model,test)
df=pd.read_csv('sample_submission.csv')
df.drop('Label',axis=1,inplace=True)
df['Label']=test_pred
df.to_csv('result.csv',index=False)
