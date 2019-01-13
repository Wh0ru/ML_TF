from net import resnet18
from keras.utils import to_categorical
from keras.optimizers import Adam,SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import os
from sklearn.model_selection import train_test_split
from aug import Aug
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from net import Net
import matplotlib.pyplot as plt
import cv2
# np.random.seed(2019)

def callback():
    checkpoint=ModelCheckpoint(save_path,monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True)
    lrducer=ReduceLROnPlateau(monitor='val_acc',verbose=1,patience=3,min_lr=1e-7,factor=0.4)
    earlystop=EarlyStopping(monitor='val_acc',patience=10,verbose=1)
    callbacks=[checkpoint,lrducer,earlystop]
    return callbacks


def get_data(df, aug=True):
    while True:
        for start in range(0,len(df),batchsize):
            end=min(start+batchsize,len(df))
            xs=[]
            ys=[]
            train_df=df[start:end]
            for i in train_df:
                label=i[0]
                img=i[1:].reshape(28,28)
                if aug:
                    img=Aug(img)
                img=img*1./255.
                xs.append(img)
                ys.append(label)
            xs=np.array(xs,dtype=np.float32).reshape(-1,height,width,1)
            ys=np.array(ys)
            ys=to_categorical(ys,num_classes=numclass)
            yield xs,ys

def dt(df):
    xs=[]
    ys=[]
    for i in df:
        img=i[1:].reshape(28,28)
        label=i[0]
        img=img*1./255.
        xs.append(img)
        ys.append(label)
    xs=np.array(xs,dtype=np.float32).reshape(-1,height,width,1)
    ys=np.array(ys)
    ys=to_categorical(ys,num_classes=numclass)
    return xs,ys

TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'
epochs=100
batchsize=32
height=28
width=28
channels=1
numclass=10
save_path='outputs/valloss{val_loss:.4f}_valacc{val_acc:.4f}_epoch{epoch:02d}.hdf5'
new_weight_path = 'outputs/new_weights.hdf5'

video_file = 1
cap = cv2.VideoCapture(video_file)

df=pd.read_csv(TRAIN_CSV)
data=df.values
np.random.shuffle(data)
tindx=int(0.8*len(data))

# train,labels=dt(data)
# trainX,vaildX,trainY,vaildY=train_test_split(train,labels,test_size=0.2,random_state=2019)

model=Net((height,width,channels),numclass)
adam=Adam(lr=2*1e-4)
callbacks=callback()
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['acc'])
H=model.fit_generator(get_data(data[:tindx]),tindx//batchsize,epochs=epochs,validation_data=get_data(data[tindx:]),
                    validation_steps=(len(data)-tindx)//batchsize,callbacks=callbacks)
#
# H=model.fit(trainX,trainY,batch_size=batchsize,epochs=epochs,validation_data=(vaildX,vaildY),shuffle=False)

# model = Net((height, width, channels), numclass)
# model.load_weights(new_weight_path, by_name=True)
#
# print('[INFO] evaluating network...')
# pred = model.predict(vaildX, batchsize)
# print(classification_report(vaildY.argmax(axis=1),
#                             pred.argmax(axis=1)))

# xs,ys=next(get_data(data[:tindx]))
# im=xs[0]
# print(xs.shape)
# print(ys.shape)
# plt.imshow(np.squeeze(im))
# plt.show()

