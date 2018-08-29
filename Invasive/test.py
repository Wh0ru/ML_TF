from keras.models import load_model
from pre_data import load_test
from keras.preprocessing.image import ImageDataGenerator
import cv2
import pandas as pd
import numpy as np

test_path='test/'
test_files,test_set=load_test(test_path)

X_test=[]
IMAGE_SIZE=224
CHANNELS=3
EPOCHES=60
BATCH_SIZE=16

for i in test_files:
    img=cv2.imread(i)
    img=cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
    X_test.append(img)
X_test=np.array(X_test)

test_datagen=ImageDataGenerator(
    rescale=1./255.
)

test_generator=test_datagen.flow(X_test,batch_size=BATCH_SIZE,shuffle=False)
model=load_model('weights_best_04.hdf5')
pred=model.predict_generator(test_generator,steps=test_generator.n/BATCH_SIZE,verbose=1)[:,-1]

df=pd.read_csv('sample_submission.csv')
df.drop('invasive',axis=1,inplace=True)
df['invasive']=pred
df.to_csv('result_inceptionv3.csv',index=False)
