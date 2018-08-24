from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def preprocess_input(x):
    blurImg=cv2.GaussianBlur(x,(5,5),0)
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsvImg, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    bMask = mask > 0

    clear = np.zeros_like(x, np.float32)
    clear[bMask] = x[bMask]
    clear=clear/255
    return clear

def get_data():
    ScaleTo=224

    path='train/*/*.png'
    files=glob(path)

    trainImg=[]
    trainLabel=[]

    for i in files:
        trainImg.append(cv2.resize(cv2.imread(i),(ScaleTo,ScaleTo)))
        trainLabel.append(i.split('\\')[-2])

    trainImg=np.array(trainImg)
    trainLabel=pd.DataFrame(trainLabel)

    clearTrainImg=[]

    for img in trainImg:
        blurImg=cv2.GaussianBlur(img,(5,5),0)
        hsvImg=cv2.cvtColor(blurImg,cv2.COLOR_BGR2HSV)
        lower_green = (25, 40, 50)
        upper_green = (75, 255, 255)
        mask=cv2.inRange(hsvImg,lower_green,upper_green)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        bMask=mask>0

        clear=np.zeros_like(img,np.uint8)
        clear[bMask]=img[bMask]
        clearTrainImg.append(clear)

    clearTrainImg=np.array(clearTrainImg)
    clearTrainImg=clearTrainImg/255

    le=preprocessing.LabelEncoder()
    encoder=le.fit_transform(trainLabel[0])
    clearTrainLabel=np_utils.to_categorical(encoder)
    # trainLabel = np_utils.to_categorical(encoder)

    trainX,vidX,trainY,vidY=train_test_split(clearTrainImg,clearTrainLabel,
                                               test_size=0.1,shuffle=True)

    return trainX,trainY,vidX,vidY

def get_test():
    ScaleTo=224

    path='test/*.png'
    files=glob(path)

    trainImg=[]

    for i in files:
        trainImg.append(cv2.resize(cv2.imread(i),(ScaleTo,ScaleTo)))

    trainImg=np.array(trainImg)

    return trainImg