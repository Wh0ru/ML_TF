import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

FTRAIN = "training.csv"
FTEST  = "test.csv"
FIdLookup = 'IdLookupTable.csv'

def guassian_k(x0,y0,sigma,width,height):
    '''
    :return: (96,96)
    '''
    x=np.arange(0,width,1,float)
    y=np.arange(0,height,1,float)[:,np.newaxis]
    return np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma**2))


def generate_hm(height,width,landmarks,s=3):
    '''
    :return: (96,96,15)
    '''
    Nlandmarks=len(landmarks)
    hm=np.zeros((height,width,Nlandmarks),dtype=np.float32)
    for i in range(Nlandmarks):
        if not np.array_equal(landmarks[i],[-1,-1]):
            hm[:,:,i]=guassian_k(landmarks[i][0],
                                landmarks[i][1],
                                s,height,width)
        else:
            hm[:,:,i]=np.zeros((height,width))
    return hm


def get_y_as_heatmap(df, height, width, sigma):
    '''
    :return: (None,96,96,15)
    '''
    columns_lmxy = df.columns[:-1]
    columns_lm = []
    for c in columns_lmxy:
        c = c[:-2]
        if c not in columns_lm:
            columns_lm.extend([c])

    y_train = []
    for i in range(df.shape[0]):
        landmarks = []
        for colnm in columns_lm:
            x = df[colnm + "_x"].iloc[i]
            y = df[colnm + "_y"].iloc[i]
            if np.isnan(x) or np.isnan(y):
                x, y = -1, -1
            landmarks.append([x, y])

        y_train.append(generate_hm(height, width, landmarks, sigma))
    y_train = np.array(y_train)

    return (y_train, df[columns_lmxy], columns_lmxy)


def load(test=False, width=96, height=96, sigma=5):
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(fname)
    #将字符串转换为np数组
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    df = df.fillna(-1)
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)

    if not test:
        y, y0, nm_landmark = get_y_as_heatmap(df, height, width, sigma)
        X, y, y0 = shuffle(X, y, y0, random_state=42)
        y = y.astype(np.float32)
    else:
        y, y0, nm_landmark = None, None, None
    return X, y, y0, nm_landmark

def load2d(test=False,width=96,height=96,sigma=5):
    re=load(test,width,height,sigma)
    X=re[0].reshape(-1,width,height,1)
    y,y0,nm_landmarks=re[1:]
    return X,y,y0,nm_landmarks