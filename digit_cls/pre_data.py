import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

TRAIN_PATH='train.csv'
TEST_PATH='test.csv'

def augment(src,choice):
    if choice==0:
        np.rot90(src,1)
    if choice==1:
        src=np.flipud(src)
    if choice==2:
        src=np.rot90(src,2)
    if choice==3:
        src=np.fliplr(src)
    if choice==4:
        src=np.rot90(src,3)
    if choice==5:
        src=np.rot90(src,2)
        src=np.fliplr(src)
    return src

def get_data():
    df = pd.read_csv(TRAIN_PATH)
    X_train = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    y_train = df.iloc[:, 0].values
    return X_train,y_train

def get_test():
    df = pd.read_csv(TEST_PATH)
    test = df.iloc[:,:].values.reshape(-1, 28, 28, 1)
    return test