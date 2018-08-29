import pandas as pd
import numpy as np

train_path='train/'
test_path='test/'

def load_train(path):
    train_set = pd.read_csv('train_labels.csv')
    train_label = np.array(train_set['invasive'])
    train_files = []
    for i in range(len(train_set)):
        train_files.append(path+str(int(train_set.iloc[i][0]))+'.jpg')
    train_set['name']=train_files
    return train_files,train_set,train_label


# train_files,train_set,train_label=load_train(train_path)

def load_test(path):
    test_set = pd.read_csv('sample_submission.csv')
    test_files = []
    for i in range(len(test_set)):
        test_files.append(path+str(int(test_set.iloc[i][0]))+'.jpg')
    return test_files,test_set


def centering_image(img):
    size = [256, 256]

    img_size = img.shape[:2]

    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized