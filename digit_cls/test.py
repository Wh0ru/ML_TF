from net import Net
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from aug import Aug

weight_path = 'outputs/valloss0.0210_valacc0.9951_epoch49.hdf5'
weight_path_2 = 'outputs/valloss0.0219_valacc0.9950_epoch40.hdf5'
weight_path_3 = 'outputs/valloss0.0219_valacc0.9946_epoch36.hdf5'
new_weight_path = 'outputs/se_newweights.hdf5'
save_path='results/se_weight.csv'
TEST_CSV='test.csv'
RES_CSV='sample_submission.csv'
batchsize=1
height=28
width=28
channels=1
numclass=10


def get_new_weight():
    f1 = h5py.File(weight_path, 'r+')
    f2 = h5py.File(weight_path_2, 'r+')
    f3 = h5py.File(weight_path_3, 'r+')
    f4 = h5py.File(new_weight_path, 'r+')

    layer_names = [n.decode('utf8') for n in f1.attrs['layer_names']]

    for layer_name in layer_names:
        g1 = f1[layer_name]
        g2 = f2[layer_name]
        g3 = f3[layer_name]
        g4 = f4[layer_name]
        weight_names = [n.decode('utf8') for n in g1.attrs['weight_names']]
        for weight_name in weight_names:
            g4[weight_name][:] = (g1[weight_name][:] + g2[weight_name][:] + g3[weight_name][:]) / 3

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    print('SWA Finished!')


def get_test(df):
    xs = []
    for i in df:
        img = i.reshape(28, 28)
        img = img*1./255.
        xs.append(img)
    xs = np.array(xs, dtype=np.float32).reshape(-1, height, width, 1)
    return xs


get_new_weight()
model = Net((height, width, channels), numclass)
model.load_weights(new_weight_path, by_name=True)

df = pd.read_csv(TEST_CSV)
test = df.values

result = []
xs = get_test(test)
# print(xs.shape)

pre = model.predict(xs, batch_size=32)
# print(pre.shape)
pre = np.argmax(pre, axis=-1)
# print(pre.shape)
# result.append(pre)

resdf = pd.read_csv(RES_CSV)
resdf['Label'] = pre
resdf.to_csv(save_path, index=False)
