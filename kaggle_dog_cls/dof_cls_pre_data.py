import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def get_files():
    data=pd.read_csv(r'C:\xlxz\Tensorflow_practise\kaggle\dog\labels.csv')
    train_data=np.array(data)
    np.random.shuffle(train_data)
    train_list=train_data[:,0]
    label_list=train_data[:,1]
    label=set()
    labels={}
    count=0
    for i in label_list:
        if i not in label:
            labels[i]=count
            label.add(i)
            count+=1
    label=[]
    for i in label_list:
        label.append(labels[i])
    train=[]
    for i in train_list:
        train.append('C:/xlxz/Tensorflow_practise/kaggle/dog/train/'+i+'.jpg')
    rate=int(len(train)*0.8)
    trainX=train[:rate]
    trainY=label[:rate]
    vidX=train[rate:]
    vidY=label[rate:]
    return trainX,trainY,vidX,vidY


def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int64)

    input_queue=tf.train.slice_input_producer([image,label])
    image_contents=tf.read_file(input_queue[0])
    label=input_queue[1]
    image=tf.image.decode_jpeg(image_contents,channels=3)

    image=tf.image.resize_images(image,[image_H,image_W],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image=tf.cast(image,tf.float32)
    image_batch,label_batch=tf.train.batch([image,label],batch_size,
                                           num_threads=64,capacity=capacity)
    label_batch=tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch

# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 224
# IMG_H = 224
#
# image_list, label_list = get_files()
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     try:
#         while not coord.should_stop() and i < 1:
#             img, label = sess.run([image_batch, label_batch])
#
#             for j in np.arange(BATCH_SIZE):
#                 print("label: %d" % label[j])
#                 plt.imshow(img[j, :, :, :])
#                 plt.show()
#             i += 1
#     except tf.errors.OutOfRangeError:
#         print("done!")
#     finally:
#         coord.request_stop()
#     coord.join(threads)