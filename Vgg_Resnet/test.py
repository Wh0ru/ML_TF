import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from resnet import create_architecture

test_dir='test/'
IMG_W=224
IMG_H=224
BATCH_SIZE=20
CAPACITY=2000



def get_files():
    images=[test_dir+'{}.jpg'.format(i+1) for i in range(12500)]
    image_list=np.hstack(images)
    temp=np.array([image_list])
    temp=temp.transpose()

    image_list=list(temp[:,0])
    return image_list

def get_batch(image,image_W,image_H,batch_size,capacity):
    image=tf.cast(image,tf.string)

    input_queue=tf.train.slice_input_producer([image],shuffle=False)
    image_contents=tf.read_file(input_queue[0])
    image=tf.image.decode_jpeg(image_contents,channels=3)

    # image=tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image=tf.image.resize_images(image,[image_H,image_W],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image=tf.cast(image,tf.float32)
    # image=tf.image.per_image_standardization(image)
    image_batch=tf.train.batch([image],batch_size=batch_size,
                                           num_threads=1,capacity=capacity,)

    return image_batch

test=get_files()
test_batch=get_batch(test,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)


tfmodel='model/vgg16_iter_10.ckpt'

tfconfig=tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True

sess=tf.Session(config=tfconfig)

coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)

X=tf.placeholder(tf.float32,[None,224,224,3])

pred=create_architecture(X,'TEST',2)

saver=tf.train.Saver()

saver.restore(sess,tfmodel)
sess.run(tf.global_variables_initializer())


try:
    for i in range(625):
        if coord.should_stop():
            break
        xs=sess.run(test_batch)
        prediction=sess.run(pred,feed_dict={X:xs})

        dog_pre=prediction
        print(dog_pre)
        # dog_pre=np.maximum(0.005,dog_pre)
        # print(dog_pre)
        # dog_pre=np.minimum(0.995,dog_pre)
        # print(dog_pre)
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:coord.request_stop()

coord.join(threads)
sess.close()
    #
    # length=np.arange(len(dog_pre))+1
    # dog_arr=np.array((length,dog_pre))
    # dog=pd.DataFrame(dog_arr.T,columns=['id','label'])
    #
    # dog.to_csv('csv/dog_{:d}.csv'.format(i),index=False)

