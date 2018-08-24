import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

def get_file(file_dir):
    images=[]
    label=[]
    root_path=os.listdir(file_dir)
    for ind,path in enumerate(root_path):
        image_path=os.path.join(file_dir,path)
        image=os.listdir(image_path)
        for im in image:
            images.append(os.path.join(image_path,im))
            label.append(ind)
    images_list=np.hstack(images)
    label_list=np.hstack(label)
    temp=np.array([images_list,label_list])
    temp=temp.transpose()
    np.random.shuffle(temp)

    train_size=int(0.8*len(images_list))

    train=list(temp[:train_size,0])
    vid=list(temp[train_size:,0])

    train_label=list(temp[:train_size,1])
    vid_label=list(temp[train_size:,1])

    train_label=[int(i)for i in train_label]
    vid_label=[int(i)for i in vid_label]

    return train,train_label,vid,vid_label

# train,train_label,vid,vid_label=get_file('flower_photos')


def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int64)

    input_queue=tf.train.slice_input_producer([image,label],shuffle=False)
    image_contents=tf.read_file(input_queue[0])
    label=input_queue[1]
    image=tf.image.decode_jpeg(image_contents,channels=3)

    image=tf.image.resize_images(image,[image_W,image_H],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image=tf.cast(image,tf.float32)

    image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,
                                           num_threads=64,capacity=capacity)
    label_batch=tf.reshape(label_batch,[batch_size])

    return image_batch,label_batch


# train,train_label,vid,vid_label=get_file('flower_photos')
# train_batch,label_batch=get_batch(train,train_label,224,224,16,2000)



# with tf.Session() as sess:
#     coord=tf.train.Coordinator()
#     threaeds=tf.train.start_queue_runners(sess=sess,coord=coord)
#
#     try:
#         for i in range(1):
#             if coord.should_stop():
#                 break
#             xs,ys=sess.run([train_batch,label_batch])
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threaeds)
#     sess.close()

