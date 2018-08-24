import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from pre_image import get_files,get_batch
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
from test import get_test_files,get_test_batch
import pandas as pd

train_dir='train/'
IMG_W=224
IMG_H=224
BATCH_SIZE=16
CAPACITY=2000
LOGDIR='logs'

train,train_label=get_files(train_dir)
train_batch,train_label_batch=get_batch(train,train_label,
                                        IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

test=get_test_files()
test_batch=get_test_batch(test,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

num_layers=101
scope='resnet_v1_%d'%num_layers
fix_blocks=1

blocks=[resnet_v1_block('block1',64,3,stride=2),
        resnet_v1_block('block2',128,4,stride=2),
        resnet_v1_block('block3',256,23,stride=2),
        resnet_v1_block('block4',512,3,stride=1)]


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params={
        'is_training':is_training,
        'decay':batch_norm_decay,
        'epsilon':batch_norm_epsilon,
        'scale':batch_norm_scale,
        'trainable':False,
        'updates_collections':tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        trainable=is_training,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params):
            with slim.arg_scope([slim.max_pool2d],padding='SAME') as arg_sc:
                return arg_sc

def build_base(input_tensor):
    with tf.variable_scope(scope,scope):
        net=resnet_utils.conv2d_same(input_tensor,64,7,stride=2,scope='conv1')
        net=tf.pad(net,[[0,0],[1,1],[1,1],[0,0]])
        net=slim.max_pool2d(net,[3,3],stride=2,padding='VALID',scope='pool1')
    return net

def image_to_head(input_tensor,is_training,reuse=None):
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net_conv=build_base(input_tensor)
    if fix_blocks>0:
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net_conv,_=resnet_v1.resnet_v1(net_conv,blocks[0:fix_blocks],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=reuse,scope=scope)
    if fix_blocks<4:
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net_conv,_=resnet_v1.resnet_v1(net_conv,
                                           blocks[fix_blocks:],
                                           global_pool=True,
                                           include_root_block=False,
                                           reuse=reuse,
                                           scope=scope)
    return net_conv

def full_con(input_tensor,num_cls,is_training,initializer):
    cls_score=slim.fully_connected(input_tensor,num_cls,
                                   activation_fn=None,
                                   scope='cls_score',
                                   trainable=is_training,
                                   weights_initializer=initializer)
    cls_prob=tf.nn.softmax(cls_score,name='cls_prob')
    cls_prob=tf.reshape(cls_prob,(-1,num_cls))
    return cls_prob

def build_network(input_tensor,num_cls,is_training=True):
    initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01)
    fc7=image_to_head(input_tensor,is_training)
    cls_prob=full_con(fc7,num_cls,is_training,initializer)

    return cls_prob

def build_vgg_network(input_tensor,is_training,class_num):
    with tf.variable_scope('vgg_16','vgg_16'):
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

        net=build_vgg(input_tensor,is_training)
        cls_prob=build_predictions(net,is_training,initializer,class_num)

        return cls_prob

def build_vgg(input_tensor,is_training):
    net=slim.repeat(input_tensor,2,slim.conv2d,64,[3,3],trainable=False,scope='conv1')
    net=slim.max_pool2d(net,[2,2],padding='SAME',scope='pool1')

    net=slim.repeat(net,2,slim.conv2d,128,[3,3],trainable=False,scope='conv2')
    net=slim.max_pool2d(net,[2,2],padding='SAME',scope='pool2')

    net=slim.repeat(net,3,slim.conv2d,256,[3,3],trainable=is_training,scope='conv3')
    net=slim.max_pool2d(net,[2,2],padding='SAME',scope='pool3')

    net=slim.repeat(net,3,slim.conv2d,512,[3,3],trainable=is_training,scope='conv4')
    net=slim.max_pool2d(net,[2,2],padding='SAME',scope='pool4')

    net=slim.repeat(net,3,slim.conv2d,512,[3,3],trainable=is_training,scope='conv5')
    net=slim.max_pool2d(net,[2,2],padding='SAME',scope='pool5')

    return net

def build_predictions(net,is_training,initializer,class_num):
    net_flat=slim.flatten(net,scope='flatten')

    fc6=slim.fully_connected(net_flat,4096,scope='fc6')
    if is_training:
        fc6=slim.dropout(fc6,keep_prob=0.5,is_training=True,scope='dropout6')

    fc7=slim.fully_connected(fc6,4096,scope='fc7')
    if is_training:
        fc7=slim.dropout(fc7,keep_prob=0.5,is_training=True,scope='dropout7')

    cls_score=slim.fully_connected(fc7,class_num,weights_initializer=initializer,
                                   trainable=is_training,activation_fn=None,scope='cls_score')
    cls_prob=tf.nn.softmax(cls_score,name='cls_prob')

    return cls_prob

def create_architecture(X,mode,class_num,y_=None):

    training=mode=='TRAIN'

    weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)

    with arg_scope([slim.fully_connected],
                   weights_regularizer=weights_regularizer,
                   biases_initializer=tf.constant_initializer(0.0)):
        cls_prob1=build_network(X,class_num,training)
        cls_prob2=build_vgg_network(X,training,class_num)
        cls_prob=(cls_prob2+cls_prob1)/2

        if training:
            loss=add_loss(cls_prob,y_)
            ac=acc(cls_prob,y_)
            return loss,ac,cls_prob
        else:
            return cls_prob




def add_loss(cls_prob,y_):
    cross_entroy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_prob,
                                                                labels=y_)
    loss=tf.reduce_mean(cross_entroy)
    tf.summary.scalar('loss',loss)
    return loss

def acc(cls_prob,y_):
    current=tf.equal(tf.argmax(cls_prob,1),y_)
    accuracy=tf.reduce_mean(tf.cast(current,tf.float32))
    tf.summary.scalar('accuracy',accuracy)
    return accuracy

def get_variables_to_restore_1(variables, var_keep_dic):
    variables_to_restore=[]
    # variables_to_fix={}
    for v in variables:
        # if v.name==(scope+'/conv1/weights:0'):
        #     variables_to_fix[v.name]=v
        #     continue
        if v.name.split(':')[0] in var_keep_dic:
            variables_to_restore.append(v)
    return variables_to_restore

def get_variables_to_restore_2(variables,var_keep_dict):
    variables_to_restore=[]
    _variables_to_fix={}
    for v in variables:
        if v.name=='vgg_16/fc6/weights:0' or v.name=='vgg_16/fc7/weights:0':
            _variables_to_fix[v.name]=v
            continue
        if v.name.split(':')[0] in var_keep_dict:
            variables_to_restore.append(v)
    return variables_to_restore,_variables_to_fix

def get_variables_in_checkpoint_file(file_name):
    reader=pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map=reader.get_variable_to_shape_map()
    return var_to_shape_map

# def fix_variables(sess,pre_model,variables_to_fix):
#     with tf.variable_scope('Fix_Resnet_V1'):
#         with tf.device('/cpu:0'):
#             conv1_rgb=tf.get_variable('conv1_rgb',[7,7,3,64],trainable=False)
#             restore_fc=tf.train.Saver({scope+'/conv1/weights':conv1_rgb})
#             restore_fc.restore(sess,pre_model)
#
#             sess.run(tf.assign(variables_to_fix[scope+'/conv1/weights:0'],
#                                tf.reverse(conv1_rgb,[2])))

def fix_variables(sess,pre_model,_variables_to_fix):
    with tf.variable_scope('Fix_VGG16'):
        with tf.device('/cpu:0'):
            fc6_conv=tf.get_variable('fc6_conv',[7,7,512,4096],trainable=False)
            fc7_conv=tf.get_variable('fc7_conv',[1,1,4096,4096],trainable=False)
            restorer_fc=tf.train.Saver({'vgg_16/fc6/weights':fc6_conv,
                                        'vgg_16/fc7/weights':fc7_conv,})
            restorer_fc.restore(sess,pre_model)
            sess.run(tf.assign(_variables_to_fix['vgg_16/fc6/weights:0'],
                               tf.reshape(fc6_conv, _variables_to_fix['vgg_16/fc6/weights:0'].get_shape())))
            sess.run(tf.assign(_variables_to_fix['vgg_16/fc7/weights:0'],
                               tf.reshape(fc7_conv, _variables_to_fix['vgg_16/fc7/weights:0'].get_shape())))

def train():
    num_cls=2
    pre_model1=r'C:\xlxz\Work\Faster-RCNN-TensorFlow-Python3.5-master\data\imagenet_weights\resnet_v1_101.ckpt'
    pre_model2=r'C:\xlxz\Work\Faster-RCNN-TensorFlow-Python3.5-master\data\imagenet_weights\vgg16.ckpt'
    tfconfig=tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess=tf.Session(config=tfconfig)

    X=tf.placeholder(tf.float32,[None,224,224,3])
    y_=tf.placeholder(tf.int64,[None])

    with sess.graph.as_default():
        loss,acc,cls_score=create_architecture(X,'TRAIN',num_cls,y_)
        lr=tf.Variable(0.001,trainable=False)
        momentum=0.9
        train_op=tf.train.MomentumOptimizer(lr,momentum).minimize(loss)

    variales=tf.global_variables()
    sess.run(tf.variables_initializer(variales,name='init'))
    var_keep_dic1=get_variables_in_checkpoint_file(pre_model1)
    var_keep_dic2=get_variables_in_checkpoint_file(pre_model2)
    variales_to_restore1=get_variables_to_restore_1(variales,var_keep_dic1)
    variales_to_restore2,variales_to_fix=get_variables_to_restore_2(variales,var_keep_dic2)
    restorer1=tf.train.Saver(variales_to_restore1)
    restorer1.restore(sess,pre_model1)
    restorer2= tf.train.Saver(variales_to_restore2)
    restorer2.restore(sess, pre_model2)
    fix_variables(sess,pre_model2,variales_to_fix)

    saver=tf.train.Saver(max_to_keep=4)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    merged=tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)

    try:
        for epoch in range(30000):
            if coord.should_stop():
                break
            xs,ys=sess.run([train_batch,train_label_batch])
            summary,_=sess.run([merged,train_op], feed_dict={X:xs, y_: ys})
            train_writer.add_summary(summary,epoch)
            if epoch%20==0:
                loss_val,acc_val,score=sess.run([loss,acc,cls_score],feed_dict={X:xs,y_:ys})
                print('loss:', loss_val, 'accuracy:', acc_val)
                print(score)
            if epoch%15==0 and epoch!=0:
                for j in range(800):
                    x_s=sess.run(test_batch)
                    score=sess.run([cls_score], feed_dict={X:x_s})
                    dog_pre=score[:,1]
                    dog_pre = np.maximum(0.005, dog_pre)
                    dog_pre = np.minimum(0.995, dog_pre)
                    length=np.arange(len(dog_pre))+1
                    dog_arr=np.array((length,dog_pre))
                    dog=pd.DataFrame(dog_arr.T,columns=['id','label'])
                    dog.to_csv('csv/dog_{:d}.csv'.format(j),index=False)

                # saver.save(sess,'model/vgg16_iter_{:d}'.format(epoch)+'.ckpt')
                # print('save model!')
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:coord.request_stop()

    coord.join(threads)
    sess.close()

train()






