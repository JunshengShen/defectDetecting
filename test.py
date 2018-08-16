import os
import tensorflow as tf
import glob
from skimage import io,transform
import numpy as np
from matplotlib import pyplot as plt


def weight_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.1), name='weight')


def conv_2d(x, w):
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 2, 2, 1], padding="VALID")


def evaluate(y, y_):
    y = tf.arg_max(input=y, dimension=1)
    y_ = tf.arg_max(input=y_, dimension=1)
    return tf.reduce_mean(input_tensor=tf.cast(tf.equal(y, y_), tf.float32))


def bias_variable(shape):
    return tf.Variable(initial_value=tf.constant(value=0.1, shape=shape), name='bias')


regularizer = tf.contrib.layers.l2_regularizer(0.0001)
path = './'
w = 1024
h = 1024
c = 3


def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    i = 0
    for folder in cate:
        if folder == ['./.idea']or folder==['./venv'] or folder==['./model']:
            del cate[i]
            continue
        i += 1
    imgs = []
    labels = []
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img = io.imread(im)
            img = transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx-3)#傻逼玩意

    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)


data,label = read_img(path)
#plt.imshow(data[0], cmap='gray')
#plt.imshow(data[1], cmap='gray')
#plt.imshow(data[2], cmap='gray')
print(label)


num_example = data.shape[0]
ratio = 0.8
num_train = int(ratio * num_example)
s = np.int(num_train)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]


x = tf.placeholder(tf.float32,shape=[num_train,w,h,c],name='x')
y_ = tf.placeholder(tf.int32,shape=[num_train,],name='y_')

w_convX1 = weight_variable(shape=[2, 2, 3, 9])
b_convX1 = bias_variable(shape=[9])
convX1_out = tf.nn.relu(conv_2d(x,w_convX1)+b_convX1)
convX1_out = tf.nn.max_pool(convX1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

w_convX2 = weight_variable(shape=[2, 2, 9, 27])
b_convX2 = bias_variable(shape=[27])
convX2_out = tf.nn.relu(conv_2d(convX1_out,w_convX2)+b_convX2)
convX2_out = tf.nn.max_pool(convX2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

w_convX3 = weight_variable(shape=[2, 2, 27, 54])
b_convX3 = bias_variable(shape=[54])
convX3_out = tf.nn.relu(conv_2d(convX2_out,w_convX3)+b_convX3)
convX3_out = tf.nn.max_pool(convX3_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

w_convX4 = weight_variable(shape=[2, 2, 54, 108])
b_convX4 = bias_variable(shape=[108])
convX4_out = tf.nn.relu(conv_2d(convX3_out,w_convX4)+b_convX4)
convX4_out = tf.nn.max_pool(convX4_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

w_convX5 = weight_variable(shape=[2, 2, 108, 216])
b_convX5 = bias_variable(shape=[216])
convX5_out = tf.nn.relu(conv_2d(convX4_out,w_convX5)+b_convX5)

w_convX6 = weight_variable(shape=[2, 2, 216, 432])
b_convX6 = bias_variable(shape=[432])
convX6_out = tf.nn.relu(conv_2d(convX5_out,w_convX6)+b_convX6)
convX6_out = tf.reshape(convX6_out,[-1,432])


fc1_weights = tf.get_variable("weight", [432, 256],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
tf.add_to_collection('losses', regularizer(fc1_weights))
fc1_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.1))

fc1 = tf.nn.relu(tf.matmul(convX6_out, fc1_weights) + fc1_biases)
fc1 = tf.nn.dropout(fc1, 0.5)


fc2_weights = tf.get_variable("weight1", [256, 2],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
tf.add_to_collection('losses', regularizer(fc2_weights))
fc2_biases = tf.get_variable("bias1", [2], initializer=tf.constant_initializer(0.1))
logit = tf.matmul(fc1, fc2_weights) + fc2_biases
#print(logit)
print(logit)
#print(y_)
Loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=y_)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(Loss)
initialized_variables = tf.initialize_all_variables()
sess = tf.Session()
sess.run(fetches=initialized_variables)
saver = tf.train.Saver()
#saver.restore(sess,"/defectDetecting/model/1.ckpt")
for times in range(5):
    for iter in range(1):
        # batch = dataset.train.next_batch(batch_size=batch_size)
        # sess.run(fetches=Step_train, feed_dict={x:batch[0], y:batch[1], dropout_prob:0.5})
        # Accuracy = sess.run(fetches=accuracy, feed_dict={x:batch[0], y:batch[1], dropout_prob:1})
        # print('Iter num %d ,the train accuracy is %.3f' % (iter+1, Accuracy))
        a = 0
        for i in range(1200):
            sess.run(fetches=train_op, feed_dict={x: x_train.reshape((num_train, 1024, 1024, 3)),
                                                    y_: y_train})

            a = sess.run(fetches=logit, feed_dict={x: x_train.reshape((num_train, 1024, 1024, 3)),
                                                    y_: y_train})
            a = a.tolist()
            print(iter + 1, a)
            print(a[0])
            print(a[0].index(max(a[0])))
        save_path = saver.save(sess, "/defectDetecting/model/" + str(times + 1) + ".ckpt")
