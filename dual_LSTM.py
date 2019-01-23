# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print (mnist.train.images.shape)
# 超参数设置
lr = 1e-3
# dropout: 每个元素被保留的概率
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])

input_size = 28
timestep_size = 28
hidden_size = 256
layer_num = 1
class_num = 10

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])

X = tf.reshape(_X, [-1, 28, 28])

def lstm1():
    stacked_rnn = []
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    stacked_rnn.append(lstm_cell)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    return mlstm_cell

def lstm2():
    stacked_rnn = []
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    stacked_rnn.append(lstm_cell)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    return mlstm_cell

init_state_single = lstm1().zero_state(batch_size, dtype=tf.float32)
outputs_single = list()
state_single = init_state_single
with tf.variable_scope('RNN_single'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state_single) = lstm1()(X[:, timestep, :], state_single)
        outputs_single.append(cell_output)
h_state_single = outputs_single[-1]

outputs_second = list()
init_state_second = lstm2().zero_state(batch_size, dtype=tf.float32)
with tf.variable_scope('RNN_second'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        
        (cell_output, init_state_second) = lstm2()(outputs_single[timestep], init_state_second)
        outputs_second.append(cell_output)
h_state_second = outputs_second[-1]

W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state_second, W) + bias)

cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))

train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
for i in range(5000):
    _batch_size = 128
    batch = mnist.train.next_batch(_batch_size)
    if (i+1)%200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            _X:batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
        print("Iter%d, step %d, training accuracy %g" % ( mnist.train.epochs_completed, (i+1), train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})

print("test accuracy %g"% sess.run(accuracy, feed_dict={
    _X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0]}))
