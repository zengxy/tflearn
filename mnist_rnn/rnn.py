import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# params
data_dir = "../data/MNIST_data"
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# network params
n_input = 28
n_step = 28
n_hidden = 128
n_classes = 10

# placeholder
x = tf.placeholder(tf.float32, [None, n_step, n_input])
y = tf.placeholder(tf.int32, [None])

weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))

biases = tf.Variable(tf.random_normal([n_classes]))


def RNN(x, weights, biases):
    # [batch_size, n_step, n_input] -> [n_step, batch_size, n_input]
    x = tf.transpose(x, [1, 0, 2])
    # reshape to [batch_size*n_step, n_input]
    x = tf.reshape(x, [-1, n_input])
    # split
    x = tf.split(0, n_step, x)

    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    out_puts, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(out_puts[-1], weights) + biases


pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.nn.in_top_k(pred, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init_op = tf.initialize_all_variables()

mnist_data = input_data.read_data_sets(data_dir)

with tf.Session() as sess:
    sess.run(init_op)
    step = 1
    while step*batch_size<training_iters:
        batch_x, batch_y = mnist_data.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, [batch_size, n_step, n_input])
        sess.run(optimizer, feed_dict={x: batch_x,
                                       y: batch_y})
        if step%display_step == 0:
            acc_val, loss_val = sess.run([accuracy, cost], feed_dict={x: batch_x,
                                                                  y: batch_y})
            print("Iter: {:d}, acc: {:.3f}, loss:{:.2f}".format(step*batch_size, acc_val, loss_val))
        step+=1
    print("Training Finished")

    test_x = mnist_data.test.images
    test_x = np.reshape(test_x, [-1, n_step, n_input])
    test_y = mnist_data.test.labels
    test_acc_val = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
    print("Test Accuracy:{:3f}".format(test_acc_val))

