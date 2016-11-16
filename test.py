import tensorflow as tf
import numpy as np
from tensorflow.models.image.mnist import convolutional
from tensorflow.examples.tutorials.mnist import input_data

def test1():
    convolutional.main()

def softmaxClassify():
    W = tf.Variable(tf.random_uniform([784, 10], -1., 1.))
    b = tf.Variable(tf.zeros([10]))

    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(5000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            if i % 50 == 0:
                print("Step: ", i,
                      "Accuracy: ", sess.run(accuracy,
                                             feed_dict={x: mnist.test.images,
                                                        y_: mnist.test.labels})
                      )

if __name__ == '__main__':
    test1()
    # softmaxClassify()
    # print(convolutional.data_type())







