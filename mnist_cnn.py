import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 3, 3, 1], padding="SAME")


def cnn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_train = tf.reshape(mnist.train.images, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_train,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    print(h_pool1.get_shape())



def shape_test():
    a = tf.Variable(tf.random_uniform([1,4,4,1]))
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(a))
        print(sess.run(max_pool_2x2(a)))

if __name__ == '__main__':
    cnn()