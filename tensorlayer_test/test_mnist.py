import tensorlayer as tl
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("training_data_dir", "../data/MNIST_data", "Directory of training data.")


def test():
    (train_data, validation_data, test_data) = \
        input_data.read_data_sets(FLAGS.training_data_dir, fake_data=FLAGS.fake_data)

    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    network = tl.layers.ReshapeLayer(x, [-1, 28, 28, 1])
    network = tl.layers.InputLayer(x)
    network = tl.layers.Conv2d(network, 32, act=tf.nn.relu, name="conv2d_1")
    network = tl.layers.MaxPool2d(network, strides=[1, 1, 1, 1], name="max_pool_1")
    network = tl.layers.Conv2d(network, 64, act=tf.nn.relu, name="conv2d_2")
    network = tl.layers.MaxPool2d(network, strides=[1, 1, 1, 1], name="max_pool_2")
    network = tl.layers.FlattenLayer
