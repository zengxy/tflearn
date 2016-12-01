import tensorflow as tf
import input_data
import collections
import numpy as np


conv_parmas = collections.namedtuple("conv_params",
                                     ["window_width",
                                      "window_height",
                                      "stride",
                                      "out_channel",
                                      "padding"])
max_pool_params = collections.namedtuple("max_pool_params",
                                         ["ksize",
                                          "stride",
                                          "padding"])


def inference(images,
              conv_1_params, max_pool_1_params,
              conv_2_params, max_pool_2_params,
              full_connected_units):
    with tf.name_scope("conv1"):
        W = tf.Variable(tf.truncated_normal([conv_1_params.window_width,
                                             conv_1_params.window_height,
                                             input_data.IMAGE_CHANNEL,
                                             conv_1_params.out_channel],
                                            stddev=0.1))
        b = tf.Variable(tf.zeros([conv_1_params.out_channel]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(images, W, conv_1_params.stride, conv_1_params.padding)
                             + b)

    with tf.name_scope("max_pool1"):
        h_max_pool_1 = tf.nn.max_pool(h_conv1,
                                      ksize=max_pool_1_params.ksize,
                                      strides=max_pool_1_params.stride,
                                      padding=max_pool_1_params.padding)

    with tf.name_scope("conv2"):
        W = tf.Variable(tf.truncated_normal([conv_2_params.window_width,
                                             conv_2_params.window_height,
                                             conv_1_params.out_channel,
                                             conv_2_params.out_channel],
                                            stddev=0.1))
        b = tf.Variable(tf.zeros([conv_2_params.out_channel]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_max_pool_1, W, conv_2_params.stride, conv_2_params.padding)
                             + b)

    with tf.name_scope("max_pool1"):
        h_max_pool_2 = tf.nn.max_pool(h_conv2,
                                      ksize=max_pool_2_params.ksize,
                                      strides=max_pool_2_params.stride,
                                      padding=max_pool_2_params.padding)

    h_max_pool_2_shape = h_max_pool_2.get_shape()
    flat_dim = np.prod(h_max_pool_2_shape[1:])
    h_max_pool_2_flat = tf.reshape(h_max_pool_2, shape=[-1, flat_dim])

    with tf.name_scope("fully_connected"):
        W = tf.Variable(tf.truncated_normal(shape=[flat_dim, full_connected_units],
                                            stddev=np.sqrt(flat_dim)))
        b = tf.Variable(tf.zeros([full_connected_units]))
        h_fc = tf.nn.relu(tf.mul(h_max_pool_2_flat, W) + b)

    with tf.name_scope('softmax_out'):
        W = tf.Variable(tf.truncated_normal(shape=[full_connected_units, input_data.LABEL_NUM],
                                            stddev=np.sqrt(full_connected_units)))
        b = tf.Variable(tf.zeros([input_data.LABEL_NUM]))
        logits = tf.mul(h_fc, W) + b
    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    if len(labels.get_shape()) == 1:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name="cross_entropy")
    else:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name="cross_entropy")
    loss = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
    return loss


def evaluation(logits, labels, k=1):
    correct_num = tf.nn.in_top_k(logits, labels, k)
    return tf.reduce_sum(tf.cast(correct_num, tf.int32))


def train(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op





if __name__ == '__main__':
    input_data.maybe_download_and_extract(FLAGS.data_dir)
