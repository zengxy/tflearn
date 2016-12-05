import tensorflow as tf
import collections
import numpy as np

FLAGS = tf.app.flags.FLAGS

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
              full_connected_units,
              keep_prob):
    # cifar-10数据的原始格式为 red(1024) + blue(1024) + green(1024), batch为NCHW格式
    images = tf.reshape(images, [-1, FLAGS.IMAGE_CHANNE, FLAGS.IMAGE_HEIGHT, FLAGS.IMAGE_WIDTH])
    # transpose为NHWC格式为:(red, blue, green)*1024
    images = tf.transpose(images, [0, 2, 3, 1])
    with tf.name_scope("conv_1"):
        W = tf.Variable(tf.truncated_normal([conv_1_params.window_width,
                                             conv_1_params.window_height,
                                             FLAGS.IMAGE_CHANNEL,
                                             conv_1_params.out_channel],
                                            stddev=0.1),
                        name="weights")
        b = tf.Variable(tf.zeros([conv_1_params.out_channel]),
                        name="biases")
        h_conv_1 = tf.nn.relu(tf.nn.conv2d(images, W, conv_1_params.stride, conv_1_params.padding, data_format="NCHW")
                              + b)

    with tf.name_scope("max_pool_1"):
        h_max_pool_1 = tf.nn.max_pool(h_conv_1,
                                      ksize=max_pool_1_params.ksize,
                                      strides=max_pool_1_params.stride,
                                      padding=max_pool_1_params.padding)

    with tf.name_scope("conv_2"):
        W = tf.Variable(tf.truncated_normal([conv_2_params.window_width,
                                             conv_2_params.window_height,
                                             conv_1_params.out_channel,
                                             conv_2_params.out_channel],
                                            stddev=0.1),
                        name="weights")
        b = tf.Variable(tf.zeros([conv_2_params.out_channel]),
                        name="biases")
        h_conv_2 = tf.nn.relu(tf.nn.conv2d(h_max_pool_1, W, conv_2_params.stride, conv_2_params.padding)
                              + b)

    with tf.name_scope("max_pool_2"):
        h_max_pool_2 = tf.nn.max_pool(h_conv_2,
                                      ksize=max_pool_2_params.ksize,
                                      strides=max_pool_2_params.stride,
                                      padding=max_pool_2_params.padding)

    h_max_pool_2_shape = h_max_pool_2.get_shape()
    flat_dim = int(np.prod(h_max_pool_2_shape[1:]))
    h_max_pool_2_flat = tf.reshape(h_max_pool_2, shape=[-1, flat_dim])

    with tf.name_scope("fully_connected"):
        W = tf.Variable(tf.truncated_normal(shape=[flat_dim, full_connected_units],
                                            stddev=0.1),
                        name="weights")
        b = tf.Variable(tf.zeros([full_connected_units]),
                        name="biases")
        h_fc = tf.nn.relu(tf.matmul(h_max_pool_2_flat, W) + b)

    with tf.name_scope("drop_out"):
        h_fc_dropped = tf.nn.dropout(h_fc, keep_prob=keep_prob)

    with tf.name_scope('softmax_out'):
        W = tf.Variable(tf.truncated_normal(shape=[full_connected_units, FLAGS.LABEL_NUM],
                                            stddev=0.1),
                        name="weights")
        b = tf.Variable(tf.zeros([FLAGS.LABEL_NUM]), name="biases")
        logits = tf.matmul(h_fc_dropped, W) + b
    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    if len(labels.get_shape()) == 1:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name="cross_entropy")
    else:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name="cross_entropy")
    # mnist, learning=1e-4, batch_size=50, reduce_sum比reduce_mean好太多
    loss = tf.reduce_sum(cross_entropy, name="cross_entropy_mean")
    return loss


def evaluation(logits, labels, k=1):
    correct_num = tf.nn.in_top_k(logits, labels, k)
    return tf.reduce_sum(tf.cast(correct_num, tf.int32))


def train(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def predict(logits):
    prediction = tf.arg_max(logits, dimension=1)
    return prediction


if __name__ == '__main__':
    pass
