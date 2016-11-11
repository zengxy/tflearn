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
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def cnn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # convolution 1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # max pool 1
    h_pool1 = max_pool_2x2(h_conv1)

    # convolution 2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = weight_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # max pool 2
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # full connect 1
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = weight_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # drop out
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出的softmax层
    w_softmax = weight_variable([1024, 10])
    b_softmax = bias_variable([10])

    y = tf.matmul(h_fc1_drop, w_softmax) + b_softmax
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))


    update = tf.train.MomentumOptimizer(0.01, 0.9).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(1000):
            if i % 50 == 0:
                print("Step: ", i,
                      "Accuracy: ", sess.run(accuracy,
                                             feed_dict={x: mnist.test.images,
                                                        y_: mnist.test.labels,
                                                        keep_prob: 1.0
                                                        })
                      )

            xs, ys = mnist.train.next_batch(100)
            sess.run(update,
                     feed_dict={x: xs, y_: ys, keep_prob: 0.8})
    print(h_pool1.get_shape())


def shape_test():
    a = tf.Variable(tf.random_uniform([1, 28, 28, 1]))
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # print(sess.run(a))
        # print(sess.run(max_pool_2x2(a)))
        print(max_pool_2x2(a).get_shape())


if __name__ == '__main__':
    cnn()
    # shape_test()
