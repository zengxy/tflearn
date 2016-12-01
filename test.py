import tensorflow as tf
from tensorflow.models.image.mnist import convolutional
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials import mnist
from tensorflow.models.image.cifar10 import cifar10
import pprint


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


def test_of_softmax_cross_entropy_with_logits():
    x_input = tf.cast(tf.constant([[2, 3, 4, 5], [5, 5, 7, 8]]), "float")
    y_ = tf.cast(tf.constant([[0, 0, 1, 0], [0, 0, 1, 0]]), "float")

    # softmax and reduce_sum
    y = tf.nn.softmax(x_input)
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(x_input, y_)

    cross_entropy3 = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.constant([2, 2]))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for item in sess.run([y, y_ * tf.log(y), cross_entropy, cross_entropy2, cross_entropy3]):
            pprint.pprint(item)


def testOpOrder():
    a = tf.Variable(1)
    addOne = tf.add(a, tf.constant(1))
    addTwo = tf.add(a, tf.constant(2))
    updateOne = tf.assign(a, addOne)
    updateTwo = tf.assign(a, addTwo)

    # 输出为:
    # [2，2，2, 2]，[3, 3, 3, 3], [4, 4, 4, 4]
    # 计算值按照图统一调配, 结果上看计算方式为按照图的依赖关系
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # 输出为: [2，2，2, 2]，[3, 3, 3, 3], [4, 4, 4, 4]
        # 计算值按照图统一调配, 结果上看计算方式为按照图的依赖关系调度，a未加锁
        # print(sess.run([a, updateOne, updateTwo, a]))

        print(sess.run([a, updateOne, a]))
        print(sess.run([a, updateTwo, a]))


def test_read_cifar10():
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('my_train_dir', '../cifar10_model/model1',
                               """Directory where to write event logs """
                               """and checkpoint.""")
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.my_train_dir):
        tf.gfile.DeleteRecursively(FLAGS.my_train_dir)
    tf.gfile.MakeDirs(FLAGS.my_train_dir)
    with tf.Session() as sess:
        images, labels = cifar10.distorted_inputs()
        sess.run(tf.initialize_all_variables())
        a, b = sess.run([images, labels])
        print(len(a), len(a[0]))





if __name__ == '__main__':
    # test_read_cifar10()
    FLAGS = tf.app.flags.FLAGS
    import os
    tf.app.flags.DEFINE_string('my_train_dir',
                               os.path.join('../cifar10_model/model1', os.path.basename(__file__).split(".")[0]),
                               "1111")
    print(os.path.join('../cifar10_model/model1', os.path.basename(__file__).split(".")[0]))
    # mnist_data_set = input_data.py.read_data_sets("MNIST_data/", one_hot=True)
    # for i in range(20):
    #     print(mnist_data_set.test.next_batch(1000)[1][0])
    # test_of_softmax_cross_entropy_with_logits()
    # a = tf.constant([1,2,3])
    # with tf.Session() as sess:
    #     print(len(a.get_shape()))
    # def jj():
    #     return 1, 2, 3, 4, 5, 6
    # a, b, *_ = jj()
    # print(a, b, _)
    # softmaxClassify()
    # print(convolutional.data_type())
