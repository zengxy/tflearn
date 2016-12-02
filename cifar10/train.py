import os
import pprint

import tensorflow as tf
import numpy as np

import cnn as nn_structure
# import input_data
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("model_dir",
                           os.path.join("../model/", nn_structure.__name__),
                           "Directory of model.")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size")
tf.app.flags.DEFINE_integer("max_step", 10000, "Max iteration step")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
# tf.app.flags.DEFINE_integer("fully connected units", 128, "units num")
tf.app.flags.DEFINE_integer("IMAGE_WIDTH", 28, "image width")
tf.app.flags.DEFINE_integer("IMAGE_HEIGHT", 28, "image height")
tf.app.flags.DEFINE_integer("IMAGE_CHANNEL", 1, "image channel")
tf.app.flags.DEFINE_integer("LABEL_NUM", 10, "label number")

# 网络参数
conv_1_params = nn_structure.conv_parmas(window_height=5, window_width=5, stride=[1, 1, 1, 1],
                                         out_channel=32, padding="SAME")
max_pool_1_params = nn_structure.max_pool_params(ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding="SAME")

conv_2_params = conv_1_params
max_pool_2_params = max_pool_1_params

full_connected_units = 1024


def place_holder(batch_size):
    image_placeholder = tf.placeholder(tf.float32,
                                       shape=[batch_size,
                                              FLAGS.IMAGE_WIDTH, FLAGS.IMAGE_HEIGHT,
                                              FLAGS.IMAGE_CHANNEL])
    label_placeholder = tf.placeholder(tf.int32, shape=[batch_size])
    keep_prob = tf.placeholder(tf.float32)
    return image_placeholder, label_placeholder, keep_prob


def fill_feed_dic(dataSet, image_placeholder, label_placeholder):
    images, labels = dataSet.next_batch(FLAGS.batch_size)
    images = np.reshape(images,
                        [FLAGS.batch_size, FLAGS.IMAGE_WIDTH, FLAGS.IMAGE_HEIGHT, FLAGS.IMAGE_CHANNEL])
    return {
        image_placeholder: images,
        label_placeholder: labels
    }


def run_training():
    train_data, test_data, validation_data = input_data.read_data_sets("../data/MNIST_data/")
    with tf.Graph().as_default():
        image_pl, label_pl, keep_prob_pl = place_holder(FLAGS.batch_size)
        logits = nn_structure.inference(image_pl,
                                        conv_1_params, max_pool_1_params,
                                        conv_2_params, max_pool_2_params,
                                        full_connected_units,
                                        keep_prob_pl)
        loss = nn_structure.loss(logits, label_pl)
        train_op = nn_structure.train(loss, FLAGS.learning_rate)
        init = tf.initialize_all_variables()
        eval = nn_structure.evaluation(logits, label_pl, k=1)
        prediction = nn_structure.predict(logits)

        with tf.Session() as sess:
            sess.run(init)
            for i in range(FLAGS.max_step):
                filled_dict = fill_feed_dic(train_data, image_pl, label_pl)
                filled_dict[keep_prob_pl] = 0.5
                _, loss_value, eval_value = sess.run([train_op, loss, eval], feed_dict=filled_dict)
                if i % 100 == 0:
                    print(i, loss_value, eval_value)
                    filled_dict = fill_feed_dic(test_data, image_pl, label_pl)
                    filled_dict[keep_prob_pl] = 1.0
                    prediction_value = sess.run(prediction, filled_dict)
                    pprint.pprint(prediction_value)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
