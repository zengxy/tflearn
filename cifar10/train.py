import os

import tensorflow as tf
import numpy as np

import cnn as nn_structure
import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("model_dir",
                           os.path.join("../model/", nn_structure.__name__),
                           "Directory of model.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size")
tf.app.flags.DEFINE_integer("max_step", 10000, "Max iteration step")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
# tf.app.flags.DEFINE_integer("fully connected units", 128, "units num")

# 网络参数
conv_1_params = nn_structure.conv_parmas(window_height=5, window_width=5, stride=[1, 1, 1, 1],
                                         out_channel=10, padding="SAME")
max_pool_1_params = nn_structure.max_pool_params(ksize=[1, 4, 4, 1], stride=[1, 4, 4, 1], padding="SAME")

conv_2_params = conv_1_params
max_pool_2_params = max_pool_1_params

full_connected_units = 128


def place_holder(batch_size):
    image_placeholder = tf.placeholder(tf.float32,
                                       shape=[batch_size,
                                              input_data.IMAGE_WIDTH, input_data.IMAGE_HEIGHT,
                                              input_data.IMAGE_CHANNEL])
    label_placeholder = tf.placeholder(tf.int32, shape=[batch_size])
    return image_placeholder, label_placeholder


def fill_feed_dic(dataSet, image_placeholder, label_placeholder):
    images, labels = dataSet.next_batch(FLAGS.batch_size)
    images = np.reshape(images,
                        [FLAGS.batch_size, input_data.IMAGE_WIDTH, input_data.IMAGE_HEIGHT, input_data.IMAGE_CHANNEL])
    return {
        image_placeholder: images,
        label_placeholder: labels
    }


def run_training():
    train_data, test_data, validation_data = input_data.load_data()
    with tf.Graph().as_default():
        image_pl, label_pl = place_holder(FLAGS.batch_size)
        logits = nn_structure.inference(image_pl,
                                        conv_1_params, max_pool_1_params,
                                        conv_2_params, max_pool_2_params,
                                        full_connected_units)
        loss = nn_structure.loss(logits, label_pl)
        train_op = nn_structure.train(loss, FLAGS.learning_rate)
        init = tf.initialize_all_variables()
        eval = nn_structure.evaluation(logits, label_pl, k=3)

        with tf.Session() as sess:
            sess.run(init)
            filled_dict = fill_feed_dic(train_data, image_pl, label_pl)
            for i in range(FLAGS.max_step):
                _, loss_value, eval_value = sess.run([train_op, loss, eval], feed_dict=filled_dict)
                if i%100 == 0:
                    print(i, loss_value, eval_value)
                    # a = train_data.next_batch(FLAGS.batch_size)[1]
                    # print(a)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
