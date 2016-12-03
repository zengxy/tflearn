import os
import time

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import input_data

import cnn as nn_structure

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("model_dir",
                           os.path.join("../model/", nn_structure.__name__),
                           "Directory of model.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size")
tf.app.flags.DEFINE_integer("max_step", 50000, "Max iteration step")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
# tf.app.flags.DEFINE_integer("fully connected units", 128, "units num")
tf.app.flags.DEFINE_integer("IMAGE_WIDTH", 32, "image width")
tf.app.flags.DEFINE_integer("IMAGE_HEIGHT", 32, "image height")
tf.app.flags.DEFINE_integer("IMAGE_CHANNEL", 3, "image channel")
tf.app.flags.DEFINE_integer("IMAGE_PIXES", 32*32*3, "pixes number")
tf.app.flags.DEFINE_integer("LABEL_NUM", 10, "label number")

# 网络参数
conv_1_params = nn_structure.conv_parmas(window_height=5, window_width=5, stride=[1, 1, 1, 1],
                                         out_channel=32, padding="SAME")
max_pool_1_params = nn_structure.max_pool_params(ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding="SAME")

conv_2_params = nn_structure.conv_parmas(window_height=5, window_width=5, stride=[1, 1, 1, 1],
                                         out_channel=64, padding="SAME")
max_pool_2_params = max_pool_1_params

full_connected_units = 1024


def place_holder(batch_size):
    image_placeholder = tf.placeholder(tf.float32,
                                       shape=[batch_size, FLAGS.IMAGE_PIXES])
    label_placeholder = tf.placeholder(tf.int32, shape=[batch_size])
    keep_prob = tf.placeholder(tf.float32)
    return image_placeholder, label_placeholder, keep_prob


def fill_feed_dict(dataSet, keep_prob, image_placeholder, label_placeholder, keep_prob_placeholder):
    images, labels = dataSet.next_batch(FLAGS.batch_size)
    # images = np.reshape(images,
    #                     [FLAGS.batch_size, FLAGS.IMAGE_WIDTH, FLAGS.IMAGE_HEIGHT, FLAGS.IMAGE_CHANNEL])
    return {
        image_placeholder: images,
        label_placeholder: labels,
        keep_prob_placeholder: keep_prob
    }


def do_eval(sess, eval_correct, data_set, images_placeholder, labels_placeholder, keep_prob_placeholder):
    correct_count = 0
    step_num = data_set.num_examples // FLAGS.batch_size
    true_num_examples = step_num * FLAGS.batch_size
    for _ in range(step_num):
        feed_dict = fill_feed_dict(data_set, 1.0, images_placeholder, labels_placeholder, keep_prob_placeholder)
        correct_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = correct_count / true_num_examples * 100
    print("#examples: {:d}, #correct: {:d}, precision: {:.2f}%"
          .format(true_num_examples, correct_count, precision))
    return tf.constant(precision)


def run_training():
    # for mnist
    # train_data, test_data, validation_data = input_data.read_data_sets("../data/MNIST_data/")
    # for cifar-10
    train_data, test_data, validation_data = input_data.load_data()

    with tf.Graph().as_default():
        image_pl, label_pl, keep_prob_pl = place_holder(FLAGS.batch_size)
        logits = nn_structure.inference(image_pl,
                                        conv_1_params, max_pool_1_params,
                                        conv_2_params, max_pool_2_params,
                                        full_connected_units,
                                        keep_prob_pl)
        loss = nn_structure.loss(logits, label_pl)
        train_op = nn_structure.train(loss, FLAGS.learning_rate)
        eval_correct = nn_structure.evaluation(logits, label_pl, k=1)
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            start_time = time.time()
            for step in range(FLAGS.max_step):
                feed_dict = fill_feed_dict(train_data, 0.5, image_pl, label_pl, keep_prob_pl)
                _, loss_value = sess.run([train_op, loss], feed_dict)

                if step % 100 == 0:
                    duration = time.time() - start_time
                    print("Step: {:d}, Training Loss: {:.4f}, {:.1f}ms/step".
                          format(step, loss_value, duration * 10))
                    start_time = time.time()

                if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_step:
                    print("Train Eval:")
                    do_eval(sess, eval_correct, train_data, image_pl, label_pl, keep_prob_pl)
                    print("Validation Eval:")
                    do_eval(sess, eval_correct, validation_data, image_pl, label_pl, keep_prob_pl)
                    print("Test Eval:")
                    do_eval(sess, eval_correct, test_data, image_pl, label_pl, keep_prob_pl)



def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
