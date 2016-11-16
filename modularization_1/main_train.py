import tensorflow as tf
import time
import os

from tensorflow.examples.tutorials.mnist import input_data
import fully_connected_2_hidden as nn_structure

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("max_step", 2000, "Numbers of training steps.")
flags.DEFINE_integer("hidden1_unit", 128, "Numbers of hidden1 layer units.")
flags.DEFINE_integer("hidden2_unit", 32, "Numbers of hidden2 layer units.")
flags.DEFINE_integer("batch_size", 100, "Batch size for training.")
flags.DEFINE_string("training_data_dir", "../MNIST_data", "Directory of training data.")
flags.DEFINE_boolean("fake_data", False, 'If True, use fake_date for unit test')


def place_holder(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=[batch_size, nn_structure.IMAGE_PIXELS])
    labels_placeholder = tf.placeholder(tf.int32, shape=[batch_size])

    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    images, labels = data_set.next_batch(FLAGS.batch_size)
    return {images_pl: images,
            labels_pl: labels}


def do_eval(sess, eval_correct, data_set, images_placeholder, labels_placeholder):
    correct_count = 0
    step_num = data_set.num_examples // FLAGS.batch_size
    true_num_examples = step_num * FLAGS.batch_size
    for _ in range(step_num):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        correct_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = correct_count / true_num_examples * 100
    print("#examples: {:d}, #correct: {:d}, precision: {:.2f}%"
          .format(true_num_examples, correct_count, precision))
    return tf.constant(precision)


def run_training():
    (train_data, validation_data, test_data) = \
        input_data.read_data_sets(FLAGS.training_data_dir, fake_data=FLAGS.fake_data)

    with tf.Graph().as_default():
        (images_pl, labels_pl) = place_holder(FLAGS.batch_size)

        logits = nn_structure.inference(images_pl, FLAGS.hidden1_unit, FLAGS.hidden2_unit)
        loss = nn_structure.loss(logits, labels_pl)
        train_op = nn_structure.train(loss, FLAGS.learning_rate)
        eval_correct = nn_structure.evalutaion(logits, labels_pl)

        summary = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            summary_writer = tf.train.SummaryWriter(FLAGS.training_data_dir, sess.graph)
            sess.run(init)
            start_time = time.time()
            for step in range(FLAGS.max_step):
                feed_dict = fill_feed_dict(train_data, images_pl, labels_pl)
                _, loss_value = sess.run([train_op, loss], feed_dict)

                if step % 100 == 0:
                    duration = time.time() - start_time
                    print("Step: {:d}, Training Loss: {:.4f}, {:.1f}ms/step".
                          format(step, loss_value, duration * 10))
                    start_time = time.time()

                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_step:
                    print("Train Eval:")
                    precision = do_eval(sess, eval_correct, train_data, images_pl, labels_pl)
                    # sss = tf.scalar_summary("train precision", precision)
                    # summary_writer.add_summary(sss,step)
                    # summary_writer.flush()
                    print("Validation Eval:")
                    do_eval(sess, eval_correct, validation_data, images_pl, labels_pl)
                    print("Test Eval:")
                    do_eval(sess, eval_correct, test_data, images_pl, labels_pl)

                    checkpoint_file = os.path.join(FLAGS.training_data_dir, "checkpoint")
                    saver.save(sess, checkpoint_file, global_step=step)


def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()