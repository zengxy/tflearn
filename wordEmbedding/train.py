import tensorflow as tf
import input_data
import numpy as np
import word2vec
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("vocabulary_size", 40000, "vocabulary_size")
tf.app.flags.DEFINE_float("learning_rate", 1.0, "learning_rate")
tf.app.flags.DEFINE_integer("max_step", 100001, "max_step")
tf.app.flags.DEFINE_integer("validation_size", 10, "validation_size")
tf.app.flags.DEFINE_integer("embedding_size", 70, "embedding_size")
tf.app.flags.DEFINE_integer("neg_sampled", 70, "neg_sampled")

tf.app.flags.DEFINE_integer("batch_size", 128, "vocabulary_size")
tf.app.flags.DEFINE_integer("num_skips", 2, "vocabulary_size")
tf.app.flags.DEFINE_integer("skip_window", 1, "vocabulary_size")


def place_holder(batch_size, validation_size):
    batch_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    batch_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_ids = tf.placeholder(tf.int32, shape=[validation_size])
    return batch_inputs, batch_labels, valid_ids


def fill_feed_dict(dataSet, batch_inputs, batch_labels,
                   batch_size, num_skips, skip_window):
    input_value, label_value = dataSet.next_batch(batch_size, num_skips, skip_window)
    return {batch_inputs: input_value,
            batch_labels: label_value}


def run_training():
    text_dataset = input_data.load_data("yitian.txt", max_vocabulary_size=40000)
    valid_window = np.array(range(5, 15))
    sample_p = (19 - valid_window) / np.sum(valid_window)
    valid_ids = np.random.choice(valid_window, FLAGS.validation_size, p=sample_p, replace=False)

    with tf.Graph().as_default():
        batch_inputs_pl, batch_labels_pl, valid_ids_pl = place_holder(FLAGS.batch_size, FLAGS.validation_size)
        loss, embeddings = word2vec.loss(batch_inputs_pl, batch_labels_pl)
        train_op = word2vec.train(loss)
        sim_compute = word2vec.compute_sim(valid_ids_pl, embeddings)
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            start_time = time.time()
            for step in range(FLAGS.max_step):
                filled_dict = fill_feed_dict(text_dataset,
                                             batch_inputs_pl, batch_labels_pl,
                                             FLAGS.batch_size, FLAGS.num_skips, FLAGS.skip_window)
                _, loss_value = sess.run([train_op, loss], filled_dict)

                if step % 1000 == 0:
                    duration = time.time() - start_time
                    print("Step: {:d}, Training Loss: {:.4f}, {:.1f}us/step".
                          format(step, loss_value, duration * 1000))

                if (step + 1) % 5000 == 0 or (step + 1) == FLAGS.max_step:
                    sim_words_id, _ = sess.run(sim_compute, {valid_ids_pl: valid_ids})
                    for (i, word_id) in enumerate(valid_ids):
                        word = text_dataset.word_count[word_id][0]
                        sim_words = []
                        for sim_word_id in sim_words_id[i]:
                            sim_words.append(text_dataset.word_count[sim_word_id][0])
                        print(word, end=":")
                        print(" ".join(sim_words))
                start_time = time.time()


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
    # batch_inputs_pl, batch_labels_pl, valid_ids_pl = place_holder(FLAGS.batch_size, FLAGS.validation_size)
    # loss, embeddings = word2vec.loss(batch_inputs_pl, batch_labels_pl)
    # train_op = word2vec.train(loss)
    # sim_compute = word2vec.compute_sim(valid_ids_pl, embeddings)
