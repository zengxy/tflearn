import tensorflow as tf
import math

FLAGS = tf.app.flags.FLAGS


def loss(batch_input, batch_label):
    # embedding矩阵使用cpu来保存
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(
            tf.random_uniform([FLAGS.vocabulary_size, FLAGS.embedding_size], -1.0, 1.0),
            name="embeddings"
        )
        batch_embed = tf.nn.embedding_lookup(embeddings, batch_input)

        nce_weights = tf.Variable(
            tf.truncated_normal([FLAGS.vocabulary_size, FLAGS.embedding_size],
                                stddev=1.0 / math.sqrt(FLAGS.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([FLAGS.vocabulary_size]))
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, batch_embed, batch_label,
                       FLAGS.neg_sampled, FLAGS.vocabulary_size))

    return loss, embeddings


def train(loss):
    return tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)


def compute_sim(valid_words_id, embeddings, top_k = 10):
    # embeddings = tf.get_variable(name="embeddings")
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_words_id)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    # print(similarity.get_shape())
    values, indices = tf.nn.top_k(similarity, top_k)
    # print(indices.get_shape())
    return indices, values


def test_word2vec():
    from wordEmbedding import input_data
    data_set = input_data.load_data()
    for i in range(10):
        print(data_set.next_batch(10, 2, 1))


if __name__ == '__main__':
    test_word2vec()