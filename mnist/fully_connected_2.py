import tensorflow as tf
import math

NUM_CLASSES = 10
IMAGE_SIZE = 28

IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(image, hidden1_units, hidden2_units):
    with tf.name_scope("hidden1"):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                                  stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                              name="weights")
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name="biases")
        h_hidden1 = tf.nn.relu(tf.matmul(image, weights) + biases)

    with tf.name_scope("hidden2"):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                                                  stddev=1.0 / math.sqrt(float(hidden1_units))),
                              name="weights")
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name="biases")
        h_hidden2 = tf.nn.relu(tf.matmul(h_hidden1, weights) + biases)

    with tf.name_scope("softmax_liner"):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                                  stddev=1.0 / math.sqrt(float(NUM_CLASSES))),
                              name="weights")
        biases = tf.Variable(tf.truncated_normal([NUM_CLASSES]),
                             name="biases")
        logits = tf.matmul(h_hidden2, weights) + biases

    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    if len(labels.get_shape()) == 1:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name="cross_entropy")
    else:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name="cross_entropy")
    loss = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
    return loss


def train(loss, learning_rate):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
