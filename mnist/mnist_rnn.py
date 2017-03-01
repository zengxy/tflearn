import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data


class MnistRNNModel:
    def __init__(self, rnn_hidden_units, batch_size=128, time_step=28, n_input=28, n_class_num=10):
        self._rnn_hidden_units = rnn_hidden_units
        self._batch_size = batch_size
        self._time_step = time_step
        self._n_input = n_input
        self._n_class_num = n_class_num

        self._x = tf.placeholder(tf.float32,
                                 shape=[self._batch_size, self._time_step, self._n_input])
        self._y = tf.placeholder(tf.int32,
                                 shape=[self._batch_size])
        # with tf.variable_scope("rnn"):
        self._cell = rnn.BasicRNNCell(num_units=self._rnn_hidden_units)
        self._stack_cells = rnn.MultiRNNCell([self._cell] * 3)
        self._W = tf.Variable(initial_value=tf.truncated_normal(
            shape=[self._rnn_hidden_units, self._n_class_num]))
        self._biases = tf.Variable(tf.zeros(shape=[self._n_class_num]))

        self._logits = self.inference
        self._y_pred = self.predict
        self._accuracy = self.accuracy

    @property
    def inference(self):
        x = tf.transpose(self._x, [1, 0, 2])
        x = tf.reshape(x, [-1, self._n_input])
        x = tf.split(x, self._time_step, axis=0)
        _state = self._stack_cells.zero_state(self._batch_size, dtype=tf.float32)
        outputs = []
        with tf.variable_scope("rnn") as vs:
            for time, _input in enumerate(x):
                if time > 0:
                    vs.reuse_variables()
                _output, _state = self._stack_cells(_input, _state)
                outputs.append(_output)
        # outputs, _ = rnn.rnn(self._basic_rnn_cell, x, dtype=tf.float32)
        logits = tf.matmul(outputs[-1], self._W) + self._biases
        return logits

    @property
    def predict(self):
        y_pred = tf.arg_max(self._logits, dimension=1)
        y_pred = tf.cast(y_pred, tf.int32)
        return y_pred

    @property
    def accuracy(self):
        correct_example = tf.cast(tf.equal(self._y, self._y_pred), tf.float32)
        return tf.reduce_mean(correct_example)

    @property
    def loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._logits, labels=self._y)
        return tf.reduce_mean(cross_entropy)

    def feed_dict(self, x_feed, y_feed):
        return {self._x: x_feed, self._y: y_feed}


def train_mnist_rnn_model():
    # params
    data_dir = "../data/MNIST_data"
    learning_rate = 0.001
    rnn_hidden_unit = 128
    training_iters = 100000
    batch_size = 128
    display_step = 100

    train_data, validation_data, test_data = input_data.read_data_sets(data_dir)
    mnist_classify_model = MnistRNNModel(rnn_hidden_unit)
    loss = mnist_classify_model.loss
    accuracy = mnist_classify_model.accuracy
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(mnist_classify_model.loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iter_time = 1
        while iter_time<training_iters:
            images, labels = train_data.next_batch(batch_size)
            images = images.reshape([128, 28, 28])
            _, loss_val, acc_val = \
                sess.run([train_op, loss, accuracy],
                         feed_dict=mnist_classify_model.feed_dict(images, labels))
            if iter_time%display_step == 0:
                print("Step: {:d}, Loss: {:.3f}, Accuracy: {:.3f}"
                      .format(iter_time, loss_val, acc_val))
            iter_time += 1
if __name__ == '__main__':
    train_mnist_rnn_model()