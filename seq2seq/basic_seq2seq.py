import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from seq2seq import input_data


class BasicSeq2SeqModel:
    def __init__(self,
                 input_len,
                 output_len,
                 vocab_size,
                 embedding_size,
                 hidden_unit,
                 num_layers,
                 batch_size):
        # 基本参数
        self._input_len = input_len
        self._output_len = output_len
        self._embedding_size = embedding_size
        self._vocab_size = vocab_size
        self._hidden_unit = hidden_unit
        self._num_layer = num_layers
        self._batch_size = batch_size

        # 输入输出id的placeholder，格式[batch_size, len]
        self._input_ids = tf.placeholder(tf.int32, [self._batch_size, self._input_len])
        self._output_ids = tf.placeholder(tf.int32, [self._batch_size, self._output_len])

        # embedding层
        with tf.name_scope("embedding"):
            with tf.device("/cpu:0"):
                self._embeddings = tf.get_variable(name="embedding",
                                                   initializer=tf.truncated_normal(
                                                       [vocab_size,embedding_size]),
                                                   dtype=tf.float32)

        # encoder
        self._stack_encode_cells = self._encode_cell = rnn.LSTMCell(hidden_unit)
        if num_layers > 1:
            self._stack_cells = rnn.MultiRNNCell([self._encode_cell] * num_layers)
        self._init_state = self._stack_encode_cells.zero_state(self._batch_size, tf.float32)

        # decoder
        self._stack_decode_cells = self._cell = rnn.LSTMCell(hidden_unit)
        if num_layers > 1:
            self._stack_cells = rnn.MultiRNNCell([self._encode_cell] * num_layers)

        # projection
        with tf.name_scope("projection"):
            self._w_proj = tf.get_variable(name="w",
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal(
                                               [self._embedding_size, self._vocab_size]))
            self._b_proj = tf.get_variable(name="b",
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal([self._vocab_size]))

        # logits格式 [-1, embedding_size],reshape from [time_step, batch_size, embedding_size]
        self._logits = self.inference

    @property
    def inference(self):
        # [batch_size, time_step, embeddings]
        input_embeddings = tf.nn.embedding_lookup(self._embeddings, self._input_ids)
        # to [time_step, batch_size, embeddings]
        input_embeddings = tf.transpose(input_embeddings, perm=[1, 0, 2])

        _state = self._init_state
        encoder_output = []
        with tf.variable_scope("encoder") as vs:
            for _time in range(self._input_len):
                _time_input = input_embeddings[0]
                if _time > 0:
                    vs.reuse_variables()
                _output, _state = self._stack_encode_cells(_time_input, _state)
                encoder_output.append(_output)

        decoder_output = []
        with tf.variable_scope("decoder") as vs:
            for _time in range(self._output_len):
                # decoder 第一个cell的输入为encoder最后一个cell输出的output, state
                if _time > 0:
                    vs.reuse_variables()
                _output, _state = self._stack_decode_cells(encoder_output[_time], _state)
                decoder_output.append(_output)

        # 将decoder_output转化为tensor对象[time_step, batch_size, embedding_size]
        decoder_output = tf.convert_to_tensor(decoder_output)

        decoder_output = tf.reshape(decoder_output, [-1, self._embedding_size])
        # projection, symbol少直接softmax
        logits = tf.matmul(decoder_output, self._w_proj) + self._b_proj
        return logits

    @property
    def loss(self):
        # [batch_size, output_len]
        labels = tf.transpose(self._output_ids)
        labels = tf.reshape(labels, [-1])
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self._logits)
        return tf.reduce_mean(cross_entropy)

    @property
    def predict(self):
        label_predict = tf.argmax(self._logits, dimension=1)
        return label_predict

    @property
    def accuracy(self):
        labels = self.predict
        labels = tf.reshape(labels, [self._output_len, self._batch_size])
        labels = tf.transpose(labels)
        labels = tf.cast(labels, dtype=tf.int32)
        correct_example = tf.cast(tf.equal(self._output_ids, labels), tf.float32)
        return tf.reduce_mean(correct_example)

    @property
    def seeee(self):
        labels = self.predict
        labels = tf.reshape(labels, [self._output_len, self._batch_size])
        labels = tf.transpose(labels)
        labels = tf.cast(labels, dtype=tf.int32)
        correct_example = tf.cast(tf.equal(self._output_ids, labels), tf.float32)
        correct_example = correct_example[:, 0]
        return tf.reduce_mean(correct_example)

    def feed_dict(self, input_ids, output_ids=None):
        return {self._input_ids: input_ids,
                self._output_ids: output_ids}


def main_train():
    train_data, test_data = input_data.load_data()
    input_len = train_data.input_len
    output_len = train_data.output_len
    vocab_size = train_data.vocab_size
    learning_rate = 0.01
    hidden_unit = 128
    embedding_size = 128
    training_iters = 100000
    batch_size = 128
    num_layers = 3
    display_step = 100

    basic_seq2seq_model = BasicSeq2SeqModel(input_len, output_len, vocab_size,
                                            hidden_unit=hidden_unit,
                                            embedding_size=embedding_size,
                                            batch_size=batch_size,
                                            num_layers=num_layers)
    loss = basic_seq2seq_model.loss
    accuracy = basic_seq2seq_model.accuracy
    seeee = basic_seq2seq_model.seeee
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    with tf.Session() as sess:
        iter_time = 1
        sess.run(tf.global_variables_initializer())
        while iter_time < training_iters:
            input_seqs, output_seqs = train_data.next_batch(batch_size)
            # input_seqs = input_seqs.reshape([batch_size, input_len])
            _, loss_val, acc_val, seeee_val = \
                sess.run([train_op, loss, accuracy, seeee],
                         feed_dict=basic_seq2seq_model.feed_dict(input_seqs, output_seqs))
            if iter_time % display_step == 0:
                print("Step: {:d}, Loss: {:.3f}, Accuracy: {:.3f}"
                      .format(iter_time, loss_val, acc_val))
                print(seeee_val)
            iter_time += 1


if __name__ == '__main__':
    main_train()
