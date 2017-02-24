import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from seq2seq import input_data


class BasicSeq2SeqModel():
    def __init__(self,
                 input_len,
                 output_len,
                 hidden_unit = 128,
                 num_layers = 3,
                 batch_size = 128):
        self._input_len = input_len
        self._output_len = output_len
        self._hidden_unit = hidden_unit
        self._num_layer = num_layers
        self._batch_size = batch_size
        self._stack_cells = self._cell = rnn.BasicRNNCell(hidden_unit)
        if num_layers > 1:
            self._stack_cells = rnn.MultiRNNCell([self._cell]*num_layers)

    @property
    def inference(self):
        pass


