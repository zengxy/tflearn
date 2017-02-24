import pickle
import os
import random
import numpy as np
from collections import namedtuple

data_dir = "../data/seq2seq/"


def load_data(data_size=10000, input_len=10, test_rate = 0.2):
    data_file = os.path.join(data_dir, "basic_seq2seq-"+str(data_size)+".pk")
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            seq_data = pickle.load(f)
    else:
        seq_data = generate_data(data_size, input_len)
        with open(data_file, "wb") as f:
            pickle.dump(seq_data, f)
    input_seq, output_seq = seq_data
    split_index = int(data_size*test_rate)
    train_dataset = DataSet(input_seq[:split_index], output_seq[:split_index])
    test_dataset = DataSet(input_seq[split_index+1:], output_seq[split_index+1:])
    return DataSets(train_dataset, test_dataset)


def generate_data(data_size, input_len):
    chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
    char_index_dict = {char: index for (index, char) in enumerate(chars)}
    chars_len = len(chars)
    input_seq = []
    output_seq = []
    for _ in range(data_size):
        input_chars = random.sample(chars, input_len)
        output_chars = []
        init_state = sum(char_index_dict[char] for char in input_chars)
        for char in input_chars:
            out_char_index = (char_index_dict[char]+init_state)%chars_len
            output_chars.append(chars[out_char_index])
        input_seq.append(input_chars)
        output_seq.append(output_chars)
    return [np.array(input_seq), np.array(output_seq)]

DataSets = namedtuple("DataSets", ["train", "test"])


class DataSet():
    def __init__(self, input_seq, output_seq):
        self._input_seq = input_seq
        self._output_seq = output_seq
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(input_seq)

    @property
    def input_seq(self):
        return self._input_seq

    @property
    def output_seq(self):
        return self._output_seq

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._input_seq = self._input_seq[perm]
            self._output_seq = self._output_seq[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._input_seq[start:end], self._output_seq[start:end]


if __name__ == '__main__':
    train_data, test_data = load_data()
    print(train_data.input_seq[0])
    print(train_data.output_seq[0])