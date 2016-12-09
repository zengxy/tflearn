import collections
import os
import sys
import urllib.request
import zipfile
import numpy as np

import tensorflow as tf

DATA_URL = 'http://mattmahoney.net/dc/text8.zip'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "../data/word2vec", "Directory of data dir.")


class DataSet(object):
    def __init__(self,
                 words,
                 vocabulary_size=50000,
                 rare_words_mark='UNK'):
        '''
        根据文本构建词汇表
        data: 将words转换为indexes
        count: 按照频率排列的word:count对，第一个为UNK
        dictionary: word: index, 其中count[index][0] = word
        reverse_dictionary: index: word
        '''
        self._words = words
        self._vocabulary_size = vocabulary_size
        self._rare_words_mark = rare_words_mark

        self._word_count = [[rare_words_mark, -1]]
        self._word_index_dict = dict()
        self._data = list()
        self.build_dataset()

        self._epochs_completed = 0
        self._index_in_epoch = 0

    def build_dataset(self):
        self._word_count.extend(collections.Counter(self._words).most_common(self._vocabulary_size - 1))
        for i, word_count in enumerate(self._word_count):
            self._word_index_dict[word_count[0]] = i
        unk_count = 0
        for word in self._words:
            if word in self._word_index_dict:
                index = self._word_index_dict[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            self._data.append(index)
        self._word_count[0][1] = unk_count
        # reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    def next_batch(self, batch_size):
        pass

    @property
    def word_index_dict(self):
        return self._word_index_dict

    @property
    def data(self):
        return self._data

    @property
    def word_count(self):
        return self._word_count


def maybe_download(data_dir):
    """Download a file if not present"""
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.mkdir(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def load_data():
    filepath = maybe_download(FLAGS.data_dir)
    with zipfile.ZipFile(filepath) as f:
        words = f.read(f.namelist()[0]).decode("utf-8").split(" ")
    text_dataset = DataSet(words)
    print('Most common words (+UNK)', text_dataset.word_count[:5])
    print('Sample data', text_dataset.data[:10], [text_dataset.word_count[i][0] for i in text_dataset.data[:10]])
    return text_dataset


if __name__ == '__main__':
    load_data()
