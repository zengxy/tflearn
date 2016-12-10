import collections
import os
import sys
import urllib.request
import zipfile
import numpy as np
import random
import jieba
import string

import tensorflow as tf

DATA_URL = 'http://mattmahoney.net/dc/text8.zip'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir",
                           os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/word2vec"),
                           "Directory of data dir.")


class DataSet(object):
    def __init__(self,
                 words,
                 max_vocabulary_size=50000,
                 rare_words_mark='UNK'):
        '''
        根据文本构建词汇表
        data: 将words转换为indexes
        count: 按照频率排列的word:count对，第一个为UNK
        dictionary: word: index, 其中count[index][0] = word
        reverse_dictionary: index: word
        '''
        self._words = words
        self._vocabulary_size = max_vocabulary_size
        self._rare_words_mark = rare_words_mark

        self._word_count = [[rare_words_mark, -1]]
        self._word_index_dict = dict()
        self._data = list()
        self.build_dataset()

        # 语料中的word可能比vocabulary_size少
        self._vocabulary_size = min(max_vocabulary_size, len(self._word_count))

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

    def next_batch(self, batch_size, num_skips, skip_window, sample_model="skip_gram"):
        if sample_model == "skip_gram":
            return self._next_batch_skip_gram(batch_size, num_skips, skip_window)

    # skip-gram model  current word -> context word
    def _next_batch_skip_gram(self, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window]
        buffer = collections.deque(maxlen=span)

        # 初始化deque
        for _ in range(span):
            buffer.append(self._data[self._index_in_epoch])
            self._index_in_epoch = (self._index_in_epoch + 1) % len(self._data)

        # skip-gram sample
        for i in range(batch_size // num_skips):
            context_words = list(range(span))
            context_words.remove(skip_window)
            context_index_choosed = random.sample(context_words, num_skips)

            for j, target in enumerate(context_index_choosed):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]

            buffer.append(self._data[self._index_in_epoch])
            self._index_in_epoch = (self._index_in_epoch + 1) % len(self._data)

        return batch, labels

    @property
    def word_index_dict(self):
        return self._word_index_dict

    @property
    def data(self):
        return self._data

    @property
    def word_count(self):
        return self._word_count

    @property
    def vocabulary_size(self):
        return self._vocabulary_size


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


def load_data(filename="text8.zip", encoding="utf-8", max_vocabulary_size=50000):
    if filename == "text8.zip":
        return _load_text8_data()
    return _load_text_data(filename, encoding, max_vocabulary_size)


def _load_text_data(filename, encoding, max_vocabulary_size):
    file_path = os.path.join(FLAGS.data_dir, filename)
    with open(file_path, encoding=encoding) as f:
        punctuation = string.punctuation + "～`@#￥%……&×（）——+-={}|·「」|、：;“‘’”《》，。？/\n\t\r"
        text = f.read()
    for dicfilename in ["bangs.txt", "kungfu.txt", "names.txt"]:
        dicfile = os.path.join(FLAGS.data_dir, dicfilename)
        with open(dicfile) as f:
            jieba.load_userdict(f)
    words = list(filter(lambda word: word not in punctuation, jieba.cut(text)))
    text_dataset = DataSet(words, max_vocabulary_size=max_vocabulary_size)
    return text_dataset


def _load_text8_data():
    filepath = maybe_download(FLAGS.data_dir)
    with zipfile.ZipFile(filepath) as f:
        words = f.read(f.namelist()[0]).decode("utf-8").split(" ")
    text_dataset = DataSet(words)
    # print('Most common words (+UNK)', text_dataset.word_count[:5])
    # print('Sample data', text_dataset.data[:10], [text_dataset.word_count[i][0] for i in text_dataset.data[:10]])
    return text_dataset


if __name__ == '__main__':
    load_data()
