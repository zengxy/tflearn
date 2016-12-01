import tensorflow as tf
import os
import sys
import urllib.request
import tarfile
import pickle
import collections
import numpy as np

FLAGS = tf.app.flags.FLAGS

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
tf.app.flags.DEFINE_string("data_dir", "../data/cifar10", "Directory of data dir.")

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNEL = 3
LABEL_NUM = 10
Dataset = collections.namedtuple('Dataset', ['data', 'label'])
Datasets = collections.namedtuple('Datasets', ['train', 'test'])


def maybe_download_and_extract(data_dir):
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
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
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def load_data():
    maybe_download_and_extract(FLAGS.data_dir)
    data_batch_files = [os.path.join(FLAGS.data_dir, "cifar-10-batches-py/data_batch_{}".format(i))
                        for i in range(1, 6)]
    test_data_file = os.path.join(FLAGS.data_dir, "cifar-10-batches-py/test_batch")

    with open(test_data_file, "rb") as f:
        temp = pickle.load(f, encoding="latin1")
        test_dataset = Dataset(data=temp["data"], label=temp["labels"])

    train_data_list = []
    train_lable_list = []
    for batch_file in data_batch_files:
        with open(batch_file, "rb") as f:
            temp = pickle.load(f, encoding="latin1")
            train_data_list.append(temp["data"])
            train_lable_list.append(temp["labels"])

    train_dataset = Dataset(data=np.concatenate(train_data_list), label=np.concatenate(train_lable_list))


def shuffle(data_set):
    data = data_set.data
    label = data_set.label
    p = np.random.permutation(len(data))
    return Dataset(data[p], label[p])


if __name__ == '__main__':
    a = Dataset(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]), np.array([1, 2, 3, 4]))
    for _ in range(4):
        a = shuffle(a)
        print(a)
