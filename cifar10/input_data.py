import tensorflow as tf
import os
import sys
import urllib.request
import tarfile
import pickle
import collections
import numpy as np
from tensorflow.python.framework import dtypes

FLAGS = tf.app.flags.FLAGS

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
tf.app.flags.DEFINE_string("data_dir",
                           os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/cifar10"),
                           "Directory of data dir.")

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNEL = 3
IMAGE_PIXES = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL
LABEL_NUM = 10

Datasets = collections.namedtuple('Datasets', ['train', 'test', 'validation'])


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * IMAGE_PIXES
            if self.one_hot:
                fake_label = [1] + [0] * (LABEL_NUM - 1)
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)
                ]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


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
        test_dataset = DataSet(images=temp["data"], labels=np.array(temp["labels"]))

    train_data_list = []
    train_lable_list = []
    for batch_file in data_batch_files[:-1]:
        with open(batch_file, "rb") as f:
            temp = pickle.load(f, encoding="latin1")
            train_data_list.append(temp["data"])
            train_lable_list.append(np.array(temp["labels"]))
    train_dataset = DataSet(images=np.concatenate(train_data_list), labels=np.concatenate(train_lable_list))

    with open(data_batch_files[-1], "rb") as f:
        temp = pickle.load(f, encoding="latin1")
        validation_dataset = DataSet(images=temp["data"], labels=np.array(temp["labels"]))

    return Datasets(train_dataset, test_dataset, validation_dataset)


def shuffle(data_set):
    data = data_set.data
    label = data_set.label
    p = np.random.permutation(len(data))
    return DataSet(data[p], label[p])


if __name__ == '__main__':
    _, _, v_data = load_data()
    for _ in range(10):
        print(v_data.next_batch(100)[1][:10])
