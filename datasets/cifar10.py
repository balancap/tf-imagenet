# /* ===========================================================================
# [2017] - Robik AI Ltd - Paul Balanca
# All Rights Reserved.

# NOTICE: All information contained herein is, and remains
# the property of Robik AI Ltd, and its suppliers
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Robik AI Ltd
# and its suppliers and may be covered by U.S., European and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Robik AI Ltd.
# =========================================================================== */
"""Definition of the Cifar-10 dataset.
"""
import os

import numpy as np
from six.moves import cPickle
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.platform import gfile

import preprocessing
from . import abstract_dataset

CIFAR10_NUM_TRAIN_IMAGES = 50000
CIFAR10_NUM_VAL_IMAGES = 10000
CIFAR10_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

class Cifar10Data(abstract_dataset.Dataset):
    """Configuration for cifar 10 dataset.
    It will mount all the input images to memory.
    """
    def __init__(self, data_dir=None, subset='train'):
        if data_dir is None:
            raise ValueError('Data directory not specified')
        super(Cifar10Data, self).__init__(
            'cifar10', subset, 32, 32, data_dir=data_dir,
            queue_runner_required=True, num_classes=10)

    def read_data_files(self, subset='train'):
        """Reads from data file and return images and labels in a numpy array."""
        if subset == 'train':
            filenames = [os.path.join(self.data_dir, 'data_batch_%d' % i)
                         for i in xrange(1, 6)]
        elif subset == 'validation':
            filenames = [os.path.join(self.data_dir, 'test_batch')]
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

        inputs = []
        for filename in filenames:
            with gfile.Open(filename, 'r') as f:
                inputs.append(cPickle.load(f))
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        all_images = np.concatenate(
            [each_input['data'] for each_input in inputs]).astype(np.float32)
        all_labels = np.concatenate(
            [each_input['labels'] for each_input in inputs])
        return all_images, all_labels

    def num_examples_per_epoch(self, subset=None):
        subset = subset if subset else self.subset
        if subset not in CIFAR10_NUM_IMAGES.keys():
            raise ValueError('Invalid data subset "%s"' % subset)
        return CIFAR10_NUM_IMAGES[subset]

    def get_image_preprocessor(self):
        return preprocessing.Cifar10ImagePreprocessor
