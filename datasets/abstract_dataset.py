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
"""Definition of an abstract dataset.
"""
from abc import abstractmethod
import os
import tensorflow as tf

import preprocessing

class Dataset(object):
    """Abstract class for cnn benchmarks dataset.
    Queue_runner_required?
    """
    def __init__(self, name, subset, height=None, width=None, depth=None, data_dir=None,
                 queue_runner_required=False, num_classes=1000):
        self.name = name
        self.subset = subset
        self.height = height
        self.width = width
        self.depth = depth or 3

        self.data_dir = data_dir
        self._queue_runner_required = queue_runner_required
        self._num_classes = num_classes

    def tf_record_pattern(self, subset):
        return os.path.join(self.data_dir, '%s-*-of-*' % subset)

    def reader(self):
        return tf.TFRecordReader()

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, val):
        self._num_classes = val

    @abstractmethod
    def num_examples_per_epoch(self, subset):
        pass

    def __str__(self):
        return self.name

    def get_image_preprocessor(self):
        return None

    def queue_runner_required(self):
        return self._queue_runner_required

    def use_synthetic_gpu_images(self):
        return False


class SyntheticData(Dataset):
    """Configuration for synthetic dataset.
    """
    def __init__(self, unused_data_dir, unused_data_subset):
        super(SyntheticData, self).__init__('synthetic', 'train')

    def get_image_preprocessor(self):
        return preprocessing.SyntheticImagePreprocessor

    def num_examples_per_epoch(self, subset):
        pass

    def use_synthetic_gpu_images(self):
        return True
