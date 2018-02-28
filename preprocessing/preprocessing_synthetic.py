# ==============================================================================
# Copyright 2018 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Synthetic data pre-processing.
"""
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

class SyntheticImagePreprocessor(object):
    """Preprocessor used for images and labels.
    """
    def __init__(self, height, width, batch_size, num_splits,
                 dtype, train, distortions, resize_method, shift_ratio,
                 summary_verbosity, distort_color_in_yiq=False,
                 fuse_decode_and_crop=False):
        del train, distortions, resize_method, summary_verbosity
        del distort_color_in_yiq
        del fuse_decode_and_crop
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.depth = 3
        self.dtype = dtype
        self.num_splits = num_splits
        self.shift_ratio = shift_ratio

    def minibatch(self, dataset, subset, use_datasets, cache_data,
                  shift_ratio=-1):
        """Get synthetic image batches.
        """
        del subset, use_datasets, cache_data, shift_ratio
        input_shape = [self.batch_size, self.height, self.width, self.depth]
        images = tf.truncated_normal(
            input_shape,
            dtype=self.dtype,
            stddev=1e-1,
            name='synthetic_images')
        labels = tf.random_uniform(
            [self.batch_size],
            minval=0,
            maxval=dataset.num_classes - 1,
            dtype=tf.int32,
            name='synthetic_labels')
        # Note: This results in a H2D copy, but no computation
        # Note: This avoids recomputation of the random values, but still
        #         results in a H2D copy.
        images = tf.contrib.framework.local_variable(images, name='images')
        labels = tf.contrib.framework.local_variable(labels, name='labels')
        if self.num_splits == 1:
            images_splits = [images]
            labels_splits = [labels]
        else:
            images_splits = tf.split(images, self.num_splits, 0)
            labels_splits = tf.split(labels, self.num_splits, 0)
        return images_splits, labels_splits


class TestImagePreprocessor(object):
    """Preprocessor used for testing.

    set_fake_data() sets which images and labels will be output by minibatch(),
    and must be called before minibatch(). This allows tests to easily specify
    a set of images to use for training, without having to create any files.

    Queue runners must be started for this preprocessor to work.
    """

    def __init__(self,
                 height,
                 width,
                 batch_size,
                 num_splits,
                 dtype,
                 train=None,
                 distortions=None,
                 resize_method=None,
                 shift_ratio=0,
                 summary_verbosity=0,
                 distort_color_in_yiq=False,
                 fuse_decode_and_crop=False):
        del height, width, train, distortions, resize_method
        del summary_verbosity, fuse_decode_and_crop, distort_color_in_yiq
        self.batch_size = batch_size
        self.num_splits = num_splits
        self.dtype = dtype
        self.expected_subset = None
        self.shift_ratio = shift_ratio

    def set_fake_data(self, fake_images, fake_labels):
        assert len(fake_images.shape) == 4
        assert len(fake_labels.shape) == 1
        assert fake_images.shape[0] == fake_labels.shape[0]
        assert fake_images.shape[0] % self.batch_size == 0
        self.fake_images = fake_images
        self.fake_labels = fake_labels

    def minibatch(self, dataset, subset, use_datasets, cache_data,
                  shift_ratio=-1):
        del dataset, use_datasets, cache_data, shift_ratio
        if (not hasattr(self, 'fake_images') or not hasattr(self, 'fake_labels')):
            raise ValueError('Must call set_fake_data() before calling minibatch '
                             'on TestImagePreprocessor')
        if self.expected_subset is not None:
            assert subset == self.expected_subset

        with tf.name_scope('batch_processing'):
            image_slice, label_slice = tf.train.slice_input_producer(
                [self.fake_images, self.fake_labels],
                shuffle=False,
                name='image_slice')
            raw_images, raw_labels = tf.train.batch(
                [image_slice, label_slice], batch_size=self.batch_size,
                name='image_batch')
            images = [[] for _ in range(self.num_splits)]
            labels = [[] for _ in range(self.num_splits)]
            for i in xrange(self.batch_size):
                split_index = i % self.num_splits
                raw_image = tf.cast(raw_images[i], self.dtype)
                images[split_index].append(raw_image)
                labels[split_index].append(raw_labels[i])
            for split_index in xrange(self.num_splits):
                images[split_index] = tf.parallel_stack(images[split_index])
                labels[split_index] = tf.parallel_stack(labels[split_index])

            return images, labels
