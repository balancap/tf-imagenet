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
"""Cifar-10 pre-processing.
"""
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

class Cifar10ImagePreprocessor(object):
    """Preprocessor for Cifar10 input images.
    """
    def __init__(self,
                 height,
                 width,
                 batch_size,
                 num_splits,
                 dtype,
                 train,
                 distortions,
                 resize_method,
                 shift_ratio,
                 summary_verbosity=0,
                 distort_color_in_yiq=False,
                 fuse_decode_and_crop=False):
        # Process images of this size. Depending on the model configuration, the
        # size of the input layer might differ from the original size of 32 x 32.
        self.height = height or 32
        self.width = width or 32
        self.depth = 3
        self.batch_size = batch_size
        self.num_splits = num_splits
        self.dtype = dtype
        self.train = train
        self.distortions = distortions
        self.shift_ratio = shift_ratio
        del distort_color_in_yiq
        del fuse_decode_and_crop
        del resize_method
        del shift_ratio  # unused, because a RecordInput is not used
        if self.batch_size % self.num_splits != 0:
            raise ValueError(
                    ('batch_size must be a multiple of num_splits: '
                     'batch_size %d, num_splits: %d') %
                    (self.batch_size, self.num_splits))
        self.batch_size_per_split = self.batch_size // self.num_splits
        self.summary_verbosity = summary_verbosity

    def _distort_image(self, image):
        """Distort one image for training a network.

        Adopted the standard data augmentation scheme that is widely used for
        this dataset: the images are first zero-padded with 4 pixels on each side,
        then randomly cropped to again produce distorted images; half of the images
        are then horizontally mirrored.

        Args:
            image: input image.
        Returns:
            distored image.
        """
        image = tf.image.resize_image_with_crop_or_pad(
            image, self.height + 8, self.width + 8)
        distorted_image = tf.random_crop(
            image, [self.height, self.width, self.depth])
        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        if self.summary_verbosity >= 3:
            tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))
        return distorted_image

    def _eval_image(self, image):
        """Get the image for model evaluation."""
        distorted_image = tf.image.resize_image_with_crop_or_pad(
            image, self.width, self.height)
        if self.summary_verbosity >= 3:
            tf.summary.image('cropped.image', tf.expand_dims(distorted_image, 0))
        return distorted_image

    def preprocess(self, raw_image):
        """Preprocessing raw image."""
        if self.summary_verbosity >= 3:
            tf.summary.image('raw.image', tf.expand_dims(raw_image, 0))
        if self.train and self.distortions:
            image = self._distort_image(raw_image)
        else:
            image = self._eval_image(raw_image)
        return image

    def minibatch(self, dataset, subset, use_datasets, cache_data,
                  shift_ratio=-1):
        # TODO(jsimsa): Implement datasets code path
        del use_datasets, cache_data, shift_ratio
        with tf.name_scope('batch_processing'):
            all_images, all_labels = dataset.read_data_files(subset)
            all_images = tf.constant(all_images)
            all_labels = tf.constant(all_labels)
            input_image, input_label = tf.train.slice_input_producer(
                [all_images, all_labels])
            input_image = tf.cast(input_image, self.dtype)
            input_label = tf.cast(input_label, tf.int32)
            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            min_queue_examples = int(dataset.num_examples_per_epoch(subset) *
                                     min_fraction_of_examples_in_queue)
            raw_images, raw_labels = tf.train.shuffle_batch(
                [input_image, input_label], batch_size=self.batch_size,
                capacity=min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=min_queue_examples)

            images = [[] for i in range(self.num_splits)]
            labels = [[] for i in range(self.num_splits)]

            # Create a list of size batch_size, each containing one image of the
            # batch. Without the unstack call, raw_images[i] would still access the
            # same image via a strided_slice op, but would be slower.
            raw_images = tf.unstack(raw_images, axis=0)
            raw_labels = tf.unstack(raw_labels, axis=0)
            for i in xrange(self.batch_size):
                split_index = i % self.num_splits
                # The raw image read from data has the format [depth, height, width]
                # reshape to the format returned by minibatch.
                raw_image = tf.reshape(raw_images[i],
                                       [dataset.depth, dataset.height, dataset.width])
                raw_image = tf.transpose(raw_image, [1, 2, 0])
                image = self.preprocess(raw_image)
                images[split_index].append(image)

                labels[split_index].append(raw_labels[i])

            for split_index in xrange(self.num_splits):
                images[split_index] = tf.parallel_stack(images[split_index])
                labels[split_index] = tf.parallel_stack(labels[split_index])
            return images, labels
