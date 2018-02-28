# ==============================================================================
# Copyright 2018 The TensorFlow Authors aud Paul Balanca. All Rights Reserved.
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
"""Abtract classification model.
"""
import tensorflow as tf
import tf_extended as tfx

slim = tf.contrib.slim

# =========================================================================== #
# Abstract classification model
# =========================================================================== #
class Model(object):
    """Base model configuration for CNN training.
    """
    def __init__(self,
                 model,
                 image_size,
                 batch_size,
                 learning_rate,
                 epoch_size=10000,
                 layer_counts=None,
                 fp16_loss_scale=128):
        self.model = model
        self.image_size = image_size
        self.batch_size = batch_size
        self.default_batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch_size = epoch_size
        self.layer_counts = layer_counts
        self.label_smoothing = 0.0
        self.aux_loss_weight = 0.4
        self.weight_decay = 4e-5
        # TODO(reedwm) Set custom loss scales for each model instead of using the
        # default of 128.
        self.variable_dtype = tf.float32
        self.fp16_loss_scale = fp16_loss_scale

    def get_model(self):
        return self.model

    def get_image_size(self):
        return self.image_size

    def get_batch_size(self):
        return self.batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size if batch_size else 32

    def get_default_batch_size(self):
        return self.default_batch_size

    def get_label_smoothing(self):
        return self.label_smoothing

    def set_label_smoothing(self, label_smoothing):
        self.label_smoothing = label_smoothing

    def get_epoch_size(self):
        return self.epoch_size

    def set_epoch_size(self, epoch_size):
        self.epoch_size = epoch_size if epoch_size else 32

    def get_weight_decay(self):
        return self.weight_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def get_layer_counts(self):
        return self.layer_counts

    def get_fp16_loss_scale(self):
        return self.fp16_loss_scale

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def get_learning_rate(self, global_step, batch_size):
        del global_step
        del batch_size
        return self.learning_rate

    def forward(self, inputs, num_classes, data_format, is_training):
        raise ValueError('Must be implemented in derived classes')

    def prescaling(self, images):
        """Default images pre-scaling to [-1, 1].
        """
        with tf.name_scope('prescaling'):
            images = tf.multiply(images, 1. / 127.5)
            images = tf.subtract(images, 1.0)
            return images

    def losses(self, logits, labels, end_points):
        """Default cross entropy losses for image classification.
        """
        with tf.name_scope('xentropy'):
            losses = {}
            # Usual cross entropy
            cross_entropy = tfx.losses.sparse_softmax_cross_entropy(
                logits=logits, labels=labels,
                label_smoothing=self.label_smoothing,
                weights=1.0)
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            losses['cross_entropy'] = loss

            # Auxillary cross entropy.
            if end_points and end_points.get('AuxLogits') is not None:
                aux_logits = end_points['AuxLogits']
                with tf.name_scope('aux_xentropy'):
                    aux_cross_entropy = tfx.losses.sparse_softmax_cross_entropy(
                        logits=aux_logits, labels=labels,
                        label_smoothing=self.label_smoothing)
                    aux_loss = tf.scalar_mul(
                        self.aux_loss_weight,
                        tf.reduce_mean(aux_cross_entropy, name='aux_loss'))
                    losses['aux_cross_entropy'] = aux_loss
                    loss = tf.add_n([loss, aux_loss])
            return loss, losses

    def get_custom_getter(self):
        """Returns a custom getter that this class's methods must be called under.

        All methods of this class must be called under a variable scope that was
        passed this custom getter. Example:

        ```python
        network = ConvNetBuilder(...)
        with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
        network.conv(...)
        # Call more methods of network here
        ```

        Currently, this custom getter only does anything if self.use_tf_layers is
        True. In that case, it causes variables to be stored as dtype
        self.variable_type, then casted to the requested dtype, instead of directly
        storing the variable as the requested dtype.
        """
        def inner_custom_getter(getter, *args, **kwargs):
            """Custom getter that forces variables to have type self.variable_type."""
            # if not self.use_tf_layers:
            #     return getter(*args, **kwargs)
            requested_dtype = kwargs['dtype']
            if not (requested_dtype == tf.float32 and
                    self.variable_dtype == tf.float16):
                # Only change the variable dtype if doing so does not decrease variable
                # precision.
                kwargs['dtype'] = self.variable_dtype
            var = getter(*args, **kwargs)
            # This if statement is needed to guard the cast, because batch norm
            # assigns directly to the return value of this custom getter. The cast
            # makes the return value not a variable so it cannot be assigned. Batch
            # norm variables are always in fp32 so this if statement is never
            # triggered for them.
            if var.dtype.base_dtype != requested_dtype:
                var = tf.cast(var, requested_dtype)
            return var
        return inner_custom_getter




# =========================================================================== #
# Useful methods
# =========================================================================== #
def data_format_scope(data_format):
    """Create the default scope for a given data format.
    Tries to combine all existing layers in one place!
    """
    with slim.arg_scope([slim.conv2d,
                         slim.separable_conv2d,
                         slim.max_pool2d,
                         slim.avg_pool2d,
                         slim.batch_norm,
                         tfx.layers.separable_conv2d,
                         tfx.layers.concat_channels,
                         tfx.layers.split_channels,
                         tfx.layers.channel_to_last,
                         tfx.layers.to_nchw,
                         tfx.layers.to_nhwc,
                         tfx.layers.channel_to_hw,
                         tfx.layers.spatial_squeeze,
                         tfx.layers.spatial_mean,
                         tfx.layers.ksize_for_squeezing],
                        data_format=data_format) as sc:
        return sc