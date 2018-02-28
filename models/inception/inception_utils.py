# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains common code shared by all inception models.

Usage of arg scope:
    with slim.arg_scope(inception_arg_scope()):
        logits, end_points = inception.inception_v3(images, num_classes,
                                                    is_training=is_training)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tf_extended as tfx
# from models_slim import custom_layers

slim = tf.contrib.slim


def inception_pre_rescaling(images, is_training):
    """Rescales an images Tensor before feeding the network
    Input tensor supposed to be in [0, 256) range.
    """
    # Rescale to [-1,1] instead of [0, 1)
    # images *= 1. / 255
    # images = tf.subtract(images, 0.5)
    # images = tf.multiply(images, 2.0)
    images -= 127.5
    images *= 1. / 127.5
    return images


def inception_arg_scope(weight_decay=0.00004,
                        data_format='NHWC',
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        is_training=True):
    """Defines the default arg scope for inception models.

    Args:
        weight_decay: The weight decay to use for regularizing the model.
        use_batch_norm: "If `True`, batch_norm is applied after each convolution.
        batch_norm_decay: Decay for batch norm moving average.
        batch_norm_epsilon: Small float added to variance to avoid dividing by zero
            in batch norm.

    Returns:
        An `arg_scope` to use for the inception models.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': True,
        'scale': False,
        'data_format': data_format,
        'is_training': is_training,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}

    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode='FAN_OUT')
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.fully_connected, slim.conv2d, slim.separable_conv2d],
                        weights_regularizer=weights_regularizer,
                        weights_initializer=weights_initializer):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params):
            with slim.arg_scope([slim.dropout,
                                 tfx.layers.drop_path],
                                is_training=is_training):
                with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                    # Data format scope...
                    with slim.arg_scope([slim.conv2d,
                                        slim.max_pool2d,
                                        slim.avg_pool2d,
                                        slim.batch_norm,
                                        tfx.layers.separable_conv2d,
                                        tfx.layers.concat_channels,
                                        tfx.layers.split_channels,
                                        tfx.layers.channel_to_last,
                                        tfx.layers.spatial_squeeze,
                                        tfx.layers.spatial_mean,
                                        tfx.layers.ksize_for_squeezing],
                                        data_format=data_format) as sc:
                        return sc


def aux_head_logits(inputs,
                    num_classes,
                    spatial_squeeze,
                    depth_multiplier=1.0,
                    min_depth=16):
    """Standard auxillary head logits used in Inception models.
    """
    if inputs is None:
        return None

    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    with tf.variable_scope('AuxLogits'):
        aux_logits = slim.avg_pool2d(
            inputs, [5, 5], stride=3, padding='VALID',
            scope='AvgPool_1a_5x5')
        aux_logits = slim.conv2d(aux_logits, depth(128), [1, 1],
                                 scope='Conv2d_1b_1x1')
        # Shape of feature map before the final layer.
        ksize = tfx.layers.ksize_for_squeezing(aux_logits, 5)
        aux_logits = slim.conv2d(
            aux_logits, depth(768), ksize,
            padding='VALID', scope='Conv2d_2a_{}x{}'.format(*ksize))
        aux_logits = slim.conv2d(
            aux_logits, num_classes, [1, 1],
            activation_fn=None, normalizer_fn=None,
            scope='Conv2d_2b_1x1')
        if spatial_squeeze:
            aux_logits = tfx.layers.spatial_squeeze(aux_logits, scope='SpatialSqueeze')
        return aux_logits


def apply_drop_path(inputs,
                    drop_path_keep_prob,
                    layer_ratio=1.0,
                    total_training_steps=350000):
    """Apply drop_path regularization to a net.
    """
    net = inputs
    if drop_path_keep_prob < 1.0:
        # with tf.device('/cpu:0'):
        #     tf.summary.scalar('layer_ratio', layer_ratio)
        drop_path_keep_prob = 1 - layer_ratio * (1 - drop_path_keep_prob)
        # Decrease the keep probability over time
        current_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
        drop_path_burn_in_steps = tf.cast(total_training_steps, tf.float32)
        current_ratio = current_step / drop_path_burn_in_steps
        current_ratio = tf.minimum(1.0, current_ratio)
        # with tf.device('/cpu:0'):
        #     tf.summary.scalar('current_ratio', current_ratio)
        drop_path_keep_prob = (
            1 - current_ratio * (1 - drop_path_keep_prob))
        # with tf.device('/cpu:0'):
        #     tf.summary.scalar('drop_path_keep_prob', drop_path_keep_prob)
        net = tfx.layers.drop_path(net, drop_path_keep_prob)
    return net
