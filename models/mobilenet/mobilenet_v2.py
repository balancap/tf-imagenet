# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""MobileNet v2.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
head (for example: embeddings, localization and classification).

As described in https://arxiv.org/pdf/1801.04381.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf
import tf_extended as tfx

from models import abstract_model

slim = tf.contrib.slim

# =========================================================================== #
# MobileNet model.
# =========================================================================== #
class MobileNetV2(abstract_model.Model):
    """MobileNetV2 model.
    """
    def __init__(self, depth_multiplier=1.0):
        self.scope = 'MobileNetV2'
        self.depth_multiplier = depth_multiplier
        super(MobileNetV2, self).__init__(self.scope, 224, 32, 0.005)

    def forward(self, inputs, num_classes, data_format, is_training):
        sc = mobilenet_v2_arg_scope(
            is_training=is_training,
            data_format=data_format,
            weight_decay=self.weight_decay,
            use_batch_norm=True,
            batch_norm_decay=0.9997,
            batch_norm_epsilon=0.001,
            regularize_depthwise=False)
        with slim.arg_scope(sc):
            logits, end_points = mobilenet_v2(
                inputs,
                num_classes=num_classes,
                dropout_keep_prob=0.8,
                is_training=is_training,
                min_depth=8,
                depth_multiplier=self.depth_multiplier,
                conv_defs=None,
                prediction_fn=tf.contrib.layers.softmax,
                spatial_squeeze=True,
                reuse=None,
                scope=self.scope,
                global_pool=False)
            return logits, end_points


# =========================================================================== #
# Functional definition.
# =========================================================================== #

# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'factor'])
Bottleneck = namedtuple('Bottleneck', ['kernel', 'stride', 'depth', 'factor'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32, factor=1),
    Bottleneck(kernel=[3, 3], stride=1, depth=16, factor=1),

    Bottleneck(kernel=[3, 3], stride=2, depth=24, factor=6),
    Bottleneck(kernel=[3, 3], stride=1, depth=24, factor=6),

    Bottleneck(kernel=[3, 3], stride=2, depth=32, factor=6),
    Bottleneck(kernel=[3, 3], stride=1, depth=32, factor=6),
    Bottleneck(kernel=[3, 3], stride=1, depth=32, factor=6),

    Bottleneck(kernel=[3, 3], stride=2, depth=64, factor=6),
    Bottleneck(kernel=[3, 3], stride=1, depth=64, factor=6),
    Bottleneck(kernel=[3, 3], stride=1, depth=64, factor=6),
    Bottleneck(kernel=[3, 3], stride=1, depth=64, factor=6),

    Bottleneck(kernel=[3, 3], stride=1, depth=96, factor=6),
    Bottleneck(kernel=[3, 3], stride=1, depth=96, factor=6),
    Bottleneck(kernel=[3, 3], stride=1, depth=96, factor=6),

    Bottleneck(kernel=[3, 3], stride=2, depth=160, factor=6),
    Bottleneck(kernel=[3, 3], stride=1, depth=160, factor=6),
    Bottleneck(kernel=[3, 3], stride=1, depth=160, factor=6),

    Bottleneck(kernel=[3, 3], stride=1, depth=320, factor=6),
    Conv(kernel=[1, 1], stride=1, depth=1280, factor=1)
]

def mobilenet_v2_base(inputs,
                      final_endpoint='Conv2d_18',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      scope=None):
    """Mobilenet v2.

    Constructs a Mobilenet v2 network from inputs to the given final endpoint.

    Args:
        inputs: a tensor of shape [batch_size, height, width, channels].
        final_endpoint: specifies the endpoint to construct the network up to. It
            can be one of ['Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',
            'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5'_pointwise,
            'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',
            'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',
            'Conv2d_12_pointwise', 'Conv2d_13_pointwise'].
        min_depth: Minimum depth value (number of channels) for all convolution ops.
            Enforced when depth_multiplier < 1, and not an active constraint when
            depth_multiplier >= 1.
        depth_multiplier: Float multiplier for the depth (number of channels)
            for all convolution ops. The value must be greater than zero. Typical
            usage will be to set this value in (0, 1) to reduce the number of
            parameters or computation cost of the model.
        conv_defs: A list of ConvDef namedtuples specifying the net architecture.
        output_stride: An integer that specifies the requested ratio of input to
            output spatial resolution. If not None, then we invoke atrous convolution
            if necessary to prevent the network from reducing the spatial resolution
            of the activation maps. Allowed values are 8 (accurate fully convolutional
            mode), 16 (fast fully convolutional mode), 32 (classification mode).
        scope: Optional variable_scope.

    Returns:
        tensor_out: output tensor corresponding to the final_endpoint.
        end_points: a set of activations for external use, for example summaries or
                                losses.

    Raises:
        ValueError: if final_endpoint is not set to one of the predefined values,
                                or depth_multiplier <= 0, or the target output_stride is not
                                allowed.
    """
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    with tf.variable_scope(scope, 'MobilenetV2', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            # The current_stride variable keeps track of the output stride of the
            # activations, i.e., the running product of convolution strides up to the
            # current network layer. This allows us to invoke atrous convolution
            # whenever applying the next convolution would result in the activations
            # having output stride larger than the target output_stride.
            current_stride = 1

            # The atrous convolution rate parameter.
            rate = 1
            net = inputs
            in_depth = 3
            for i, conv_def in enumerate(conv_defs):
                end_point_base = 'Conv2d_%d' % i
                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride

                # Normal conv2d.
                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                                      stride=conv_def.stride, scope=end_point)
                    end_points[end_point] = net

                # Bottleneck block.
                elif isinstance(conv_def, Bottleneck):
                    # Stride > 1 or different depth: no residual part.
                    # in_depth = tfx.layers.channel_dimension(net.get_shape())
                    res = net if layer_stride == 1 and in_depth == conv_def.depth else None

                    # Increase depth with 1x1 conv.
                    end_point = end_point_base + '_up_pointwise'
                    net = slim.conv2d(net, depth(in_depth * conv_def.factor),
                                      [1, 1], stride=1, scope=end_point)
                    end_points[end_point] = net
                    # Depthwise conv2d.
                    end_point = end_point_base + '_depthwise'
                    net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                depth_multiplier=1,
                                                stride=layer_stride,
                                                rate=layer_rate,
                                                scope=end_point)
                    end_points[end_point] = net
                    # Downscale 1x1 conv.
                    end_point = end_point_base + '_down_pointwise'
                    net = slim.conv2d(net, depth(conv_def.depth),
                                      [1, 1], activation_fn=None,
                                      stride=1, scope=end_point)
                    # Residual connection?
                    # print(net, res, end_point, in_depth, conv_def.depth)
                    end_point = end_point_base + '_residual'
                    net = tf.add(res, net, name=end_point) if res is not None else net
                    end_points[end_point] = net

                # Unknown...
                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))
                in_depth = conv_def.depth
                # Final end point?
                if final_endpoint in end_points:
                    return end_points[final_endpoint], end_points

    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def mobilenet_v2(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobilenetV2',
                 global_pool=False):
    """Mobilenet v2 model for classification.

    Args:
        inputs: a tensor of shape [batch_size, height, width, channels].
        num_classes: number of predicted classes. If 0 or None, the logits layer
            is omitted and the input features to the logits layer (before dropout)
            are returned instead.
        dropout_keep_prob: the percentage of activation values that are retained.
        is_training: whether is training or not.
        min_depth: Minimum depth value (number of channels) for all convolution ops.
            Enforced when depth_multiplier < 1, and not an active constraint when
            depth_multiplier >= 1.
        depth_multiplier: Float multiplier for the depth (number of channels)
            for all convolution ops. The value must be greater than zero. Typical
            usage will be to set this value in (0, 1) to reduce the number of
            parameters or computation cost of the model.
        conv_defs: A list of ConvDef namedtuples specifying the net architecture.
        prediction_fn: a function to get predictions out of logits.
        spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
                of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
        reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
        scope: Optional variable_scope.
        global_pool: Optional boolean flag to control the avgpooling before the
            logits layer. If false or unset, pooling is done with a fixed window
            that reduces default-sized inputs to 1x1, while larger inputs lead to
            larger outputs. If true, any input size is pooled down to 1x1.

    Returns:
        net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
            is a non-zero integer, or the non-dropped-out input to the logits layer
            if num_classes is 0 or None.
        end_points: a dictionary from components of the network to the corresponding
            activation.

    Raises:
        ValueError: Input rank is invalid.
    """
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))

    with tf.variable_scope(scope, 'MobilenetV2', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = mobilenet_v2_base(inputs, scope=scope,
                                                min_depth=min_depth,
                                                depth_multiplier=depth_multiplier,
                                                conv_defs=conv_defs)
            with tf.variable_scope('Logits'):
                if global_pool:
                    # Global average pooling.
                    net = tfx.layers.spatial_mean(net, keep_dims=True, scope='global_pool')
                    end_points['global_pool'] = net
                else:
                    # Pooling with a fixed kernel size.
                    ksize = tfx.layers.ksize_for_squeezing(net, 7)
                    net = slim.avg_pool2d(net, ksize, padding='VALID',
                                          scope='AvgPool_1a')
                    end_points['AvgPool_1a'] = net
                if not num_classes:
                    return net, end_points
                # 1 x 1 x 1280
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tfx.layers.spatial_squeeze(logits, scope='SpatialSqueeze')
            end_points['Logits'] = logits
            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points

mobilenet_v2.default_image_size = 224


def mobilenet_v2_arg_scope(is_training=True,
                           data_format='NHWC',
                           weight_decay=0.00004,
                           use_batch_norm=True,
                           batch_norm_decay=0.9997,
                           batch_norm_epsilon=0.001,
                           regularize_depthwise=False):
    """Defines the default MobilenetV2 arg scope.

    Args:
        is_training: Whether or not we're training the model.
        weight_decay: The weight decay to use for regularizing the model.
        stddev: The standard deviation of the trunctated normal weight initializer.
        regularize_depthwise: Whether or not apply regularization on depthwise.

    Returns:
        An `arg_scope` to use for the mobilenet v2 model.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': True,
        'scale': True,
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
        mode='FAN_AVG')
    if regularize_depthwise:
        depthwise_regularizer = weights_regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_initializer,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer):
                    # Data format scope...
                    data_sc = abstract_model.data_format_scope(data_format)
                    with slim.arg_scope(data_sc) as sc:
                        return sc
