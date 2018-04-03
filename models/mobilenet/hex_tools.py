# =========================================================================== #
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
# =========================================================================== #
"""Collection of hexagonal convolutional layers.
"""
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.ops import variables as tf_variables


@add_arg_scope
def hex_average_pooling2d(
        inputs,
        kernel_size,
        outputs_collections=None,
        reuse=None,
        data_format='NHWC',
        scope=None):
    """Hex average pooling (approximation...)
    """
    with variable_scope.variable_scope(scope, 'HexAvgPooling2d', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        # Actually apply depthwise conv instead of separable conv.
        dtype = inputs.dtype.base_dtype
        kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
        radius = kernel_h // 2
        if data_format == 'NHWC':
            num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        else:
            # No NCHW for now...
            raise NotImplementedError()

        d = [1., 7., 19., 37.]
        shape = [kernel_h, kernel_w, num_filters_in, 1]
        w = tf.constant(
            1. / d[radius-1],
            dtype=dtype,
            shape=shape,
            name='Const')
        # Constant weights for average pooling.
        outputs = nn.hex_depthwise_conv2d(inputs, w,
                                          [1, 1, 1, 1], 'SAME',
                                          data_format=data_format)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)
hex_avg_pooling2d = hex_average_pooling2d
hex_avg_pool2d = hex_average_pooling2d
