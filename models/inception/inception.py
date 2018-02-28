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
"""Brings all inception models under one namespace."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import model

# pylint: disable=unused-import
# from models_slim.inception_resnet_v2 import inception_resnet_v2
# from models_slim.inception_resnet_v2 import inception_resnet_v2_arg_scope
from models_slim.inception_v1 import inception_v1
from models_slim.inception_v1 import inception_v1_arg_scope
from models_slim.inception_v1 import inception_v1_base
from models_slim.inception_v2 import inception_v2
from models_slim.inception_v2 import inception_v2_arg_scope
from models_slim.inception_v2 import inception_v2_base
from models_slim.inception_v3 import inception_v3
from models_slim.inception_v3 import inception_v3_arg_scope
from models_slim.inception_v3 import inception_v3_base
from models_slim.inception_v4 import inception_v4
from models_slim.inception_v4 import inception_v4_arg_scope
from models_slim.inception_v4 import inception_v4_base
# pylint: enable=unused-import
from models_slim.inception_utils import inception_pre_rescaling

slim = tf.contrib.slim


# =========================================================================== #
# Inception classes.
# =========================================================================== #
class Inceptionv1Model(model.Model):

    def __init__(self):
        super(Inceptionv1Model, self).__init__('inception1', 224, 32, 0.005)

    def inference(self, images, num_classes,
                  is_training=True, data_format='NCHW', data_type=tf.float32):
        # Define VGG using functional slim definition
        arg_scope = inception_v1_arg_scope(is_training=is_training, data_format=data_format)
        with slim.arg_scope(arg_scope):
            return inception_v1(images, num_classes, is_training=is_training)

    def pre_rescaling(self, images, is_training=True):
        return inception_pre_rescaling(images, is_training)


class Inceptionv2Model(model.Model):

    def __init__(self):
        super(Inceptionv2Model, self).__init__('inception2', 224, 32, 0.005)

    def inference(self, images, num_classes,
                  is_training=True, data_format='NCHW', data_type=tf.float32):
        # Define VGG using functional slim definition
        arg_scope = inception_v2_arg_scope(is_training=is_training, data_format=data_format)
        with slim.arg_scope(arg_scope):
            return inception_v2(images, num_classes, is_training=is_training)

    def pre_rescaling(self, images, is_training=True):
        return inception_pre_rescaling(images, is_training)


class Inceptionv3Model(model.Model):

    def __init__(self):
        super(Inceptionv3Model, self).__init__('inception3', 299, 32, 0.005)

    def inference(self, images, num_classes,
                  is_training=True, data_format='NCHW', data_type=tf.float32):
        # Define VGG using functional slim definition
        arg_scope = inception_v3_arg_scope(is_training=is_training, data_format=data_format)
        with slim.arg_scope(arg_scope):
            return inception_v3(images, num_classes, is_training=is_training)

    def pre_rescaling(self, images, is_training=True):
        return inception_pre_rescaling(images, is_training)


class Inceptionv4Model(model.Model):
    def __init__(self):
        super(Inceptionv4Model, self).__init__('inception4', 299, 32, 0.005)

    def inference(self, images, num_classes,
                  is_training=True, data_format='NCHW', data_type=tf.float32):
        # Define VGG using functional slim definition
        arg_scope = inception_v4_arg_scope(is_training=is_training, data_format=data_format)
        with slim.arg_scope(arg_scope):
            return inception_v4(images, num_classes, is_training=is_training)

    def pre_rescaling(self, images, is_training=True):
        return inception_pre_rescaling(images, is_training)


# class GooglenetModel(model.Model):

#     def __init__(self):
#         super(GooglenetModel, self).__init__('googlenet', 224, 32, 0.005)
