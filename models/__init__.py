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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import all the models!
from .inception import inception_v1, inception_v2, inception_v3, inception_v4
# from .inception import inception_v2a, inception_v2b, inception_v2d

from .mobilenet import mobilenet_v1_orig, mobilenet_v1_relu, mobilenet_v2
from .resnet import resnet_v1, resnet_v2
from .nasnet import nasnet

# TEST NETS
from .mobilenet import hex_mobilenet_v1

IMAGENET_MODELS = {
    'googlenet': lambda: inception_v1.InceptionV1(False),
    'inception_v1': lambda: inception_v1.InceptionV1(True),
    'inception_v2': lambda: inception_v2.InceptionV2(),
    'inception_v3': lambda: inception_v3.InceptionV3(),
    'inception_v4': lambda: inception_v4.InceptionV4(),
    'mobilenet_v1_orig': lambda: mobilenet_v1_orig.MobileNetV1(),
    'mobilenet_v1_relu': lambda: mobilenet_v1_relu.MobileNetV1(),
    'mobilenet_v2_d1': lambda: mobilenet_v2.MobileNetV2(depth_multiplier=1.0),
    'mobilenet_v2_d14': lambda: mobilenet_v2.MobileNetV2(depth_multiplier=1.4),
    'resnet_v1_50': lambda: resnet_v1.ResNet_v1_50(224),
    'resnet_v1_101': lambda: resnet_v1.ResNet_v1_101(224),
    'resnet_v1_152': lambda: resnet_v1.ResNet_v1_152(224),
    'resnet_v2_50': lambda: resnet_v2.ResNet_v2_50(224),
    'resnet_v2_101': lambda: resnet_v2.ResNet_v2_101(224),
    'resnet_v2_152': lambda: resnet_v2.ResNet_v2_152(224),
    'resnet_v2_50_299': lambda: resnet_v2.ResNet_v2_50(299),
    'resnet_v2_101_299': lambda: resnet_v2.ResNet_v2_101(299),
    'resnet_v2_152_299': lambda: resnet_v2.ResNet_v2_152(299),
    'nasnet_v1_small': lambda: nasnet.NASNetV1Small(),
    'nasnet_v1_large': lambda: nasnet.NASNetV1Large(),
}

IMAGENET_MODELS.update({
    'hex_mobilenet_v1': lambda: hex_mobilenet_v1.HexMobileNetV1(
        ksize=5, regularize_depthwise=False),
    'hex_mobilenet_v1_5x5': lambda: hex_mobilenet_v1.HexMobileNetV1(
        ksize=5, regularize_depthwise=False),
    'hex_mobilenet_v1_5x5_w': lambda: hex_mobilenet_v1.HexMobileNetV1(
        ksize=5, regularize_depthwise=True),
    'hex_mobilenet_v1_3x3': lambda: hex_mobilenet_v1.HexMobileNetV1(
        ksize=3, regularize_depthwise=False),
})

def create_model(model_name, dataset):
    """Create a model from a name and a dataset.
    """
    models_dict = IMAGENET_MODELS

    if model_name not in models_dict.keys():
        raise ValueError("Unknown model '%s' in collection '%s'." % (model_name, IMAGENET_MODELS.keys()))

    # Build model and set some parameters.
    mc = models_dict[model_name]()

    # Always used training set as reference for epoch size.
    mc.set_epoch_size(dataset.num_examples_per_epoch())
    return mc
