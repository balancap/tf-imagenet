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

# Original nets
from . import mobilenet_v1_orig, mobilenet_v1_relu, mobilenet_v2

# TEST NETS
from . import hex_mobilenet_v1, hex_mobilenet_v2, hex_rot_mobilenet_v2
from . import (
    hex_rot_mobilenet_v2a, hex_rot_mobilenet_v2b, hex_rot_mobilenet_v2c,
    hex_rot_mobilenet_v2d, hex_rot_mobilenet_v2e)


def get_imagenet_models():
    """Get the list of MobileNets models.
    """
    d = {
        'mobilenet_v1_orig': lambda: mobilenet_v1_orig.MobileNetV1(),
        'mobilenet_v1_relu': lambda: mobilenet_v1_relu.MobileNetV1(),
        'mobilenet_v2_d1': lambda: mobilenet_v2.MobileNetV2(depth_multiplier=1.0),
        'mobilenet_v2_d14': lambda: mobilenet_v2.MobileNetV2(depth_multiplier=1.4),
        'hex_mobilenet_v1': lambda: hex_mobilenet_v1.HexMobileNetV1(
            ksize=5, regularize_depthwise=False),
        'hex_mobilenet_v1_5x5': lambda: hex_mobilenet_v1.HexMobileNetV1(
            ksize=5, regularize_depthwise=False),
        'hex_mobilenet_v1_5x5_w': lambda: hex_mobilenet_v1.HexMobileNetV1(
            ksize=5, regularize_depthwise=True),
        'hex_mobilenet_v1_3x3': lambda: hex_mobilenet_v1.HexMobileNetV1(
            ksize=3, regularize_depthwise=False),
        'hex_mobilenet_v2_5x5_d1': lambda: hex_mobilenet_v2.HexMobileNetV2(
            ksize=5, regularize_depthwise=True, depth_multiplier=1.0),
        'hex_mobilenet_v2_5x5_d14': lambda: hex_mobilenet_v2.HexMobileNetV2(
            ksize=5, regularize_depthwise=True, depth_multiplier=1.4),
        'hex_rot_mobilenet_v2_5x5_d1': lambda: hex_rot_mobilenet_v2.HexMobileNetV2(
            ksize=5, regularize_depthwise=True, depth_multiplier=1.0),
        'hex_rot_mobilenet_v2_5x5_d14': lambda: hex_rot_mobilenet_v2.HexMobileNetV2(
            ksize=5, regularize_depthwise=True, depth_multiplier=1.4),
        'hex_rot_mobilenet_v2_5x5_d1a': lambda: hex_rot_mobilenet_v2a.HexMobileNetV2(
            ksize=5, regularize_depthwise=True, depth_multiplier=1.0),
        'hex_rot_mobilenet_v2_5x5_d1b': lambda: hex_rot_mobilenet_v2b.HexMobileNetV2(
            ksize=5, regularize_depthwise=True, depth_multiplier=1.0),
        'hex_rot_mobilenet_v2_5x5_d1c': lambda: hex_rot_mobilenet_v2c.HexMobileNetV2(
            ksize=5, regularize_depthwise=True, depth_multiplier=1.0),
        'hex_rot_mobilenet_v2_5x5_d1d': lambda: hex_rot_mobilenet_v2d.HexMobileNetV2(
            ksize=5, regularize_depthwise=True, depth_multiplier=1.0),
        'hex_rot_mobilenet_v2_5x5_d1e': lambda: hex_rot_mobilenet_v2e.HexMobileNetV2(
            ksize=5, regularize_depthwise=True, depth_multiplier=1.0),
    }
    return d
