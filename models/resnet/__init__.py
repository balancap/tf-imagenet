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

from . import resnet_v1, resnet_v2

def get_imagenet_models():
    """Get the list of ResNet models.
    """
    d = {
        'resnet_v1_50': lambda: resnet_v1.ResNet_v1_50(224),
        'resnet_v1_101': lambda: resnet_v1.ResNet_v1_101(224),
        'resnet_v1_152': lambda: resnet_v1.ResNet_v1_152(224),
        'resnet_v2_50': lambda: resnet_v2.ResNet_v2_50(224),
        'resnet_v2_101': lambda: resnet_v2.ResNet_v2_101(224),
        'resnet_v2_152': lambda: resnet_v2.ResNet_v2_152(224),
        'resnet_v2_50_299': lambda: resnet_v2.ResNet_v2_50(299),
        'resnet_v2_101_299': lambda: resnet_v2.ResNet_v2_101(299),
        'resnet_v2_152_299': lambda: resnet_v2.ResNet_v2_152(299),
    }
    return d
