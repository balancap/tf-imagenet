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

from . import nasnet

def get_imagenet_models():
    """Get the list of ResNet models.
    """
    d = {
        'nasnet_v1_small': lambda: nasnet.NASNetV1Small(),
        'nasnet_v1_large': lambda: nasnet.NASNetV1Large(),
    }
    return d
