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

from . import inception_v1, inception_v2, inception_v3, inception_v4
# from . import inception_v2a, inception_v2b, inception_v2d

def get_imagenet_models():
    """Get the list of Inception models.
    """
    d = {
        'googlenet': lambda: inception_v1.InceptionV1(False),
        'inception_v1': lambda: inception_v1.InceptionV1(True),
        'inception_v2': lambda: inception_v2.InceptionV2(),
        'inception_v3': lambda: inception_v3.InceptionV3(),
        'inception_v4': lambda: inception_v4.InceptionV4(),
    }
    return d
