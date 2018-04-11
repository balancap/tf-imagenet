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

import pprint

# Import all the models!
from . import inception
from . import nasnet
from . import resnet
from . import mobilenet

IMAGENET_MODELS = {}
IMAGENET_MODELS.update(inception.get_imagenet_models())
IMAGENET_MODELS.update(nasnet.get_imagenet_models())
IMAGENET_MODELS.update(resnet.get_imagenet_models())
IMAGENET_MODELS.update(mobilenet.get_imagenet_models())


def create_model(model_name, dataset):
    """Create a model from a name and a dataset.
    """
    models_dict = IMAGENET_MODELS

    if model_name not in models_dict.keys():
        models_desc = pprint.pformat(list(IMAGENET_MODELS.keys()))
        raise ValueError("Unknown model '%s' in collection:\n%s." % (model_name, models_desc))

    # Build model and set some parameters.
    mc = models_dict[model_name]()

    # Always used training set as reference for epoch size.
    mc.set_epoch_size(dataset.num_examples_per_epoch())
    return mc
