# ==============================================================================
# Copyright 2018 Paul Balanca. All Rights Reserved.
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
from .abstract_dataset import Dataset, SyntheticData

from .imagenet import ImagenetData, IMAGENET_NUM_TRAIN_IMAGES, IMAGENET_NUM_VAL_IMAGES
from .cifar10 import Cifar10Data, CIFAR10_NUM_TRAIN_IMAGES, CIFAR10_NUM_VAL_IMAGES


def create_dataset(data_dir, data_name, data_subset):
    """Create a Dataset instance based on data_dir and data_name.
    """

    supported_datasets = {
        'synthetic': SyntheticData,
        'imagenet': ImagenetData,
        'cifar10': Cifar10Data,
    }
    if not data_dir:
        print('WARNING: no dataset directory provided, using SYNTHETIC dataset.')
        data_name = 'synthetic'

    if data_name is None:
        for supported_name in supported_datasets:
            if supported_name in data_dir.lower():
                data_name = supported_name
                break

    if data_name is None:
        raise ValueError('Could not identify name of dataset. '
                         'Please specify with --data_name option.')
    if data_name not in supported_datasets:
        raise ValueError('Unknown dataset. Must be one of %s', ', '.join(
            [key for key in sorted(supported_datasets.keys())]))
    return supported_datasets[data_name](data_dir, data_subset)
