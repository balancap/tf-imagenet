# /* ===========================================================================
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
# =========================================================================== */
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
