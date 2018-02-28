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
"""Definition of the imagenet dataset.
"""
import preprocessing
from . import abstract_dataset

IMAGENET_NUM_TRAIN_IMAGES = 1281167
IMAGENET_NUM_VAL_IMAGES = 50000
IMAGENET_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000
}

class ImagenetData(abstract_dataset.Dataset):
    """Configuration for Imagenet dataset.
    """
    def __init__(self, data_dir=None, subset='train'):
        if data_dir is None:
            raise ValueError('Data directory not specified')
        super(ImagenetData, self).__init__(
            'imagenet', subset, 300, 300, data_dir=data_dir)

    def num_examples_per_epoch(self, subset=None):
        subset = subset if subset else self.subset
        if subset not in IMAGENET_NUM_IMAGES.keys():
            raise ValueError('Invalid data subset "%s"' % subset)
        return IMAGENET_NUM_IMAGES[subset]

    def get_image_preprocessor(self):
        return preprocessing.RecordInputImagePreprocessor
