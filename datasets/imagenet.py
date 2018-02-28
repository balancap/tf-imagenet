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
