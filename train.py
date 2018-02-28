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
"""Training script for TensorFlow.

See the README for more information.
"""
from __future__ import print_function

import tensorflow as tf

import models
import datasets
import parameters
import deploy

# Define the training tf.flags.FLAGS
parameters.define_flags()


def set_model_params(model, params):
    """Set model parameters, using FLAGS.
    """
    model.set_batch_size(params.batch_size)
    model.set_label_smoothing(params.label_smoothing)
    model.set_weight_decay(params.weight_decay)


def main(extra_flags):
    # Check no unknown flags was passed.
    assert len(extra_flags) >= 1
    if len(extra_flags) > 1:
        raise ValueError('Received unknown flags: %s' % extra_flags[1:])

    # Get parameters from FLAGS passed.
    params = parameters.make_params_from_flags()
    deploy.setup_env(params)
    parameters.save_params(params, params.train_dir)

    # TF log...
    tfversion = deploy.tensorflow_version_tuple()
    deploy.log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

    # Create model and dataset.
    dataset = datasets.create_dataset(
        params.data_dir, params.data_name, params.data_subset)
    model = models.create_model(params.model, dataset)
    set_model_params(model, params)

    # Run CNN trainer.
    trainer = deploy.TrainerCNN(dataset, model, params)
    trainer.print_info()
    trainer.run()


if __name__ == '__main__':
    tf.app.run()
