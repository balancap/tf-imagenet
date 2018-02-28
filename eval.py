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
import os

import tensorflow as tf

import models
import datasets
import parameters
import train
import deploy

# Define the training tf.flags.FLAGS
parameters.define_flags()


def replace_with_train_params(params):
    """Replace some parameters with the training parameters.
    Returns the updated params.
    """
    params = params._replace(eval=True)
    train_params = parameters.load_params(params.train_dir)
    if not train_params:
        deploy.log_fn('WARNING: no training parameters file found.')
        return params

    p = params
    p = p._replace(eval_dir=os.path.join(params.train_dir, 'eval'))
    p = p._replace(eval_interval_secs=train_params.save_model_secs)

    p = p._replace(model=train_params.model)
    p = p._replace(data_dir=train_params.data_dir)
    p = p._replace(data_name=train_params.data_name)
    # Dataset subset: set validation by default if nothing specified.
    # Good also for backward compatibility.
    if p.data_subset == 'train':
        p = p._replace(data_subset='validation')

    p = p._replace(batch_size=4)
    p = p._replace(num_gpus=train_params.num_gpus)
    p = p._replace(gpu_memory_frac_for_testing=0.1)
    p = p._replace(num_intra_threads=0)
    p = p._replace(num_inter_threads=0)
    p = p._replace(summary_verbosity=1)
    p = p._replace(print_training_accuracy=True)
    p = p._replace(variable_update='parameter_server')
    p = p._replace(local_parameter_device='cpu')

    p = p._replace(data_format=train_params.data_format)
    p = p._replace(label_smoothing=train_params.label_smoothing)
    p = p._replace(weight_decay=train_params.weight_decay)
    params = p
    return params


def main(extra_flags):
    # Check no unknown flags was passed.
    assert len(extra_flags) >= 1
    if len(extra_flags) > 1:
        raise ValueError('Received unknown flags: %s' % extra_flags[1:])

    # Get parameters from FLAGS passed.
    params = parameters.make_params_from_flags()
    deploy.setup_env(params)
    # Training parameters, update using json file.
    params = replace_with_train_params(params)

    # TF log...
    tfversion = deploy.tensorflow_version_tuple()
    deploy.log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

    # Create model and dataset.
    dataset = datasets.create_dataset(
        params.data_dir, params.data_name, params.data_subset)
    model = models.create_model(params.model, dataset)
    train.set_model_params(model, params)

    # Set the number of batches to the size of the eval dataset.
    params = params._replace(
        num_batches=int(dataset.num_examples_per_epoch() / (params.batch_size * params.num_gpus)))
    # Run CNN trainer.
    trainer = deploy.TrainerCNN(dataset, model, params)
    trainer.print_info()
    trainer.run()


if __name__ == '__main__':
    tf.app.run()
