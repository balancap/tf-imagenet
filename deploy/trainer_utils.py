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
"""Trainer utils.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# ============================================================================ #
# Various helpful methods used during training.
# ============================================================================ #
class CheckpointNotFoundException(Exception):
    pass

def restore_checkpoint_fn(ckpt_filename, global_step,
                          ckpt_scope=None, moving_average_decay=None):
    """Restore variables from a checkpoint file. Using either normal values
    or moving averaged values.
    """
    # Restore moving average variables or classic stuff!
    if moving_average_decay:
        print('Restoring moving average variables.')
        variable_averages = tf.train.ExponentialMovingAverage(
            moving_average_decay, global_step)
        variables_to_restore = variable_averages.variables_to_restore(
            tf.contrib.framework.get_model_variables())
        # variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
        print('Restoring last batch variables.')
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        # Convert list to dict.
        variables_to_restore = {v.op.name: v for v in variables_to_restore}
    # Update the scope of variables.
    if ckpt_scope:
        scopes = ckpt_scope.split(':')
        variables_to_restore = {k.replace(scopes[0], scopes[1]): v
            for k, v in variables_to_restore.items()}

    # Replace v{i}/cg/ by vo/cg. UGLY HACK!
    for i in range(1, 8):
        variables_to_restore = {k.replace('v%i/cg/' % i, 'v0/cg/'): v
            for k, v in variables_to_restore.items()}

    # Restore method.
    # variables_to_restore = {}
    fn_restore = tf.contrib.framework.assign_from_checkpoint_fn(
        ckpt_filename, variables_to_restore, ignore_missing_vars=True)
    return fn_restore


def load_checkpoint(saver, sess, ckpt_dir, ckpt_scope=None, moving_average_decay=None):
    latest_filename = None
    # Is this a checkpoint file?
    if (os.path.isfile(ckpt_dir) or
            os.path.isfile(ckpt_dir + '.index') or
            os.path.isfile(ckpt_dir + '.meta')):
        # Load directly!
        global_step = 0
        fn_restore = restore_checkpoint_fn(
            ckpt_dir, global_step, ckpt_scope, moving_average_decay)
        fn_restore(sess)
        log_fn('Successfully loaded model from %s.' % ckpt_dir)
        return global_step

    # Directory: get the checkpoints there.
    ckpt = tf.train.get_checkpoint_state(ckpt_dir, latest_filename)
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
            # Restores from checkpoint with absolute path.
            model_checkpoint_path = ckpt.model_checkpoint_path
        else:
            # Restores from checkpoint with relative path.
            model_checkpoint_path = os.path.join(ckpt_dir, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        if not global_step.isdigit():
            global_step = 0
        else:
            global_step = int(global_step)
        # saver.restore(sess, model_checkpoint_path)
        fn_restore = restore_checkpoint_fn(
            model_checkpoint_path, global_step, ckpt_scope, moving_average_decay)
        fn_restore(sess)
        log_fn('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)
        return global_step
    else:
        raise CheckpointNotFoundException('No checkpoint file found.')


def setup_env(params):
    """Sets up the environment that TrainerCNN should run in.

    Args:
        params: Params tuple, typically created by make_params or make_params_from_flags.
    """
    if params.winograd_nonfused:
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    else:
        os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
    if params.autotune_threshold:
        os.environ['TF_AUTOTUNE_THRESHOLD'] = str(params.autotune_threshold)
    os.environ['TF_SYNC_ON_FINISH'] = str(int(params.sync_on_finish))
    argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Sets environment variables for MKL
    if params.mkl:
        os.environ['KMP_BLOCKTIME'] = str(params.kmp_blocktime)
        os.environ['KMP_SETTINGS'] = str(params.kmp_settings)
        os.environ['KMP_AFFINITY'] = params.kmp_affinity
        if params.num_intra_threads > 0:
            os.environ['OMP_NUM_THREADS'] = str(params.num_intra_threads)

def tensorflow_version_tuple():
    v = tf.__version__
    major, minor, patch = v.split('.')
    return (int(major), int(minor), patch)


def tensorflow_version():
    vt = tensorflow_version_tuple()
    return vt[0] * 1000 + vt[1]


def log_fn(log):
    print(log)
    if FLAGS.flush_stdout:
        sys.stdout.flush()
