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
"""Main parameters used in training and evaluation loops.

Params are passed to training and evalution loops. Params is a map from name
to value, with one field per key in DEFAULT_PARAMS.

Call make_params() or make_params_from_flags() below to construct a Params
tuple with default values from DEFAULT_PARAMS, rather than constructing
Params directly.
"""
import os
import six
import json
from collections import namedtuple

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# =========================================================================== #
# Default parameters.
# =========================================================================== #
# ParamSpec describes one parameter. _ParamSpec is the value
# type for _DEFAULT_PARAMS below.
_ParamSpec = namedtuple('_ParamSpec',
                        ['flag_type', 'default_value', 'description'])
# DEFAULT_PARAMS maps from each parameter's name to its _ParamSpec.
DEFAULT_PARAMS = {
    # ======================================================================== #
    # general parameters (model, batch size, ...)
    # ======================================================================== #
    'model':
        _ParamSpec('string', 'trivial', 'name of the model to run'),

    # The code will first check if it's running under benchmarking mode
    # or evaluation mode, depending on 'eval':
    # Under the evaluation mode, this script will read a saved model,
    #   and compute the accuracy of the model against a validation dataset.
    #   Additional ops for accuracy and top_k predictors are only used under
    #   this mode.
    # Under the benchmarking mode, user can specify whether nor not to use
    #   the forward-only option, which will only compute the loss function.
    #   forward-only cannot be enabled with eval at the same time.
    'eval':
        _ParamSpec('boolean', False, 'whether use eval or benchmarking'),
    'eval_interval_secs':
        _ParamSpec('integer', 0,
                   'How often to run eval on saved checkpoints. Usually the '
                   'same as save_model_secs from the corresponding training '
                   'run. Pass 0 to eval only once.'),
    'forward_only':
        _ParamSpec('boolean', False,
                   'whether use forward-only or training for benchmarking'),
    'print_training_accuracy':
        _ParamSpec('boolean', False,
                   'whether to calculate and print training accuracy during '
                   'training'),
    'batch_size':
        _ParamSpec('integer', 0, 'batch size per compute device'),
    'batch_group_size':
        _ParamSpec('integer', 1,
                   'number of groups of batches processed in the image '
                   'producer.'),
    'num_batches':
        _ParamSpec('integer', 100, 'number of batches to run, excluding '
                   'warmup'),
    'num_warmup_batches':
        _ParamSpec('integer', None, 'number of batches to run before timing'),
    'autotune_threshold':
        _ParamSpec('integer', None, 'The autotune threshold for the models'),
    'num_gpus':
        _ParamSpec('integer', 1, 'the number of GPUs to run on'),
    'gpu_indices':
        _ParamSpec('string', '', 'indices of worker GPUs in ring order'),
    'display_every':
        _ParamSpec('integer', 10,
                   'Number of local steps after which progress is printed out'),
    'moving_average_decay':
        _ParamSpec('float', None,
                   'Moving average decay for weights smoothing.'),
    # ======================================================================== #
    # datasets + pre-processing + device + threads
    # ======================================================================== #
    'data_dir':
        _ParamSpec('string', None,
                   'Path to dataset in TFRecord format (aka Example '
                   'protobufs). If not specified, synthetic data will be '
                   'used.'),
    'data_name':
        _ParamSpec('string', 'None',
                   'Name of dataset: imagenet or cifar10. If not specified, it '
                   'is automatically guessed based on data_dir.'),
    'data_subset':
        _ParamSpec('string', 'train',
                   'Dataset subset (usually train or validation).'),
    'resize_method':
        _ParamSpec('string', 'bilinear',
                   'Method for resizing input images: crop, nearest, bilinear, '
                   'bicubic, area, or round_robin. The `crop` mode requires '
                   'source images to be at least as large as the network input '
                   'size. The `round_robin` mode applies different resize '
                   'methods based on position in a batch in a round-robin '
                   'fashion. Other modes support any sizes and apply random '
                   'bbox distortions before resizing (even with '
                   'distortions=False).'),
    'distortions':
        _ParamSpec('boolean', True,
                   'Enable/disable distortions during image preprocessing. '
                   'These include bbox and color distortions.'),
    'use_datasets':
        _ParamSpec('boolean', True,
                   'Enable use of datasets for input pipeline'),
    'cache_data':
        _ParamSpec('boolean', False,
                   'Enable use of0a special datasets pipeline that reads a '
                   'single TFRecord into memory and repeats it infinitely many '
                   'times. The purpose of this flag is to make it possible '
                   'to write regression tests that are not bottlenecked by CNS '
                   'throughput.'),
    'local_parameter_device':
        _ParamSpec('string', 'gpu',
                   'Device to use as parameter server: cpu or gpu. For '
                   'distributed training, it can affect where caching of '
                   'variables happens.'),
    'device':
        _ParamSpec('string', 'gpu',
                   'Device to use for computation: cpu or gpu'),
    'data_format':
        _ParamSpec('string', 'NCHW',
                   'Data layout to use: NHWC (TF native) or NCHW (cuDNN '
                   'native, requires GPU).'),
    'num_intra_threads':
        _ParamSpec('integer', 1,
                   'Number of threads to use for intra-op parallelism. If set '
                   'to 0, the system will pick an appropriate number.'),
    'num_inter_threads':
        _ParamSpec('integer', 0,
                   'Number of threads to use for inter-op parallelism. If set '
                   'to 0, the system will pick an appropriate number.'),
    'trace_file':
        _ParamSpec('string', None,
                   'Enable TensorFlow tracing and write trace to this file.'),
    'graph_file':
        _ParamSpec('string', None,
                   'Write the model\'s graph definition to this file. Defaults '
                   'to binary format unless filename ends in `txt`.'),
    # ======================================================================== #
    # optimizer parameters
    # ======================================================================== #
    'optimizer':
        _ParamSpec('string', 'sgd',
                   'Optimizer to use: momentum or sgd or rmsprop'),
    'learning_rate':
        _ParamSpec('float', None, 'Initial learning rate for training.'),
    'num_epochs_per_decay':
        _ParamSpec('float', 0,
                   'Steps after which learning rate decays. If 0, the learning '
                   'rate does not decay.'),
    'learning_rate_decay_factor':
        _ParamSpec('float', 0,
                   'Learning rate decay factor. Decay by this factor every '
                   '`num_epochs_per_decay` epochs. If 0, learning rate does '
                   'not decay.'),
    'minimum_learning_rate':
        _ParamSpec('float', 0,
                   'The minimum learning rate. The learning rate will '
                   'never decay past this value. Requires `learning_rate`, '
                   '`num_epochs_per_decay` and `learning_rate_decay_factor` to '
                   'be set.'),
    'momentum':
        _ParamSpec('float', 0.9, 'Momentum for training.'),
    'rmsprop_decay':
        _ParamSpec('float', 0.9, 'Decay term for RMSProp.'),
    'rmsprop_momentum':
        _ParamSpec('float', 0.9, 'Momentum in RMSProp.'),
    'rmsprop_epsilon':
        _ParamSpec('float', 1.0, 'Epsilon term for RMSProp.'),
    'gradient_clip':
        _ParamSpec('float', None,
                   'Gradient clipping magnitude. Disabled by default.'),
    'weight_decay':
        _ParamSpec('float', 0.00004, 'Weight decay factor for training.'),
    'label_smoothing':
        _ParamSpec('float', 0.0, 'Label smoothing in cross entropy.'),
    # ======================================================================== #
    # misc. stuff
    # ======================================================================== #
    'gpu_memory_frac_for_testing':
        _ParamSpec('float', 0,
                   'If non-zero, the fraction of GPU memory that will be used. '
                   'Useful for testing the benchmark script, as this allows '
                   'distributed mode to be run on a single machine. For '
                   'example, if there are two tasks, each can be allocated '
                   '~40 percent of the memory on a single machine'),
    'use_tf_layers':
        _ParamSpec('boolean', True,
                   'If True, use tf.layers for neural network layers. This '
                   'should not affect performance or accuracy in any way.'),
    'tf_random_seed':
        _ParamSpec('integer', 1234,
                   'The TensorFlow random seed. Useful for debugging NaNs, as '
                   'this can be set to various values to see if the NaNs '
                   'depend on the seed.'),

    # ======================================================================== #
    # Performance tuning
    # ======================================================================== #
    'winograd_nonfused':
        _ParamSpec('boolean', True,
                   'Enable/disable using the Winograd non-fused algorithms.'),
    'sync_on_finish':
        _ParamSpec('boolean', False,
                   'Enable/disable whether the devices are synced after each '
                   'step.'),
    'staged_vars':
        _ParamSpec('boolean', False,
                   'whether the variables are staged from the main '
                   'computation'),
    'force_gpu_compatible':
        _ParamSpec('boolean', True,
                   'whether to enable force_gpu_compatible in GPU_Options'),
    'xla':
        _ParamSpec('boolean', False, 'whether to enable XLA'),
    'fuse_decode_and_crop':
        _ParamSpec('boolean', True,
                   'Fuse decode_and_crop for image preprocessing.'),
    'distort_color_in_yiq':
        _ParamSpec('boolean', False,
                   'Distort color of input images in YIQ space.'),
    # ======================================================================== #
    # Performance tuning (MKL)
    # ======================================================================== #
    'mkl':
        _ParamSpec('boolean', False, 'If true, set MKL environment variables.'),
    'kmp_blocktime':
        _ParamSpec('integer', 30,
                   'The time, in milliseconds, that a thread should wait, '
                   'after completing the execution of a parallel region, '
                   'before sleeping'),
    'kmp_affinity':
        _ParamSpec('string', 'granularity=fine,verbose,compact,1,0',
                   'Restricts execution of certain threads (virtual execution '
                   'units) to a subset of the physical processing units in a '
                   'multiprocessor computer.'),
    'kmp_settings':
        _ParamSpec('integer', 1, 'If set to 1, MKL settings will be printed.'),
    # ======================================================================== #
    # FP16 computation. If use_fp16=False, no other fp16 parameters apply.
    # ======================================================================== #
    'use_fp16':
        _ParamSpec('boolean', False,
                   'Use 16-bit floats for certain tensors instead of 32-bit '
                   'floats. This is currently experimental.'),
    # TODO(reedwm): The default loss scale of 128 causes most models to diverge
    # on the second step with synthetic data. Changing the tf.set_random_seed
    # call to tf.set_random_seed(1235) or most other seed values causes the
    # issue not to occur.
    'fp16_loss_scale':
        _ParamSpec('float', None,
                   'If fp16 is enabled, the loss is multiplied by this amount '
                   'right before gradients are computed, then each gradient '
                   'is divided by this amount. Mathematically, this has no '
                   'effect, but it helps avoid fp16 underflow. Set to 1 to '
                   'effectively disable.'),
    'fp16_vars':
        _ParamSpec('boolean', False,
                   'If fp16 is enabled, also use fp16 for variables. If False, '
                   'the variables are stored in fp32 and casted to fp16 when '
                   'retrieved.  Recommended to leave as False.'),

    # ======================================================================== #
    # Variables management for clusters and multi-GPUs
    # ======================================================================== #
    # The method for managing variables:
    #   parameter_server: variables are stored on a parameter server that holds
    #       the master copy of the variable. In local execution, a local device
    #       acts as the parameter server for each variable; in distributed
    #       execution, the parameter servers are separate processes in the
    #       cluster.
    #       For each step, each tower gets a copy of the variables from the
    #       parameter server, and sends its gradients to the param server.
    #   replicated: each GPU has its own copy of the variables. To apply
    #       gradients, an all_reduce algorithm or or regular cross-device
    #       aggregation is used to replicate the combined gradients to all
    #       towers (depending on all_reduce_spec parameter setting).
    #   independent: each GPU has its own copy of the variables, and gradients
    #       are not shared between towers. This can be used to check performance
    #       when no data is moved between GPUs.
    #   distributed_replicated: Distributed training only. Each GPU has a copy
    #       of the variables, and updates its copy after the parameter servers
    #       are all updated with the gradients from all servers. Only works with
    #       cross_replica_sync=true. Unlike 'replicated', currently never uses
    #       nccl all-reduce for replicating within a server.
    #   distributed_all_reduce: Distributed training where all replicas run
    #       in a single session, using all-reduce to mutally reduce the
    #       gradients.  Uses no parameter servers.  When there is only one
    #       worker, this is the same as replicated.
    'variable_update':
        _ParamSpec('string', 'parameter_server',
                   'The method for managing variables: parameter_server, '
                   'replicated, distributed_replicated, independent, '
                   'distributed_all_reduce'),
    'all_reduce_spec':
        _ParamSpec('string', None,
                   'A specification of the all_reduce algorithm to be used for '
                   'reducing gradients.  For more details, see '
                   'parse_all_reduce_spec in variable_mgr.py.  An '
                   'all_reduce_spec has BNF form:\n'
                   'int ::= positive whole number\n'
                   'g_int ::= int[KkMGT]?\n'
                   'alg_spec ::= alg | alg#int\n'
                   'range_spec ::= alg_spec | alg_spec/alg_spec\n'
                   'spec ::= range_spec | range_spec:g_int:range_spec\n'
                   'NOTE: not all syntactically correct constructs are '
                   'supported.\n\n'
                   'Examples:\n '
                   '"xring" == use one global ring reduction for all '
                   'tensors\n'
                   '"pscpu" == use CPU at worker 0 to reduce all tensors\n'
                   '"nccl" == use NCCL to locally reduce all tensors.  '
                   'Limited to 1 worker.\n'
                   '"nccl/xring" == locally (to one worker) reduce values '
                   'using NCCL then ring reduce across workers.\n'
                   '"pscpu:32k:xring" == use pscpu algorithm for tensors of '
                   'size up to 32kB, then xring for larger tensors.'),
    # ======================================================================== #
    # Distributed training parameters.
    # ======================================================================== #
    'job_name':
        _ParamSpec('string', '',
                   'One of "ps", "worker", "".  Empty for local training'),
    'ps_hosts':
        _ParamSpec('string', '', 'Comma-separated list of target hosts'),
    'worker_hosts':
        _ParamSpec('string', '', 'Comma-separated list of target hosts'),
    'controller_host':
        _ParamSpec('string', None, 'optional controller host'),
    'task_index':
        _ParamSpec('integer', 0, 'Index of task within the job'),
    'server_protocol':
        _ParamSpec('string', 'grpc', 'protocol for servers'),
    'cross_replica_sync':
        _ParamSpec('boolean', True, ''),
    # ======================================================================== #
    # Summary and Save & load checkpoints.
    # ======================================================================== #
    'summary_verbosity':
        _ParamSpec(
            'integer', 0, 'Verbosity level for summary ops. '
            '  level 0: disable any summary. '
            '  level 1: small and fast ops, e.g.: learning_rate, total_loss.'
            '  level 2: medium-cost ops, e.g. histogram of all gradients.'
            '  level 3: expensive ops: images and histogram of each gradient.'),
    'save_summaries_steps':
        _ParamSpec('integer', 0,
                   'How often to save summaries for trained models. Pass 0 to '
                   'disable summaries.'),
    'save_model_secs':
        _ParamSpec('integer', 0,
                   'How often to save trained models. Pass 0 to disable '
                   'checkpoints.'),
    'train_dir':
        _ParamSpec('string', None,
                   'Path to session checkpoints. Pass None to disable saving '
                   'checkpoint at the end.'),
    'eval_dir':
        _ParamSpec('string', '/tmp/tf_cnn_benchmarks/eval',
                   'Directory where to write eval event logs.'),
    'ckpt_scope':
        _ParamSpec('string', None,
                   'Change the checkpoint main scope. In the form old:new'),
    'result_storage':
        _ParamSpec('string', None,
                   'Specifies storage option for benchmark results. None means '
                   'results won\'t be stored. `cbuild_benchmark_datastore` '
                   'means results will be stored in cbuild datastore (note: '
                   'this option requires special permissions and meant to be '
                   'used from cbuilds).'),
}

# =========================================================================== #
# Main definition + tools
# =========================================================================== #
"""Parameters class.
"""
Params = namedtuple('Params', DEFAULT_PARAMS.keys())  # pylint: disable=invalid-name


def make_params(**kwargs):
    """Create a Params tuple for from kwargs.

    Default values are filled in from DEFAULT_PARAMS.

    Args:
        **kwargs: kwarg values will override the default values.
    Returns:
        Params namedtuple for constructing.
    """
    # Create a (name: default_value) map from PARAMS.
    default_kwargs = {
        name: DEFAULT_PARAMS[name].default_value
        for name in DEFAULT_PARAMS
    }
    return Params(**default_kwargs)._replace(**kwargs)

def make_params_from_flags():
    """Create a Params tuple for training CNN from tf.flags.FLAGS.

    Returns:
        Params namedtuple for constructing BenchmarkCNN.
    """
    # Collect (name: value) pairs for FLAGS with matching names in
    # _DEFAULT_PARAMS.
    flag_values = {name: getattr(FLAGS, name) for name in DEFAULT_PARAMS.keys()}
    return Params(**flag_values)

def define_flags():
    """Define a command line FLAG for each ParamSpec in DEFAULT_PARAMS.
    """
    define_flag = {
        'boolean': tf.flags.DEFINE_boolean,
        'float': tf.flags.DEFINE_float,
        'integer': tf.flags.DEFINE_integer,
        'string': tf.flags.DEFINE_string,
    }
    for name, param_spec in six.iteritems(DEFAULT_PARAMS):
        if param_spec.flag_type not in define_flag:
            raise ValueError('Unknown flag_type %s' % param_spec.flag_type)
        else:
            # Define if not already existing.
            if not name in FLAGS:
                define_flag[param_spec.flag_type](
                    name, param_spec.default_value, param_spec.description)

def save_params(params, path):
    """Save parameters into a directory. Useful for evaluation.
    """
    # Create the directory if necessary...
    if path and not os.path.isdir(path):
        os.makedirs(path)

    if path and os.path.isdir(path):
        filename = os.path.join(path, 'parameters.json')
        print('Saving parameters into: %s.' % filename)
        with open(filename, 'w') as fp:
            json.dump(params._asdict(), fp, indent=4)
    else:
        print('No directory specified to save parameters.')

def load_params(path):
    """Load parameters from a directory. Useful for evaluation.
    """
    filename = os.path.join(path if path else '', 'parameters.json')
    if path and os.path.isfile(filename):
        print('Loading parameters into: %s.' % filename)
        with open(filename) as fp:
            data = json.load(fp)
            return make_params(**data)
    else:
        print('No directory specified to load parameters.')
        return None
