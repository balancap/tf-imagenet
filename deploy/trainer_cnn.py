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
"""Trainer handling CNNs.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import namedtuple
import math
import os
import threading
import time

import numpy as np

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest

import datasets

from . import benchmark_storage
from . import cnn_util
from . import convnet_builder
from . import variable_mgr

from .trainer_utils import load_checkpoint, restore_checkpoint_fn
from .benchmark_cnn import (
    get_data_type, loss_function, create_config_proto, GlobalStepWatcher,
    get_mode_from_params, benchmark_one_step, get_perf_timing_str,
    store_benchmarks)

from deploy.cnn_util import log_fn
from .models_bench import model_config


class TrainerCNN(object):
    """Class for training a cnn network."""

    def __init__(self, dataset, model, params):
        """Initialize TrainerCNN.

        Args:
            params: Params tuple, typically created by make_params or
                make_params_from_flags.
        Raises:
            ValueError: Unsupported params settings.
        """
        self.params = params
        self.dataset = dataset
        self.model = model
        # self.dataset = datasets.create_dataset(self.params.data_dir,
        #                                        self.params.data_name)
        # self.model = model_config.get_model_config(self.params.model, self.dataset)

        self.trace_filename = self.params.trace_file
        self.data_format = self.params.data_format
        self.num_batches = self.params.num_batches
        autotune_threshold = self.params.autotune_threshold if (
            self.params.autotune_threshold) else 1
        min_autotune_warmup = 5 * autotune_threshold * autotune_threshold
        self.num_warmup_batches = self.params.num_warmup_batches if (
            self.params.num_warmup_batches is not None) else max(
                10, min_autotune_warmup)
        self.graph_file = self.params.graph_file
        self.resize_method = self.params.resize_method
        self.sync_queue_counter = 0
        self.num_gpus = self.params.num_gpus
        if self.params.gpu_indices:
            self.gpu_indices = [int(x) for x in self.params.gpu_indices.split(',')]
        else:
            self.gpu_indices = [x for x in range(self.num_gpus)]
        self.use_synthetic_gpu_images = self.dataset.use_synthetic_gpu_images()

        if (self.params.device == 'cpu' and self.params.data_format == 'NCHW' and
                not self.params.mkl):
            raise ValueError('device=cpu requires that data_format=NHWC')

        if self.params.use_tf_layers and self.params.use_fp16:
            raise ValueError('if use_fp16=true, use_tf_layers must be false.')

        if ((self.params.num_epochs_per_decay or
                self.params.learning_rate_decay_factor) and
                not (self.params.learning_rate and self.params.num_epochs_per_decay and
                self.params.learning_rate_decay_factor)):
            raise ValueError('If one of num_epochs_per_decay or '
                             'learning_rate_decay_factor is set, both must be set'
                             'and learning_rate must be set')
        if (self.params.minimum_learning_rate and
                not (self.params.learning_rate and self.params.num_epochs_per_decay and
                self.params.learning_rate_decay_factor)):
            raise ValueError('minimum_learning_rate requires learning_rate,'
                             'num_epochs_per_decay, and '
                             'learning_rate_decay_factor to be set')

        if (self.params.use_fp16 and self.params.fp16_vars and
                'replicated' in self.params.variable_update and
                'nccl' in self.params.all_reduce_spec):
            raise ValueError('fp16 variables are not supported with NCCL')

        # Use the batch size from the command line if specified, otherwise use the
        # model's default batch size.  Scale the benchmark's batch size by the
        # number of GPUs.
        if self.params.batch_size > 0:
            self.model.set_batch_size(self.params.batch_size)
        self.batch_size = self.model.get_batch_size() * self.num_gpus
        self.batch_group_size = self.params.batch_group_size

        if self.params.use_fp16:
            self.loss_scale = (self.params.fp16_loss_scale or
                               self.model.get_fp16_loss_scale())
        else:
            self.loss_scale = 1.

        self.job_name = self.params.job_name  # "" for local training
        self.ps_hosts = self.params.ps_hosts.split(',')
        self.worker_hosts = self.params.worker_hosts.split(',')
        self.controller_host = self.params.controller_host

        if len(self.worker_hosts) > 1 and self.params.all_reduce_spec == 'nccl':
            raise ValueError('--all_reduce_spec=nccl is invalid in a '
                             'multi-worker job')

        # PS server is used for distributed jobs not using all-reduce.
        use_ps_server = self.job_name and (
            self.params.variable_update != 'distributed_all_reduce')
        # controller is used for distributed_all_reduce with > 1 worker.
        use_controller = (self.params.variable_update == 'distributed_all_reduce'
                          and self.job_name)
        if use_controller and not self.controller_host:
            raise ValueError('When variable_update==distributed_all_reduce '
                             'controller_host must also be specified.')

        self.local_parameter_device_flag = self.params.local_parameter_device
        if self.job_name:
            self.task_index = self.params.task_index
            if use_controller:
                assert not use_ps_server
                self.cluster = tf.train.ClusterSpec(
                    {'controller': [self.controller_host],
                     'worker': self.worker_hosts})
            else:
                assert use_ps_server
                self.cluster = tf.train.ClusterSpec({'ps': self.ps_hosts,
                                                     'worker': self.worker_hosts})

            self.server = None
            if self.job_name != 'controller':
                if not self.server:
                    self.server = tf.train.Server(self.cluster, job_name=self.job_name,
                                                  task_index=self.task_index,
                                                  config=create_config_proto(self.params),
                                                  protocol=self.params.server_protocol)

            worker_prefix = '/job:worker/task:%s' % self.task_index
            if use_ps_server:
                self.param_server_device = tf.train.replica_device_setter(
                    worker_device=worker_prefix + '/cpu:0', cluster=self.cluster)
                # This device on which the queues for managing synchronization between
                # servers should be stored.
                num_ps = len(self.ps_hosts)
                self.sync_queue_devices = ['/job:ps/task:%s/cpu:0' % i
                                           for i in range(num_ps)]
            else:
                self.sync_queue_devices = ['/job:worker/task:0/cpu:0']
        else:
            self.task_index = 0
            self.cluster = None
            self.server = None
            worker_prefix = ''
            self.param_server_device = '/%s:0' % self.params.local_parameter_device
            self.sync_queue_devices = [self.param_server_device]

        # Device to use for ops that need to always run on the local worker's CPU.
        self.cpu_device = '%s/cpu:0' % worker_prefix

        # Device to use for ops that need to always run on the local worker's
        # compute device, and never on a parameter server device.
        self.raw_devices = [
            '%s/%s:%i' % (worker_prefix, self.params.device, i)
            for i in xrange(self.num_gpus)
        ]

        if (self.params.staged_vars and
                self.params.variable_update != 'parameter_server'):
            raise ValueError('staged_vars for now is only supported with '
                             'variable_update=parameter_server')

        if self.params.variable_update == 'parameter_server':
            if self.job_name:
                if not self.params.staged_vars:
                    self.variable_mgr = variable_mgr.VariableMgrDistributedFetchFromPS(
                        self)
                else:
                    self.variable_mgr = (
                        variable_mgr.VariableMgrDistributedFetchFromStagedPS(self))
            else:
                if not self.params.staged_vars:
                    self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
                else:
                    self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromStagedPS(
                        self)
        elif self.params.variable_update == 'replicated':
            if self.job_name:
                raise ValueError('Invalid variable_update in distributed mode: %s' %
                                 self.params.variable_update)
            self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
                self, self.params.all_reduce_spec)
        elif self.params.variable_update == 'distributed_all_reduce':
            assert self.params.cross_replica_sync
            self.variable_mgr = variable_mgr.VariableMgrDistributedAllReduce(
                self, self.params.all_reduce_spec,
                'worker' if len(self.worker_hosts) > 1 else 'localhost',
                len(self.worker_hosts))
        elif self.params.variable_update == 'distributed_replicated':
            assert self.params.cross_replica_sync
            if not self.job_name:
                raise ValueError('Invalid variable_update in local mode: %s' %
                                 self.params.variable_update)
            self.variable_mgr = variable_mgr.VariableMgrDistributedReplicated(self)
        elif self.params.variable_update == 'independent':
            if self.job_name:
                raise ValueError('Invalid variable_update in distributed mode: %s' %
                                 self.params.variable_update)
            self.variable_mgr = variable_mgr.VariableMgrIndependent(self)
        else:
            raise ValueError(
                'Invalid variable_update: %s' % self.params.variable_update)

        # Device to use for running on the local worker's compute device, but
        # with variables assigned to parameter server devices.
        self.devices = self.variable_mgr.get_devices()
        if self.job_name:
            if use_ps_server:
                self.global_step_device = self.param_server_device
            else:
                self.global_step_device = '/job:worker/task:0/cpu:0'
        else:
            self.global_step_device = self.cpu_device

        self.image_preprocessor = self.get_image_preprocessor()
        self.init_global_step = 0

    def reset_devices_for_task(self, task_num, is_local=False):
        """Used to imitate another task when building a distributed graph."""
        worker_prefix = ('job:localhost' if is_local
                                         else '/job:worker/task:%s' % task_num)
        self.cpu_device = '%s/cpu:0' % worker_prefix
        self.raw_devices = ['%s/%s:%i' % (worker_prefix, self.params.device, i)
                            for i in xrange(self.num_gpus)]
        self.devices = self.variable_mgr.get_devices()

    def raw_devices_across_tasks(self, is_local=False):
        """Returns list of raw device names across all tasks."""
        num_tasks = len(self.worker_hosts)
        if is_local:
            assert num_tasks == 1
            return self.raw_devices
        else:
            return ['job:worker/task%s/%s:%i' % (t, self.params.device, i)
                    for t in xrange(num_tasks)
                    for i in xrange(self.num_gpus)]

    def print_info(self):
        """Print basic information."""
        log_fn('Model:       %s' % self.model.get_model())
        log_fn('Mode:        %s' % get_mode_from_params(self.params))
        single_session = self.params.variable_update == 'distributed_all_reduce'
        log_fn('SingleSess:  %s' % single_session)
        if single_session:
            device_list = self.raw_devices_across_tasks()
            num_workers = len(self.worker_hosts)
        else:
            device_list = self.raw_devices
            num_workers = 1
        batch_size = num_workers * self.batch_size
        log_fn('Batch size:  %s global' % batch_size)
        log_fn('             %s per device' % (batch_size / len(device_list)))
        if self.batch_group_size > 1:
            log_fn('             %d batches per prepocessing group' %
                self.batch_group_size)
        log_fn('Devices:     %s' % device_list)
        log_fn('Data format: %s' % self.data_format)
        log_fn('Optimizer:   %s' % self.params.optimizer)
        log_fn('Variables:   %s' % self.params.variable_update)
        if (self.params.variable_update == 'replicated' or
                self.params.variable_update == 'distributed_all_reduce'):
            log_fn('AllReduce:   %s' % self.params.all_reduce_spec)
        if self.job_name:
            log_fn('Sync:        %s' % self.params.cross_replica_sync)
        if self.params.staged_vars:
            log_fn('Staged vars: %s' % self.params.staged_vars)
        log_fn('==========')

    def run(self):
        """Run the training task assigned to this process.

        Returns:
            Dictionary of statistics for training or eval.
        Raises:
             ValueError: unrecognized job name.
        """
        if self.params.job_name == 'ps':
            log_fn('Running parameter server %s' % self.task_index)
            self.server.join()
            return {}

        # For distributed_all_reduce with multiple workers, drive
        # from a separate controller process.
        if self.params.variable_update == 'distributed_all_reduce':
            if self.params.job_name == 'worker':
                log_fn('Starting worker %s' % self.task_index)
                self.server.join()
                return
            elif self.params.job_name and self.params.job_name != 'controller':
                raise ValueError('unrecognized job name: %s' % self.params.job_name)

        with tf.Graph().as_default():
            if self.params.eval:
                return self._eval_cnn()
            else:
                return self._benchmark_cnn()

    def _eval_cnn(self):
        """Evaluate a model every self.params.eval_interval_secs.

        Returns:
            Dictionary containing eval statistics. Currently returns an empty
            dictionary.
        """
        (image_producer_ops, enqueue_ops, fetches) = self._build_model()
        saver = tf.train.Saver(self.variable_mgr.savable_variables())
        summary_writer = tf.summary.FileWriter(self.params.eval_dir,
                                               tf.get_default_graph())
        target = ''
        local_var_init_op = tf.local_variables_initializer()
        variable_mgr_init_ops = [local_var_init_op]
        with tf.control_dependencies([local_var_init_op]):
            variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
        local_var_init_op_group = tf.group(*variable_mgr_init_ops)
        summary_op = tf.summary.merge_all()
        while True:
            self._eval_once(
                saver, summary_writer, target, local_var_init_op_group,
                image_producer_ops, enqueue_ops, fetches, summary_op)
            if self.params.eval_interval_secs <= 0:
                break
            log_fn('Wait for next evaluation checkpoint...')
            time.sleep(self.params.eval_interval_secs)
        return {}

    def _eval_once(self, saver, summary_writer, target,
                   local_var_init_op_group, image_producer_ops, enqueue_ops,
                   fetches, summary_op):
        """Evaluate the model from a checkpoint using validation dataset."""
        with tf.Session(
                target=target, config=create_config_proto(self.params)) as sess:
            if self.params.train_dir is None:
                raise ValueError('Trained model directory not specified')
            try:
                global_step = load_checkpoint(
                    saver, sess, self.params.train_dir,
                    self.params.ckpt_scope, self.params.moving_average_decay)
            except CheckpointNotFoundException:
                log_fn('Checkpoint not found in %s' % self.params.train_dir)
                return
            sess.run(local_var_init_op_group)
            if self.dataset.queue_runner_required():
                tf.train.start_queue_runners(sess=sess)
            image_producer = cnn_util.ImageProducer(sess, image_producer_ops,
                self.batch_group_size)
            image_producer.start()
            for i in xrange(len(enqueue_ops)):
                sess.run(enqueue_ops[:(i+1)])
                image_producer.notify_image_consumption()
            start_time = time.time()

            # Losses and metrics.
            losses_sum = {}
            metrics_sum = {}
            total_eval_count = self.num_batches * self.batch_size
            for step in xrange(self.num_batches):
                if (self.params.save_summaries_steps > 0 and
                        (step + 1) % self.params.save_summaries_steps == 0):
                    results, summary_str = sess.run([fetches, summary_op])
                    summary_writer.add_summary(summary_str)
                else:
                    results = sess.run(fetches)
                # Record losses and metrics.
                for k, v in results.items():
                    if 'losses/' in k:
                        losses_sum[k] = losses_sum.get(k, 0.) + v
                    if 'metrics/' in k:
                        metrics_sum[k] = metrics_sum.get(k, 0.) + v

                if (step + 1) % self.params.display_every == 0:
                    duration = time.time() - start_time
                    examples_per_sec = (
                        self.batch_size * self.params.display_every / duration)
                    log_fn('%i\t%.1f examples/sec' % (step + 1, examples_per_sec))
                    start_time = time.time()
                image_producer.notify_image_consumption()
            image_producer.done()

            summary = tf.Summary()
            losses = {}
            metrics = {}
            # Losses and metrics summaries.
            log_losses = '<<< Model LOSSES >>> \n'
            for k, v in losses_sum.items():
                losses[k] = v / self.num_batches
                summary.value.add(tag=k, simple_value=v / self.num_batches)
                log_losses += '{:20}: {}\n'.format(k, losses[k])
            log_metrics = '<<< Model METRICS >>> \n'
            for k, v in metrics_sum.items():
                metrics[k] = v / self.num_batches
                summary.value.add(tag=k, simple_value=v / self.num_batches)
                log_metrics += '{:20}: {}\n'.format(k, metrics[k])
            # Save summaries and log.
            summary_writer.add_summary(summary, global_step)
            log_fn(log_losses)
            log_fn(log_metrics)

    def _benchmark_cnn(self):
        """Run cnn in benchmark mode. When forward_only on, it forwards CNN.

        Returns:
            Dictionary containing training statistics (num_workers, num_steps,
            average_wall_time, images_per_sec).
        """
        if self.params.variable_update == 'distributed_all_reduce':
            self.single_session = True
            (image_producer_ops, enqueue_ops, fetches) = (
                self._build_model_single_session())
        else:
            self.single_session = False
            (image_producer_ops, enqueue_ops, fetches) = self._build_model()
        fetches_list = nest.flatten(list(fetches.values()))
        main_fetch_group = tf.group(*fetches_list)
        execution_barrier = None
        if (not self.single_session and self.job_name and
                not self.params.cross_replica_sync):
            execution_barrier = self.add_sync_queues_and_barrier(
                'execution_barrier_', [])

        global_step = tf.train.get_global_step()
        with tf.device(self.global_step_device):
            with tf.control_dependencies([main_fetch_group]):
                fetches['inc_global_step'] = global_step.assign_add(1)

        if ((not self.single_session) and self.job_name and
                self.params.cross_replica_sync):
            # Block all replicas until all replicas are ready for next step.
            fetches['sync_queues'] = self.add_sync_queues_and_barrier(
                    'sync_queues_step_end_', [main_fetch_group])

        local_var_init_op = tf.local_variables_initializer()
        variable_mgr_init_ops = [local_var_init_op]
        with tf.control_dependencies([local_var_init_op]):
            variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
        local_var_init_op_group = tf.group(*variable_mgr_init_ops)

        summary_op = tf.summary.merge_all()
        is_chief = (not self.job_name or self.task_index == 0)
        summary_writer = None
        if (is_chief and self.params.summary_verbosity and self.params.train_dir and
                self.params.save_summaries_steps > 0):
            summary_writer = tf.summary.FileWriter(self.params.train_dir,
                                                   tf.get_default_graph())

        # We want to start the benchmark timer right after a image_producer barrier
        # and avoids undesired wating times on barriers.
        if ((self.num_warmup_batches + len(enqueue_ops) - 1) %
                self.batch_group_size) != 0:
            self.num_warmup_batches = int(
                math.ceil((self.num_warmup_batches + len(enqueue_ops) - 1.0) /
                           self.batch_group_size) * self.batch_group_size
                           - len(enqueue_ops) + 1)
            log_fn('Round up warm up steps to %d to match batch_group_size' %
                self.num_warmup_batches)
            assert ((self.num_warmup_batches + len(enqueue_ops) - 1) %
                     self.batch_group_size) == 0
        # We run the summaries in the same thread as the training operations by
        # passing in None for summary_op to avoid a summary_thread being started.
        # Running summaries and training operations in parallel could run out of
        # GPU memory.
        saver = tf.train.Saver(self.variable_mgr.savable_variables())
        ready_for_local_init_op = None
        if self.job_name and not self.single_session:
            # In distributed mode, we don't want to run local_var_init_op_group until
            # the global variables are initialized, because local_var_init_op_group
            # may use global variables (such as in distributed replicated mode). We
            # don't set this in non-distributed mode, because in non-distributed mode,
            # local_var_init_op_group may itself initialize global variables (such as
            # in replicated mode).
            ready_for_local_init_op = tf.report_uninitialized_variables(
                tf.global_variables())
        # Restoring a checkpoint to fine-tune.
        restore_init_fn = None
        if self.params.ckpt_finetune is not None:
            restore_init_fn = restore_checkpoint_fn(
                self.params.ckpt_finetune,
                0,
                self.params.ckpt_scope,
                self.params.ckpt_moving_average_decay)

        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=self.params.train_dir,
            ready_for_local_init_op=ready_for_local_init_op,
            # init_op=restore_init_op,
            init_fn=restore_init_fn,
            local_init_op=local_var_init_op_group,
            saver=saver,
            global_step=global_step,
            summary_op=None,
            save_model_secs=self.params.save_model_secs,
            summary_writer=summary_writer)

        step_train_times = []
        start_standard_services = (self.params.summary_verbosity >= 1 or
                                   self.dataset.queue_runner_required())
        if self.job_name == 'controller':
            master_target = self.worker_hosts[0]
        else:
            master_target = self.server.target if self.server else ''
        with sv.managed_session(
                master=master_target,
                config=create_config_proto(self.params),
                start_standard_services=start_standard_services) as sess:
            image_producer = cnn_util.ImageProducer(sess, image_producer_ops,
                                                    self.batch_group_size)
            image_producer.start()
            for i in xrange(len(enqueue_ops)):
                sess.run(enqueue_ops[:(i+1)])
                image_producer.notify_image_consumption()
            self.init_global_step, = sess.run([global_step])
            if not self.single_session:
                global_step_watcher = GlobalStepWatcher(
                    sess, global_step,
                    len(self.worker_hosts) * self.num_warmup_batches +
                    self.init_global_step,
                    len(self.worker_hosts) * (
                        self.num_warmup_batches + self.num_batches) - 1)
                global_step_watcher.start()
            else:
                global_step_watcher = None

            if self.graph_file is not None:
                path, filename = os.path.split(self.graph_file)
                as_text = filename.endswith('txt')
                log_fn('Writing GraphDef as %s to %s' % (
                       'text' if as_text else 'binary', self.graph_file))
                tf.train.write_graph(sess.graph_def, path, filename, as_text)

            log_fn('Running warm up')
            local_step = -1 * self.num_warmup_batches

            if self.single_session or (self.params.cross_replica_sync and
                                       self.params.job_name):
                # In cross-replica sync mode, all workers must run the same number of
                # local steps, or else the workers running the extra step will block.
                done_fn = lambda: local_step == self.num_batches
            else:
                done_fn = global_step_watcher.done
            loop_start_time = time.time()
            while not done_fn():
                if local_step == 0:
                    log_fn('Done warm up')
                    if execution_barrier:
                        log_fn('Waiting for other replicas to finish warm up')
                        assert global_step_watcher.start_time == 0
                        sess.run([execution_barrier])

                    header_str = 'Step\tImg/sec\t\t\t\t\t\tloss'
                    if self.params.print_training_accuracy or self.params.forward_only:
                        header_str += '\ttop_1_accuracy\ttop_5_accuracy'
                    log_fn(header_str)
                    assert len(step_train_times) == self.num_warmup_batches
                    # reset times to ignore warm up batch
                    step_train_times = []
                    loop_start_time = time.time()
                if (summary_writer and
                        (local_step + 1) % self.params.save_summaries_steps == 0):
                    fetch_summary = summary_op
                else:
                    fetch_summary = None
                summary_str = benchmark_one_step(
                    sess, fetches, local_step,
                    self.batch_size * (len(self.worker_hosts) if self.single_session else 1),
                    step_train_times, self.trace_filename, image_producer,
                    self.params, fetch_summary)
                if summary_str is not None and is_chief:
                    sv.summary_computed(sess, summary_str)
                local_step += 1
            loop_end_time = time.time()
            # Waits for the global step to be done, regardless of done_fn.
            if global_step_watcher:
                while not global_step_watcher.done():
                    time.sleep(.25)
            if self.single_session:
                num_workers = len(self.worker_hosts)
                num_steps = local_step
                elapsed_time = loop_end_time - loop_start_time
            else:
                num_workers = 1
                num_steps = global_step_watcher.num_steps()
                elapsed_time = global_step_watcher.elapsed_time()

            average_wall_time = elapsed_time / num_steps if num_steps > 0 else 0
            images_per_sec = ((num_workers * self.batch_size) /
                               average_wall_time if average_wall_time > 0 else 0)

            log_fn('-' * 64)
            log_fn('total images/sec: %.2f' % images_per_sec)
            log_fn('-' * 64)
            image_producer.done()
            if is_chief:
                store_benchmarks({'total_images_per_sec': images_per_sec}, self.params)
            # Save the model checkpoint.
            if self.params.train_dir is not None and is_chief:
                checkpoint_path = os.path.join(self.params.train_dir, 'model.ckpt')
                if not gfile.Exists(self.params.train_dir):
                    gfile.MakeDirs(self.params.train_dir)
                sv.saver.save(sess, checkpoint_path, global_step)

            if execution_barrier:
                # Wait for other workers to reach the end, so this worker doesn't
                # go away underneath them.
                sess.run([execution_barrier])
        sv.stop()
        return {
            'num_workers': num_workers,
            'num_steps': num_steps,
            'average_wall_time': average_wall_time,
            'images_per_sec': images_per_sec
        }

    def _build_image_processing(self, shift_ratio=0):
        """"Build the image (pre)processing portion of the model graph."""
        with tf.device(self.cpu_device):
            if self.params.eval:
                subset = 'validation'
            else:
                subset = 'train'
            image_producer_ops = []
            image_producer_stages = []
            images_splits, labels_splits = self.image_preprocessor.minibatch(
                self.dataset, subset=subset, use_datasets=self.params.use_datasets,
                cache_data=self.params.cache_data, shift_ratio=shift_ratio)
            images_shape = images_splits[0].get_shape()
            labels_shape = labels_splits[0].get_shape()
            for device_num in range(len(self.devices)):
                image_producer_stages.append(data_flow_ops.StagingArea(
                    [images_splits[0].dtype, labels_splits[0].dtype],
                    shapes=[images_shape, labels_shape]))
                for group_index in xrange(self.batch_group_size):
                    if not self.use_synthetic_gpu_images:
                        batch_index = group_index + device_num * self.batch_group_size
                        put_op = image_producer_stages[device_num].put(
                            [images_splits[batch_index], labels_splits[batch_index]])
                        image_producer_ops.append(put_op)
        return (image_producer_ops, image_producer_stages)

    def _build_model(self):
        """Build the TensorFlow graph."""
        tf.set_random_seed(self.params.tf_random_seed)
        np.random.seed(4321)
        phase_train = not (self.params.eval or self.params.forward_only)

        log_fn('Generating model')
        losses = []
        device_grads = []
        all_logits = []
        all_top_1_ops = []
        all_top_5_ops = []
        enqueue_ops = []
        gpu_compute_stage_ops = []
        gpu_grad_stage_ops = []
        # Dictionary of losses and metrics
        d_losses = {}
        d_metrics = {}

        with tf.device(self.global_step_device):
            global_step = tf.train.get_or_create_global_step()

        # Build the processing and model for the worker.
        (image_producer_ops, image_producer_stages) = self._build_image_processing(
            shift_ratio=0)
        image_producer_ops = tf.group(*image_producer_ops)
        update_ops = None
        staging_delta_ops = []

        for device_num in range(len(self.devices)):
            with self.variable_mgr.create_outer_variable_scope(
                    device_num), tf.name_scope('tower_%i' % device_num) as name_scope:
                results = self.add_forward_pass_and_gradients(
                    phase_train, device_num, device_num,
                    image_producer_stages[device_num],
                    gpu_compute_stage_ops, gpu_grad_stage_ops)
                if phase_train:
                    losses.append(results['loss'])
                    device_grads.append(results['gradvars'])
                else:
                    all_logits.append(results['logits'])
                if not phase_train or self.params.print_training_accuracy:
                    all_top_1_ops.append(results['top_1_op'])
                    all_top_5_ops.append(results['top_5_op'])

                # List of losses and metrics.
                for k, v in results.items():
                    if 'losses/' in k:
                        l = d_losses.get(k, [])
                        l.append(v)
                        d_losses[k] = l
                    if 'metrics/' in k:
                        l = d_metrics.get(k, [])
                        l.append(v)
                        d_metrics[k] = l

                if device_num == 0:
                    # Retain the Batch Normalization updates operations only from the
                    # first tower. These operations update the moving mean and moving
                    # variance variables, which are updated (but not used) during
                    # training, and used during evaluation. The moving mean and variance
                    # approximate the true mean and variance across all images in the
                    # dataset. Therefore, in replicated mode, these moving averages would
                    # be almost identical for each tower, and so we only update and save
                    # the moving averages for one tower. In parameter server mode, all
                    # towers share a copy of the variables so we also only need to update
                    # and save the moving averages once.
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                    staging_delta_ops = list(self.variable_mgr.staging_delta_ops)

        # Moving average of trainable variables.
        #   Like for BN, only need to do it on the first device.
        #   Optimize? Run only on the first GPU?
        # with tf.device(self.global_step_device):
        device_num = 0
        # Compute moving avg vars on Device 0? Faster update this way?
        with tf.device(self.devices[device_num]):
            if self.params.moving_average_decay and phase_train:
                # Get the model variables in the device scope.
                with self.variable_mgr.create_outer_variable_scope(device_num) as sc:
                    model_vars = tf.contrib.framework.get_model_variables(scope=sc)
                    # print(model_vars)
                # OLD way. Do not capture BN mean and variance.
                # model_vars = self.variable_mgr.trainable_variables_on_device(
                #     device_num, device_num)
                variable_averages = tf.train.ExponentialMovingAverage(
                    self.params.moving_average_decay, global_step)
                update_ops = [update_ops, variable_averages.apply(model_vars)]

        if self.variable_mgr.supports_staged_vars():
            for staging_ops in self.variable_mgr.staging_vars_on_devices:
                gpu_compute_stage_ops.extend(
                    [put_op for _, (put_op, _) in six.iteritems(staging_ops)])
        enqueue_ops.append(tf.group(*gpu_compute_stage_ops))
        if gpu_grad_stage_ops:
            staging_delta_ops += gpu_grad_stage_ops
        if staging_delta_ops:
            enqueue_ops.append(tf.group(*(staging_delta_ops)))

        fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
                                      enqueue_ops, update_ops, all_top_1_ops,
                                      all_top_5_ops, phase_train,
                                      d_losses, d_metrics)
        return (image_producer_ops, enqueue_ops, fetches)

    def _build_fetches(self, global_step, all_logits, losses, device_grads,
                       enqueue_ops, update_ops, all_top_1_ops, all_top_5_ops,
                       phase_train, d_losses, d_metrics):
        """Complete construction of model graph, populating the fetches map."""
        fetches = {'enqueue_ops': enqueue_ops}
        if all_top_1_ops:
            fetches['top_1_accuracy'] = tf.reduce_sum(all_top_1_ops) / self.batch_size
        if all_top_5_ops:
            fetches['top_5_accuracy'] = tf.reduce_sum(all_top_5_ops) / self.batch_size

        # Losses and metrics summaries.
        for k, l in d_losses.items():
            fetches[k] = tf.reduce_mean(l)
            if self.task_index == 0 and self.params.summary_verbosity >= 1:
                tf.summary.scalar(k, fetches[k])
        # Metrics summaries
        for k, l in d_metrics.items():
            fetches[k] = tf.reduce_sum(l) / self.batch_size
            if self.task_index == 0 and self.params.summary_verbosity >= 1:
                tf.summary.scalar(k, fetches[k])

        if not phase_train:
            if self.params.forward_only:
                fetches['all_logits'] = tf.concat(all_logits, 0)
            return fetches
        apply_gradient_devices, gradient_state = (
            self.variable_mgr.preprocess_device_grads(device_grads))

        training_ops = []
        for d, device in enumerate(apply_gradient_devices):
            with tf.device(device):
                total_loss = tf.reduce_sum(losses)
                avg_grads = self.variable_mgr.get_gradients_to_apply(d, gradient_state)

                gradient_clip = self.params.gradient_clip
                learning_rate = (self.params.learning_rate or
                                 self.model.get_learning_rate(global_step,
                                                              self.batch_size))
                if ((not self.use_synthetic_gpu_images) and
                        self.params.learning_rate and
                        self.params.num_epochs_per_decay > 0 and
                        self.params.learning_rate_decay_factor > 0):
                    num_batches_per_epoch = (
                        float(self.dataset.num_examples_per_epoch()) / self.batch_size)
                    decay_steps = int(
                        num_batches_per_epoch * self.params.num_epochs_per_decay)

                    # Decay the learning rate exponentially based on the number of steps.
                    learning_rate = tf.train.exponential_decay(
                        self.params.learning_rate,
                        global_step,
                        decay_steps,
                        self.params.learning_rate_decay_factor,
                        staircase=True)

                    if self.params.minimum_learning_rate != 0.:
                        learning_rate = tf.maximum(learning_rate,
                                                   self.params.minimum_learning_rate)

                if gradient_clip is not None:
                    clipped_grads = [
                        (tf.clip_by_value(grad, -gradient_clip, +gradient_clip), var)
                        for grad, var in avg_grads
                    ]
                else:
                    clipped_grads = avg_grads

                learning_rate = tf.identity(learning_rate, name='learning_rate')
                if self.params.optimizer == 'momentum':
                    opt = tf.train.MomentumOptimizer(
                        learning_rate, self.params.momentum, use_nesterov=True)
                elif self.params.optimizer == 'sgd':
                    opt = tf.train.GradientDescentOptimizer(learning_rate)
                elif self.params.optimizer == 'rmsprop':
                    opt = tf.train.RMSPropOptimizer(
                        learning_rate,
                        self.params.rmsprop_decay,
                        momentum=self.params.rmsprop_momentum,
                        epsilon=self.params.rmsprop_epsilon)
                else:
                    raise ValueError('Optimizer "%s" was not recognized',
                                     self.params.optimizer)

                self.variable_mgr.append_apply_gradients_ops(
                    gradient_state, opt, clipped_grads, training_ops)
        train_op = tf.group(*(training_ops + update_ops))

        with tf.device(self.cpu_device):
            if self.task_index == 0 and self.params.summary_verbosity >= 1:
                # Global information: total loss, lr and epoch.
                tf.summary.scalar('learning_rate', learning_rate)
                tf.summary.scalar('total_loss', total_loss)
                tf.summary.scalar(
                    'epoch_step', (global_step * self.batch_size /
                                   self.dataset.num_examples_per_epoch()))

                # Histogram of log values of all non-zero gradients.
                if self.params.summary_verbosity >= 2:
                    all_grads = []
                    for grad, var in avg_grads:
                        all_grads.append(tf.reshape(grad, [-1]))
                    grads = tf.abs(tf.concat(all_grads, 0))
                    # exclude grads with zero values.
                    indices_for_non_zero_grads = tf.where(tf.not_equal(grads, 0))
                    log_grads = tf.reshape(
                        tf.log(tf.gather(grads, indices_for_non_zero_grads)), [-1])
                    tf.summary.histogram('log_gradients', log_grads)

                if self.params.summary_verbosity >= 3:
                    for grad, var in avg_grads:
                        if grad is not None:
                            tf.summary.histogram(var.op.name + '/gradients', grad)
                    for var in tf.trainable_variables():
                        tf.summary.histogram(var.op.name, var)

        fetches['train_op'] = train_op
        fetches['total_loss'] = total_loss
        return fetches

    def _build_model_single_session(self):
        """Build the TensorFlow graph for multiple replicas in a single_session.

        Returns:
            image_producer_ops:
            enqueue_ops:
            fetches:

        Raises:
             ValueError: optimizer not recognized.

        Single session runs multiple model replicas as part of one large
        distributed graph, whose global execution is always step-synchronized.
        """
        # verify assumptions
        assert self.params.task_index == 0
        assert not self.params.eval
        assert not self.params.forward_only
        assert not self.params.staged_vars

        tf.set_random_seed(self.params.tf_random_seed)
        np.random.seed(4321)
        phase_train = True

        log_fn('Generating model')
        losses = []
        device_grads = []
        all_logits = []
        all_top_1_ops = []
        all_top_5_ops = []
        enqueue_ops = []
        gpu_compute_stage_ops = []
        gpu_grad_stage_ops = []
        # Dictionary of losses and metrics
        d_losses = {}
        d_metrics = {}

        with tf.device(self.global_step_device):
            global_step = tf.train.get_or_create_global_step()

        update_ops = []
        global_image_producer_ops = []

        num_workers = len(self.worker_hosts)
        is_local = not self.job_name
        if is_local:
            assert num_workers == 1
        for task_num in range(num_workers):
            # Reset the devices that self.variable_mgr knows about to those
            # belonging to the next worker (task).
            self.reset_devices_for_task(task_num, is_local)
            # Build the per-worker image processing
            (image_producer_ops, image_producer_stages) = (
                self._build_image_processing(
                    shift_ratio=(float(task_num) / num_workers)))
            global_image_producer_ops.extend(image_producer_ops)
            # Build the per-worker model replica.
            for rel_device_num in range(len(self.devices)):
                abs_device_num = task_num * len(self.devices) + rel_device_num
                with self.variable_mgr.create_outer_variable_scope(
                        abs_device_num), tf.name_scope(
                            'task_%i_tower_%i' % (task_num, rel_device_num)) as name_scope:
                    task_results = self.add_forward_pass_and_gradients(
                        phase_train, rel_device_num, abs_device_num,
                        image_producer_stages[rel_device_num],
                        gpu_compute_stage_ops, gpu_grad_stage_ops)
                    if phase_train:
                        losses.append(task_results['loss'])
                        device_grads.append(task_results['gradvars'])
                    else:
                        all_logits.append(task_results['logits'])
                    if not phase_train or self.params.print_training_accuracy:
                        all_top_1_ops.append(task_results['top_1_op'])
                        all_top_5_ops.append(task_results['top_5_op'])

                    # List of losses and metrics, for logging purposes.
                    for k, v in task_results.items():
                        if 'losses/' in k:
                            l = d_losses.get(k, [])
                            l.append(v)
                            d_losses[k] = l
                        if 'metrics/' in k:
                            l = d_losses.get(k, [])
                            l.append(v)
                            d_losses[k] = l

                    if rel_device_num == 0:
                        # Retain the Batch Normalization updates operations only
                        # from the first tower. These operations update the moving
                        # mean and moving variance variables, which are updated
                        # (but not used) during training, and used during
                        # evaluation. The moving mean and variance approximate the
                        # true mean and variance across all images in the
                        # dataset. Therefore, in replicated mode, these moving
                        # averages would be almost identical for each tower, and
                        # so we only update and save the moving averages for one
                        # tower. In parameter server mode, all towers share a copy
                        # of the variables so we also only need to update and save
                        # the moving averages once.
                        update_ops.extend(tf.get_collection(
                            tf.GraphKeys.UPDATE_OPS, name_scope))
                        assert not self.variable_mgr.staging_delta_ops

        enqueue_ops.append(tf.group(*gpu_compute_stage_ops))
        assert not self.variable_mgr.supports_staged_vars()
        assert not gpu_grad_stage_ops

        fetches = self._build_fetches(global_step, all_logits, losses,
                                      device_grads, enqueue_ops, update_ops,
                                      all_top_1_ops, all_top_5_ops, phase_train,
                                      d_losses, d_metrics)
        global_image_producer_ops = tf.group(*global_image_producer_ops)
        return (global_image_producer_ops, enqueue_ops, fetches)

    def add_forward_pass_and_gradients(
            self, phase_train, rel_device_num, abs_device_num, image_producer_stage,
            gpu_compute_stage_ops, gpu_grad_stage_ops):
        """Add ops for forward-pass and gradient computations."""
        nclass = self.dataset.num_classes + 1
        input_data_type = get_data_type(self.params)
        data_type = get_data_type(self.params)
        if not self.use_synthetic_gpu_images:
            with tf.device(self.cpu_device):
                host_images, host_labels = image_producer_stage.get()
                images_shape = host_images.get_shape()
                labels_shape = host_labels.get_shape()
        with tf.device(self.raw_devices[rel_device_num]):
            if not self.use_synthetic_gpu_images:
                gpu_compute_stage = data_flow_ops.StagingArea(
                    [host_images.dtype, host_labels.dtype],
                    shapes=[images_shape, labels_shape]
                )
                # The CPU-to-GPU copy is triggered here.
                gpu_compute_stage_op = gpu_compute_stage.put(
                    [host_images, host_labels])
                images, labels = gpu_compute_stage.get()
                images = tf.reshape(images, shape=images_shape)
                gpu_compute_stage_ops.append(gpu_compute_stage_op)
            else:
                # Minor hack to avoid H2D copy when using synthetic data
                image_size = self.model.get_image_size()
                image_shape = [self.batch_size // self.num_gpus, image_size, image_size,
                               self.dataset.depth]
                labels_shape = [self.batch_size // self.num_gpus]
                # Synthetic image should be within [0, 255].
                images = tf.truncated_normal(
                    image_shape,
                    dtype=input_data_type,
                    mean=127,
                    stddev=60,
                    name='synthetic_images')
                images = tf.contrib.framework.local_variable(
                    images, name='gpu_cached_images')
                labels = tf.random_uniform(
                    labels_shape, minval=0, maxval=nclass-1,
                    dtype=tf.int32, name='synthetic_labels')

        with tf.device(self.devices[rel_device_num]):
            # Model pre-scaling inputs.
            images = self.model.prescaling(images)

            # Convert images to CHW format and cast.
            with tf.name_scope('inputs_format'):
                if self.data_format == 'NCHW':
                    images = tf.transpose(images, [0, 3, 1, 2])
                if input_data_type != data_type:
                    images = tf.cast(images, data_type)
                var_type = tf.float32
                if data_type == tf.float16 and self.params.fp16_vars:
                    var_type = tf.float16

            # TODO: support of tf.float16 in tf.slim?
            # network = convnet_builder.ConvNetBuilder(
            #     images, self.dataset.depth, phase_train, self.params.use_tf_layers,
            #     self.data_format, data_type, var_type)

            with tf.variable_scope('cg', custom_getter=self.model.get_custom_getter()):
                # Inference on model...
                logits, end_points = self.model.forward(
                    images, nclass, self.data_format, phase_train)
                # Aux logits?
                aux_logits = None
                # if network.aux_top_layer is not None:
                #     with network.switch_to_aux_top_layer():
                #         aux_logits = network.affine(nclass, activation='linear', stddev=0.001)

            if data_type == tf.float16:
                # TODO(reedwm): Determine if we should do this cast here.
                logits = tf.cast(logits, tf.float32)
                if aux_logits is not None:
                    aux_logits = tf.cast(aux_logits, tf.float32)

            # The returned values
            results = {}
            # Metrics. TODO: integrate in models
            if not phase_train or self.params.print_training_accuracy:
                # Metrics
                top_1_op = tf.reduce_sum(
                    tf.cast(tf.nn.in_top_k(logits, labels, 1), data_type))
                top_5_op = tf.reduce_sum(
                    tf.cast(tf.nn.in_top_k(logits, labels, 5), data_type))
                results['metrics/accuracy_at_1'] = top_1_op
                results['metrics/recall_at_5'] = top_5_op
                # TODO: legacy
                results['top_1_op'] = top_1_op
                results['top_5_op'] = top_5_op

            # Model losses: cross entropy + others...
            loss, losses = self.model.losses(logits, labels, end_points)
            for k in losses.keys():
                results['losses/' + k] = losses[k]

            # Not training: don't need to compute gradients and regularization.
            if not phase_train:
                results['logits'] = logits
                return results

            # Regularization loss: only on first device.
            results['losses/weight_decay'] = tf.constant(0., dtype=tf.float32)
            weight_decay = self.params.weight_decay
            if rel_device_num == 0 and weight_decay:
                # Regularization losses in the present name scope.
                nm_sc = tf.contrib.framework.get_name_scope()
                reg_losses = tf.losses.get_regularization_losses(nm_sc)
                reg_loss = tf.add_n(reg_losses, name='total_regularization_loss')
                results['losses/weight_decay'] = reg_loss
                loss += reg_loss
                # TODO: fp16 convertion???

            # # L2 loss: get the trainable variables...
            # # print(tf.contrib.framework.get_name_scope())
            # # print(params)
            # # print(tf.losses.get_regularization_losses(tf.contrib.framework.get_name_scope()))
            # if data_type == tf.float16 and self.params.fp16_vars:
            #     # fp16 reductions are very slow on GPUs, so cast to fp32 before calling
            #     # tf.nn.l2_loss and tf.add_n.
            #     # TODO(b/36217816): Once the bug is fixed, investigate if we should do
            #     # this reduction in fp16.
            #     fp32_params = (tf.cast(p, tf.float32) for p in params)
            #     l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in fp32_params])
            # else:
            #     l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params])
            # # Weight decay. TODO: using layers scopes?
            # weight_decay = self.params.weight_decay
            # if weight_decay is not None and weight_decay != 0.:
            #     results['losses/weight_decay'] = weight_decay * l2_loss
            #     loss += weight_decay * l2_loss
            # else:
            #     results['losses/weight_decay'] = 0

            # Gradients of trainable vars.
            params = self.variable_mgr.trainable_variables_on_device(
                rel_device_num, abs_device_num)
            aggmeth = tf.AggregationMethod.DEFAULT
            scaled_loss = loss if self.loss_scale == 1. else loss * self.loss_scale
            grads = tf.gradients(scaled_loss, params, aggregation_method=aggmeth)
            if self.loss_scale != 1.:
                # TODO(reedwm): We could avoid these multiplications by directly
                # modifying the learning rate instead. If this is done, care must be
                # taken to ensure that this scaling method is correct, as some
                # optimizers square gradients and do other operations which might not be
                # compatible with modifying both the gradients and the learning rate.
                grads = [grad * (1. / self.loss_scale) for grad in grads]

            if self.params.staged_vars:
                grad_dtypes = [grad.dtype for grad in grads]
                grad_shapes = [grad.shape for grad in grads]
                grad_stage = data_flow_ops.StagingArea(grad_dtypes, grad_shapes)
                grad_stage_op = grad_stage.put(grads)
                # In general, this decouples the computation of the gradients and
                # the updates of the weights.
                # During the pipeline warm up, this runs enough training to produce
                # the first set of gradients.
                gpu_grad_stage_ops.append(grad_stage_op)
                grads = grad_stage.get()

            param_refs = self.variable_mgr.trainable_variables_on_device(
                rel_device_num, abs_device_num, writable=True)
            gradvars = list(zip(grads, param_refs))
            results['loss'] = loss
            results['gradvars'] = gradvars
            return results

    def get_image_preprocessor(self):
        """Returns the image preprocessor to used, based on the model.

        Returns:
            The image preprocessor, or None if synthetic data should be used.
        """
        image_size = self.model.get_image_size()
        input_data_type = get_data_type(self.params)

        shift_ratio = 0
        if self.job_name:
            # shift_ratio prevents multiple workers from processing the same batch
            # during a step
            assert self.worker_hosts
            shift_ratio = float(self.task_index) / len(self.worker_hosts)

        processor_class = self.dataset.get_image_preprocessor()
        assert processor_class
        return processor_class(
            image_size,
            image_size,
            self.batch_size * self.batch_group_size,
            len(self.devices) * self.batch_group_size,
            dtype=input_data_type,
            train=(not self.params.eval),
            distortions=self.params.distortions,
            resize_method=self.resize_method,
            eval_method=self.params.eval_method,
            shift_ratio=shift_ratio,
            summary_verbosity=self.params.summary_verbosity,
            distort_color_in_yiq=self.params.distort_color_in_yiq,
            fuse_decode_and_crop=self.params.fuse_decode_and_crop)

    def add_sync_queues_and_barrier(self, name_prefix,
                                    enqueue_after_list):
        """Adds ops to enqueue on all worker queues.

        Args:
            name_prefix: prefixed for the shared_name of ops.
            enqueue_after_list: control dependency from ops.

        Returns:
            an op that should be used as control dependency before starting next step.
        """
        self.sync_queue_counter += 1
        num_workers = self.cluster.num_tasks('worker')
        with tf.device(self.sync_queue_devices[
                self.sync_queue_counter % len(self.sync_queue_devices)]):
            sync_queues = [
                tf.FIFOQueue(num_workers, [tf.bool], shapes=[[]],
                                shared_name='%s%s' % (name_prefix, i))
                for i in range(num_workers)]
            queue_ops = []
            # For each other worker, add an entry in a queue, signaling that it can
            # finish this step.
            token = tf.constant(False)
            with tf.control_dependencies(enqueue_after_list):
                for i, q in enumerate(sync_queues):
                    if i == self.task_index:
                        queue_ops.append(tf.no_op())
                    else:
                        queue_ops.append(q.enqueue(token))

            # Drain tokens off queue for this worker, one for each other worker.
            queue_ops.append(
                sync_queues[self.task_index].dequeue_many(len(sync_queues) - 1))
            return tf.group(*queue_ops)
