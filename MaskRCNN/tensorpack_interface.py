# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
# File: interface.py

import tensorflow.compat.v1 as tf
import os
import sys
import time
from datetime import timedelta

import tensorpack_logger as logger
from tensorpack_callbacks import Callback, Callbacks, Monitors, MonitorBase, MaintainStepCounter
from tensorpack_utils import humanize_time_delta
from tensorpack_train import DEFAULT_MONITORS, TrainConfig
from tensorpack_tfutils import FilterNoneGrad, JustCurrentSession, backup_collection
from tabulate import tabulate
from config import config as cfg

from tensorpack_tfutils import get_current_tower_context, TrainContext, get_global_step_var

import horovod.tensorflow as hvd


__all__ = ['launch_train_with_config']

def _make_get_grad_fn(input, get_cost_fn, get_opt_fn, XLA_COMPILE=False):
    """
    Internal use only.

    Returns:
        a get_grad_fn for GraphBuilder to use.
    """
    assert input.setup_done()

    def get_grad_fn():
        ctx = get_current_tower_context()
        inputs = input.get_input_tensors()

        def compute_grad_from_inputs(*inputs):
            cost = get_cost_fn(*inputs)
            assert isinstance(cost, tf.Tensor), cost
            assert cost.shape.ndims == 0, "Cost must be a scalar, but found {}!".format(cost)

            if not ctx.is_training:
                return None     # this is the tower function, could be called for inference

            if ctx.has_own_variables:
                varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            else:
                varlist = tf.trainable_variables()

            if os.getenv("TENSORPACK_FP16"):

                loss_scale = 1024.0

                if os.getenv("CUSTOM_LOSS_SCALE"):
                    loss_scale = float(os.getenv("CUSTOM_LOSS_SCALE"))

                print(f'TENSORPACK_FP16 set. Using FP16 loss scaling of {loss_scale}')
                cost *= loss_scale

            opt = get_opt_fn()
            grads = opt.compute_gradients(
                cost, var_list=varlist,
                gate_gradients=False,
                colocate_gradients_with_ops=True,
                aggregation_method=tf.AggregationMethod.DEFAULT)
            grads = FilterNoneGrad().process(grads)

            if os.getenv("TENSORPACK_FP16"):
                grads = [(g * 1.0 / loss_scale, v) for g, v in grads]

            if os.getenv("TENSORPACK_SUMMARY_GRADIENT"):
                grads = SummaryGradient().process(grads)

            if os.getenv("TENSORPACK_FREEZE_VARS"):
                grads = [ (g - g, v) for g, v in grads ]

            return grads

        if not XLA_COMPILE:
            return compute_grad_from_inputs(*inputs)
        else:
            from tensorflow.python.compiler.xla import xla

            def xla_func():
                grads = compute_grad_from_inputs(*inputs)
                # unpack, because the return value
                # of xla function cannot have nested structure
                grads = [x[0] for x in grads]
                return grads

            grads_no_vars = xla.compile(xla_func)
            if ctx.has_own_variables:
                varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            else:
                varlist = tf.trainable_variables()
            return list(zip(grads_no_vars, varlist))

    return get_grad_fn

def allreduce(grads, compression=hvd.Compression.none, average=True):
    if hvd.size() == 1:
        return grads
    # copied from https://github.com/uber/horovod/blob/master/horovod/tensorflow/__init__.py
    averaged_gradients = []
    with tf.name_scope("AllReduce"):
        for grad, var in grads:
            if grad is not None:
                avg_grad = hvd.allreduce(grad, average=average, compression=compression)
                averaged_gradients.append((avg_grad, var))
            else:
                averaged_gradients.append((None, var))
    return averaged_gradients

def register_callback(cb):
    is_chief = hvd.rank() == 0
    if not is_chief and cb.chief_only:
        logger.warn("Callback {} is chief-only, skipped.".format(str(cb)))
        return False
    else:
        return True

def describe_trainable_vars():
    """
    Print a description of the current model parameters.
    Skip variables starting with "tower", as they are just duplicates built by data-parallel logic.
    """
    from termcolor import colored
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if len(train_vars) == 0:
        logger.warn("No trainable variables in the graph!")
        return
    total = 0
    total_bytes = 0
    data = []
    for v in train_vars:
        if v.name.startswith('tower'):
            continue
        shape = v.get_shape()
        ele = shape.num_elements()
        if ele is None:
            logger.warn("Shape of variable {} is not fully defined but {}.".format(v.name, shape))
            ele = 0
        try:
            shape = shape.as_list()
        except ValueError:
            shape = '<unknown>'

        total += ele
        total_bytes += ele * v.dtype.size
        data.append([v.name, shape, ele, v.device, v.dtype.base_dtype.name])
    headers = ['name', 'shape', 'dim', 'device', 'dtype']

    dtypes = set([x[4] for x in data])
    if len(dtypes) == 1:
        for x in data:
            del x[4]
        del headers[4]

    devices = set([x[3] for x in data])
    if len(devices) == 1:
        # don't log the device if all vars on the same device
        for x in data:
            del x[3]
        del headers[3]

    table = tabulate(data, headers=headers)

    size_mb = total_bytes / 1024.0**2
    summary_msg = colored(
        "\nTotal #vars={}, #params={}, size={:.02f}MB".format(
            len(data), total, size_mb), 'cyan')
    logger.info(colored("Trainable Variables: \n", 'cyan') + table + summary_msg)

class TrainLoop(object):
    """
    Manage the double for loop.
    """

    def __init__(self, monitors=None):
        self._epoch_num = 0
        self._global_step = 0
        self._local_step = -1
        self.monitors = monitors

    def config(self, steps_per_epoch, starting_epoch, max_epoch):
        """
        Configure the loop given the settings.
        """
        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)
        self.steps_per_epoch = int(steps_per_epoch)
        # Allow empty epoch (no steps), if we want to run the callbacks only.
        assert self.steps_per_epoch >= 0 and self.max_epoch >= 0

        self._epoch_num = starting_epoch - 1

    def update_global_step(self):
        """
        Update the Python-side global_step from TF.
        This must be called under initialized default session.
        """
        self._global_step =  tf.train.global_step(tf.get_default_session(), get_global_step_var())

    @property
    def epoch_num(self):
        """
        The number of the currently ongoing epoch.

        An epoch is defined to cover the moment before calling `before_epoch` until after calling `trigger_epoch`.
        i.e., in the `trigger_epoch` of epoch 3, `self.epoch_num` is 3.
        If you need use `self.epoch_num` in your callback, you'll need to know this.
        """
        return self._epoch_num

    @property
    def global_step(self):
        """
        The tensorflow global_step, i.e. how many times ``hooked_sess.run`` has been called.

        Note:
            1. global_step is incremented **after** each ``hooked_sess.run`` returns from TF runtime.
            2. If you make zero or more than one calls to ``hooked_sess.run`` in one
               :meth:`run_step`, local_step and global_step may increment at different speed.
        """
        return self._global_step

    @property
    def local_step(self):
        """
        The number of steps that have finished in the current epoch.
        """
        return self._local_step

def setup_callbacks(callbacks, monitors, loop):
    is_chief = hvd.rank() == 0
    cbs = []
    for cb in callbacks:
        if register_callback(cb):
            cbs.append(cb)
    for cb in cbs:
        assert not isinstance(cb, MonitorBase), "Monitor cannot be pre-registered for now!"
    registered_monitors = []
    for m in monitors:
        if register_callback(m):
            cbs.append(m)
            registered_monitors.append(m)
    monitors = Monitors(registered_monitors)
    loop.monitors = monitors
    if register_callback(monitors):
        cbs.append(monitors)  # monitors is also a callback

    # some final operations that might modify the graph
    logger.info("Setup callbacks graph ...")
    _callbacks = Callbacks(cbs)
    _callbacks.setup_graph(loop)
    return _callbacks

def get_default_sess_config(mem_fraction=0.99):
    """
    Return a tf.ConfigProto to use as default session config.
    You can modify the returned config to fit your needs.

    Args:
        mem_fraction(float): see the `per_process_gpu_memory_fraction` option
        in TensorFlow's GPUOptions protobuf:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto

    Returns:
        tf.ConfigProto: the config to use.
    """
    conf = tf.ConfigProto()

    conf.allow_soft_placement = True
    # conf.log_device_placement = True

    conf.intra_op_parallelism_threads = 1
    conf.inter_op_parallelism_threads = 0
    # TF benchmark use cpu_count() - gpu_thread_count(), e.g. 80 - 8 * 2
    # Didn't see much difference.

    conf.gpu_options.per_process_gpu_memory_fraction = mem_fraction

    # This hurt performance of large data pipeline:
    # https://github.com/tensorflow/benchmarks/commit/1528c46499cdcff669b5d7c006b7b971884ad0e6
    # conf.gpu_options.force_gpu_compatible = True

    conf.gpu_options.allow_growth = True

    # from tensorflow.core.protobuf import rewriter_config_pb2 as rwc
    # conf.graph_options.rewrite_options.memory_optimization = \
    #     rwc.RewriterConfig.HEURISTICS

    # May hurt performance?
    # conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # conf.graph_options.place_pruned_graph = True
    return conf

class ReuseSessionCreator(tf.train.SessionCreator):
    """
    Returns an existing session.
    """
    def __init__(self, sess):
        """
        Args:
            sess (tf.Session): the session to reuse
        """
        self.sess = sess

    def create_session(self):
        return self.sess


def initialize(session_init, _callbacks, target='', config=None):
    import multiprocessing as mp
    is_chief = hvd.rank() == 0

    with tf.name_scope('horovod_broadcast'):
        _broadcast_op = hvd.broadcast_global_variables(0)

    session_init._setup_graph()

    logger.info("Creating the session ...")

    if config is None:
        config = get_default_sess_config()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.inter_op_parallelism_threads = mp.cpu_count() // hvd.local_size()

    sess = tf.Session(target=target, config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    hooks = _callbacks.get_hooks()
    hooked_sess = tf.train.MonitoredSession(
        session_creator=ReuseSessionCreator(sess), hooks=hooks)

    if is_chief:
        logger.info("Initializing the session ...")
        session_init._run_init(sess)
    else:
        if not isinstance(session_init, JustCurrentSession):
            logger.warn("This is not a chief worker, 'session_init' was ignored!")

    sess.graph.finalize()
    logger.info("Graph Finalized.")

    if is_chief:
        logger.info("Broadcasting initialized variables ...")
    else:
        logger.info("Rank {} waiting for initialization broadcasting ...".format(hvd.rank()))
    sess.run(_broadcast_op)

    return sess, hooked_sess

class StopTraining(Exception):
    """
    An exception thrown to stop training.
    """
    pass

def launch_train_with_config(config):
    """
    Train with a :class:`TrainConfig` and a :class:`Trainer`, to
    present the simple and old training interface. It basically does the following
    3 things (and you can easily do them by yourself if you need more control):

    1. Setup the input with automatic prefetching heuristics,
       from `config.data` or `config.dataflow`.
    2. Call `trainer.setup_graph` with the input as well as `config.model`.
    3. Call `trainer.train` with rest of the attributes of config.

    Args:
        config (TrainConfig):
        trainer (Trainer): an instance of :class:`SingleCostTrainer`.

    Example:

    .. code-block:: python

        launch_train_with_config(
            config, SyncMultiGPUTrainerParameterServer(8, ps_device='gpu'))
    """
    #assert isinstance(config, TrainConfig), config
    assert config.model is not None
    assert config.dataflow is not None or config.data is not None

    model = config.model
    input = config.data or config.dataflow

    # This is the only place where the `ModelDesc` abstraction is useful.
    # We should gradually stay away from this unuseful abstraction.
    # TowerFuncWrapper is a better abstraction (similar to tf.defun in the future)
    inputs_desc = model.get_inputs_desc()
    _build_graph_get_cost = model._build_graph_get_cost
    get_opt_fn = model.get_optimizer

    # Special treatment for the eval callback
    if config.callbacks:
        config.callbacks[4].build_graph_func = _build_graph_get_cost
        config.callbacks[4].inputs_desc = inputs_desc

    # Setup inputs
    assert not input.setup_done()
    input_callbacks = input.setup(inputs_desc)

    # Setup graph
    with TrainContext(''):
        grads = _make_get_grad_fn(input, _build_graph_get_cost, get_opt_fn, XLA_COMPILE=False)()
        grads = allreduce(grads)

        opt = get_opt_fn()
        train_op = opt.apply_gradients(grads, name='train_op')

    describe_trainable_vars()

    # Setup all callbacks
    loop = TrainLoop()
    if config.callbacks:
        callbacks = input_callbacks + config.callbacks + config.extra_callbacks
    else:
        callbacks = input_callbacks
    callbacks.append(MaintainStepCounter())
    _callbacks = setup_callbacks(callbacks, DEFAULT_MONITORS(), loop)

    # Create session, load saved weights and run horovod broadcast op
    session_init = config.session_init or JustCurrentSession()
    sess, hooked_sess = initialize(session_init, _callbacks)

    # Start the training
    with sess.as_default():
        loop.config(config.steps_per_epoch, config.starting_epoch, config.max_epoch)
        loop.update_global_step()
        try:
            _callbacks.before_train()
            # refresh global step (might have changed by callbacks) TODO ugly
            # what if gs is changed later?
            loop.update_global_step()
            for loop._epoch_num in range(
                    loop.starting_epoch, loop.max_epoch + 1):
                logger.info("Start Epoch {} ...".format(loop.epoch_num))
                _callbacks.before_epoch()
                start_time = time.time()
                for loop._local_step in range(loop.steps_per_epoch):
                    if hooked_sess.should_stop():
                        return
                    if cfg.TRAIN.SHOULD_STOP:
                        logger.info("Target accuracy has been reached, stop.....")
                        return
                    hooked_sess.run(train_op)
                    _callbacks.trigger_step()
                _callbacks.after_epoch()
                logger.info("Epoch {} (global_step {}) finished, time:{}.".format(
                    loop.epoch_num, loop.global_step, humanize_time_delta(time.time() - start_time)))

                # trigger epoch outside the timing region.
                _callbacks.trigger_epoch()
            logger.info("Training has finished!")
        except (StopTraining, tf.errors.OutOfRangeError) as e:
            logger.info("Training was stopped by exception {}.".format(str(e)))
        except KeyboardInterrupt:
            logger.info("Detected Ctrl-C and exiting main loop.")
            raise
        finally:
            _callbacks.after_train()
            hooked_sess.close()
