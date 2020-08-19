import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from time import time as timer
import six
import re
import multiprocessing as mp
import numpy as np
import tensorflow.compat.v1 as tf
from abc import ABCMeta, abstractmethod
import json
import operator
import os
import shutil
import time
from datetime import datetime
import tqdm
from six.moves import zip


from tensorpack_tfutils import get_op_or_tensor_by_name, get_global_step_var, get_op_tensor_name, \
                                create_image_summary, create_scalar_summary, MOVING_SUMMARY_OPS_KEY, gpu_available_in_session
from tensorpack_utils import humanize_time_delta, GLOBAL_STEP_INCR_OP_NAME, get_tqdm_kwargs, \
                             start_proc_mask_signal, HIDE_DOC, StoppableThread
import tensorpack_logger as logger

if six.PY3:
    from time import perf_counter as timer

@six.add_metaclass(ABCMeta)
class Callback(object):
    """ Base class for all callbacks. See
    `Write a Callback
    <http://tensorpack.readthedocs.io/tutorial/extend/callback.html>`_
    for more detailed explanation of the callback methods.

    Attributes:
        epoch_num(int): trainer.epoch_num
        global_step(int): trainer.global_step
        local_step(int): trainer.local_step
        trainer(Trainer): the trainer.
        graph(tf.Graph): the graph.

    Note:
        These attributes are available only after (and including)
        :meth:`_setup_graph`.

    .. document private functions
    .. automethod:: _setup_graph
    .. automethod:: _before_train
    .. automethod:: _after_train
    .. automethod:: _before_run
    .. automethod:: _after_run
    .. automethod:: _before_epoch
    .. automethod:: _after_epoch
    .. automethod:: _trigger_step
    .. automethod:: _trigger_epoch
    .. automethod:: _trigger
    """

    _chief_only = True

    def setup_graph(self, trainer):
        self.trainer = trainer
        self.graph = tf.get_default_graph()
        scope_name = type(self).__name__
        scope_name = scope_name.replace('_', '')
        with tf.name_scope(scope_name):
            self._setup_graph()

    def _setup_graph(self):
        """
        Called before finalizing the graph.
        Override this method to setup the ops used in the callback.
        This is the same as ``tf.train.SessionRunHook.begin()``.
        """
        pass

    def before_train(self):
        self._before_train()

    def _before_train(self):
        """
        Called right before the first iteration. The main difference to
        `setup_graph` is that at this point the graph is finalized and a default session is initialized.
        Override this method to, e.g. run some operations under the session.

        This is similar to ``tf.train.SessionRunHook.after_create_session()``, but different:
        it is called after the session is initialized by :class:`tfutils.SessionInit`.
        """
        pass

    def before_epoch(self):
        self._before_epoch()

    def _before_epoch(self):
        """
        Called right before each epoch.
        Usually you should use the :meth:`trigger` callback to run something between epochs.
        Use this method only when something really needs to be run **immediately** before each epoch.
        """
        pass

    def after_epoch(self):
        self._after_epoch()

    def _after_epoch(self):
        """
        Called right after each epoch.
        Usually you should use the :meth:`trigger` callback to run something between epochs.
        Use this method only when something really needs to be run **immediately** after each epoch.
        """
        pass

    def before_run(self, ctx):
        fetches = self._before_run(ctx)
        if fetches is None:
            return None
        if isinstance(fetches, tf.train.SessionRunArgs):
            return fetches

        # also support list of names
        assert isinstance(fetches, list), fetches
        ret = []
        for f in fetches:
            if isinstance(f, (tf.Tensor, tf.Operation)):
                ret.append(f)
            else:
                # warn about speed
                ret.append(get_op_or_tensor_by_name(f))
        return tf.train.SessionRunArgs(fetches=ret)

    def _before_run(self, ctx):
        """
        It is called before every ``hooked_sess.run()`` call, and it
        registers some extra op/tensors to run in the next call.
        This method is the same as ``tf.train.SessionRunHook.before_run``.
        Refer to TensorFlow docs for more details.
        """
        return None

    def after_run(self, run_context, run_values):
        self._after_run(run_context, run_values)

    def _after_run(self, run_context, run_values):
        """
        It is called after every ``hooked_sess.run()`` call, and it
        processes the values requested by the corresponding :meth:`before_run`.
        It is equivalent to ``tf.train.SessionRunHook.after_run()``, refer to
        TensorFlow docs for more details.
        """
        pass

    def trigger_step(self):
        self._trigger_step()

    def _trigger_step(self):
        """
        Called after each :meth:`Trainer.run_step()` completes. Defaults to no-op.

        You can override it to implement, e.g. a ProgressBar.
        """
        pass

    def trigger_epoch(self):
        self._trigger_epoch()

    def _trigger_epoch(self):
        """
        Called after the completion of every epoch. Defaults to call ``self.trigger()``
        """
        self.trigger()

    def trigger(self):
        self._trigger()

    def _trigger(self):
        """
        Override this method to define a general trigger behavior, to be used with trigger schedulers.
        Note that the schedulers (e.g. :class:`PeriodicTrigger`) might call this
        method both inside an epoch and after an epoch.

        When used without the scheduler, this method by default will be called by `trigger_epoch()`.
        """
        pass

    def after_train(self):
        self._after_train()

    def _after_train(self):
        """
        Called after training.
        """
        pass

    @property
    def epoch_num(self):
        return self.trainer.epoch_num

    @property
    def global_step(self):
        return self.trainer.global_step

    @property
    def local_step(self):
        return self.trainer.local_step

    @property
    def chief_only(self):
        """
        Only run this callback on chief training process.

        Returns: bool
        """
        return self._chief_only

    @chief_only.setter
    def chief_only(self, v):
        self._chief_only = v

    def set_chief_only(self, v=True):
        """
        Set chief_only property, and returns the callback itself.
        """
        self._chief_only = v
        return self

    def __str__(self):
        return type(self).__name__

    # TODO RENAME: same function to be used to get ops as well
    def get_tensors_maybe_in_tower(self, names):
        """
        Get tensors in the graph with the given names.
        Will automatically check for the *first training tower*
        if no existing tensor is found with the name.

        Returns:
            [tf.Tensor]
        """
        from ..train.tower import TowerTrainer  # noqa

        def get_tensor(name):
            msg = "Tensor {} not found in the graph!".format(name)
            try:
                return get_op_or_tensor_by_name(name)
            except KeyError:
                pass
            if not isinstance(self.trainer, TowerTrainer):
                raise KeyError(msg)
            towers = self.trainer.towers
            try:
                return towers.training()[0][name]
            except KeyError:
                raise KeyError(msg)
        return [get_tensor(name) for name in names]


class ModelSaver(Callback):
    """
    Save the model once triggered.
    """

    def __init__(self, max_to_keep=10,
                 keep_checkpoint_every_n_hours=0.5,
                 checkpoint_dir=None,
                 var_collections=None):
        """
        Args:
            max_to_keep (int): the same as in ``tf.train.Saver``.
            keep_checkpoint_every_n_hours (float): the same as in ``tf.train.Saver``.
                Note that "keep" does not mean "create", but means "don't delete".
            checkpoint_dir (str): Defaults to ``logger.get_logger_dir()``.
            var_collections (str or list of str): collection of the variables (or list of collections) to save.
        """
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        self._max_to_keep = max_to_keep
        self._keep_every_n_hours = keep_checkpoint_every_n_hours

        if not isinstance(var_collections, list):
            var_collections = [var_collections]
        self.var_collections = var_collections
        if checkpoint_dir is None:
            checkpoint_dir = logger.get_logger_dir()
        if checkpoint_dir is not None:
            if not tf.gfile.IsDirectory(checkpoint_dir):
                tf.gfile.MakeDirs(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

    def _setup_graph(self):
        assert self.checkpoint_dir is not None, \
            "ModelSaver() doesn't have a valid checkpoint directory."
        vars = []
        for key in self.var_collections:
            vars.extend(tf.get_collection(key))
        vars = list(set(vars))
        self.path = os.path.join(self.checkpoint_dir, 'model')
        self.saver = tf.train.Saver(
            var_list=vars,
            max_to_keep=self._max_to_keep,
            keep_checkpoint_every_n_hours=self._keep_every_n_hours,
            write_version=tf.train.SaverDef.V2,
            save_relative_paths=True)
        # Scaffold will call saver.build from this collection
        tf.add_to_collection(tf.GraphKeys.SAVERS, self.saver)

    def _before_train(self):
        # graph is finalized, OK to write it now.
        time = datetime.now().strftime('%m%d-%H%M%S')
        self.saver.export_meta_graph(
            os.path.join(self.checkpoint_dir,
                         'graph-{}.meta'.format(time)),
            collection_list=self.graph.get_all_collection_keys())

    def _trigger(self):
        try:
            self.saver.save(
                tf.get_default_session(),
                self.path,
                global_step=tf.train.get_global_step(),
                write_meta_graph=False)
            logger.info("Model saved to %s." % tf.train.get_checkpoint_state(self.checkpoint_dir).model_checkpoint_path)
        except (OSError, IOError, tf.errors.PermissionDeniedError,
                tf.errors.ResourceExhaustedError):   # disk error sometimes.. just ignore it
            logger.exception("Exception in ModelSaver!")


class PeakMemoryTracker(Callback):
    """
    Track peak memory used on each GPU device every epoch, by :mod:`tf.contrib.memory_stats`.
    The peak memory comes from the `MaxBytesInUse` op, which might span
    multiple session.run.
    See https://github.com/tensorflow/tensorflow/pull/13107.
    """

    _chief_only = False

    def __init__(self, devices=[0]):
        """
        Args:
            devices([int] or [str]): list of GPU devices to track memory on.
        """
        assert isinstance(devices, (list, tuple)), devices
        devices = ['/gpu:{}'.format(x) if isinstance(x, int) else x for x in devices]
        self._devices = devices

    def _setup_graph(self):
        from tensorflow.contrib.memory_stats import MaxBytesInUse
        ops = []
        for dev in self._devices:
            with tf.device(dev):
                ops.append(MaxBytesInUse())
        self._fetches = tf.train.SessionRunArgs(fetches=ops)

    def _before_train(self):
        assert gpu_available_in_session(), "PeakMemoryTracker only supports GPU!"

    def _before_run(self, _):
        if self.local_step == self.trainer.steps_per_epoch - 1:
            return self._fetches
        return None

    def _after_run(self, _, rv):
        results = rv.results
        if results is not None:
            for mem, dev in zip(results, self._devices):
                self.trainer.monitors.put_scalar('PeakMemory(MB)' + dev, mem / 1e6)


class GraphProfiler(Callback):
    """
    Enable profiling by installing session hooks,
    and write tracing files / events / metadata to ``logger.get_logger_dir()``.

    The tracing files can be loaded from ``chrome://tracing``.
    The metadata files can be processed by
    `tfprof command line utils
    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/README.md>`_.
    The event is viewable from tensorboard.

    Tips:

    Note that the profiling is by default enabled for every step and is expensive.
    You probably want to schedule it less frequently, e.g.:

    .. code-block:: none

        EnableCallbackIf(
            GraphProfiler(dump_tracing=True, dump_event=True),
            lambda self: self.trainer.global_step > 20 and self.trainer.global_step < 30)
    """
    def __init__(self, dump_metadata=False, dump_tracing=True, dump_event=False):
        """
        Args:
            dump_metadata(bool): Dump :class:`tf.RunMetadata` to be used with tfprof.
            dump_tracing(bool): Dump chrome tracing files.
            dump_event(bool): Dump to an event processed by FileWriter and
                will be shown in TensorBoard.
        """
        self._dir = logger.get_logger_dir()
        self._dump_meta = bool(dump_metadata)
        self._dump_tracing = bool(dump_tracing)
        self._dump_event = bool(dump_event)
        assert os.path.isdir(self._dir), self._dir

    def _before_run(self, _):
        opt = tf.RunOptions()
        opt.trace_level = tf.RunOptions.FULL_TRACE
        return tf.train.SessionRunArgs(fetches=None, options=opt)

    def _after_run(self, _, run_values):
        meta = run_values.run_metadata
        if self._dump_meta:
            self._write_meta(meta)
        if self._dump_tracing:
            self._write_tracing(meta)
        if self._dump_event:
            self._write_event(meta)

    def _write_meta(self, metadata):
        fname = os.path.join(
            self._dir, 'runmetadata-{}.pb'.format(self.global_step))
        with open(fname, 'wb') as f:
            f.write(metadata.SerializeToString())

    def _write_tracing(self, metadata):
        from tensorflow.python.client import timeline
        tl = timeline.Timeline(step_stats=metadata.step_stats)
        fname = os.path.join(
            self._dir, 'chrome-trace-{}.json'.format(self.global_step))
        with open(fname, 'w') as f:
            f.write(tl.generate_chrome_trace_format(
                show_dataflow=True, show_memory=True))

    def _write_event(self, metadata):
        evt = tf.Event()
        evt.tagged_run_metadata.tag = 'trace-{}'.format(self.global_step)
        evt.tagged_run_metadata.run_metadata = metadata.SerializeToString()
        self.trainer.monitors.put_event(evt)


class EstimatedTimeLeft(Callback):
    """
    Estimate the time left until completion of training.
    """
    def __init__(self, last_k_epochs=5, median=False):
        """
        Args:
            last_k_epochs (int): Use the time spent on last k epochs to estimate total time left.
            median (bool): Use mean by default. If True, use the median time spent on last k epochs.
        """
        self._times = deque(maxlen=last_k_epochs)
        self._median = median

    def _before_train(self):
        self._max_epoch = self.trainer.max_epoch
        self._last_time = time.time()

    def _trigger_epoch(self):
        duration = time.time() - self._last_time
        self._last_time = time.time()
        self._times.append(duration)

        epoch_time = np.median(self._times) if self._median else np.mean(self._times)
        time_left = (self._max_epoch - self.epoch_num) * epoch_time
        if time_left > 0:
            logger.info("Estimated Time Left: " + humanize_time_delta(time_left))


class SessionRunTimeout(Callback):
    """
    Add timeout option to each sess.run call.
    """
    def __init__(self, timeout_in_ms):
        """
        Args:
            timeout_in_ms (int):
        """
        self._timeout = int(timeout_in_ms)

        opt = tf.RunOptions(timeout_in_ms=timeout_in_ms)
        self._runargs = tf.train.SessionRunArgs(fetches=[], options=opt)

    def _before_run(self, _):
        return self._runargs

@six.add_metaclass(ABCMeta)
class HyperParam(object):
    """ Base class for a hyperparam. """

    def setup_graph(self):
        """ setup the graph in ``setup_graph`` callback stage, if necessary"""
        pass

    @abstractmethod
    def set_value(self, v):
        """
        Set the value of the param.

        Args:
            v: the value to be set
        """
        pass

    @abstractmethod
    def get_value(self):
        """
        Get the value of the param.
        """
        pass

    @property
    def readable_name(self):
        """ A name to display """
        return self._readable_name


class GraphVarParam(HyperParam):
    """ A variable in the graph (e.g. learning_rate) can be a hyperparam."""

    def __init__(self, name, shape=[]):
        """
        Args:
            name(str): name of the variable.
            shape(list): shape of the variable.
        """
        self.name = name
        self.shape = shape
        self._readable_name, self.var_name = get_op_tensor_name(name)

    def setup_graph(self):
        """ Will setup the assign operator for that variable. """
        all_vars = tf.global_variables() + tf.local_variables()
        for v in all_vars:
            if v.name == self.var_name:
                self.var = v
                break
        else:
            raise ValueError("{} is not a variable in the graph!".format(self.var_name))

    def set_value(self, v):
        """ Assign the variable a new value. """
        self.var.load(v)

    def get_value(self):
        """ Evaluate the variable. """
        return self.var.eval()

class HyperParamSetter(Callback):
    """
    An abstract base callback to set hyperparameters.

    Once the :meth:`trigger()` method is called,
    the method :meth:`_get_value_to_set` will be used to get a new value for the hyperparameter.
    """

    _chief_only = False

    """
    Also enable this hyperparam setter in the :meth:`before_train` method.
    """
    _enable_before_train = True

    def __init__(self, param):
        """
        Args:
            param(HyperParam or str): if is a :class:`str`, it is assumed to
                be a :class:`GraphVarParam`.
        """
        # if a string, assumed to be a scalar graph variable
        if isinstance(param, six.string_types):
            param = GraphVarParam(param)
        assert isinstance(param, HyperParam), type(param)
        self.param = param
        self._last_value = None
        self._last_epoch_set = -1

    def _setup_graph(self):
        self.param.setup_graph()

    def get_value_to_set(self):
        """
        Returns:
            The value to assign to the variable.

        Note:
            Subclasses will implement the abstract method
            :meth:`_get_value_to_set`, which should return a new value to
            set, or return None to do nothing.
        """
        ret = self._get_value_to_set()
        if ret is not None and ret != self._last_value:
            if self.epoch_num != self._last_epoch_set:
                # Print this message at most once every epoch
                if self._last_value is None:
                    logger.info("[HyperParamSetter] At global_step={}, {} is set to {:.6f}".format(
                        self.global_step, self.param.readable_name, ret))
                else:
                    logger.info("[HyperParamSetter] At global_step={}, {} changes from {:.6f} to {:.6f}".format(
                        self.global_step, self.param.readable_name, self._last_value, ret))
            self._last_epoch_set = self.epoch_num
            self._last_value = ret
        return ret

    @abstractmethod
    def _get_value_to_set(self):
        pass

    def get_current_value(self):
        """
        Returns:
            The current value of the param.
        """
        return self.param.get_value()

    def _trigger(self):
        self._set_param()

    def _before_train(self):
        if self._enable_before_train:
            self._set_param()

    def _set_param(self):
        v = self.get_value_to_set()
        if v is not None:
            self.param.set_value(v)


class ScheduledHyperParamSetter(HyperParamSetter):
    """
    Set hyperparameters by a predefined epoch-based schedule.
    """

    def __init__(self, param, schedule, interp=None, step_based=False):
        """
        Args:
            param: same as in :class:`HyperParamSetter`.
            schedule (list): with the format ``[(epoch1, val1), (epoch2, val2), (epoch3, val3)]``.
                Each ``(ep, val)`` pair means to set the param
                to "val" **after** the completion of epoch `ep`.
                If ep == 0, the value will be set before the first epoch
                (because by default the first is epoch 1).
                The epoch numbers have to be increasing.
            interp (str or None): Either None or 'linear'.
                If None, the parameter will only be set when the specific epoch or steps
                is reached exactly. If 'linear', perform linear interpolation (but no extrapolation)
                every time this callback is triggered.
            step_based (bool): interpret ``schedule`` as (step, value) instead
                of (epoch, value).

        Example:
            .. code-block:: python

                ScheduledHyperParamSetter('learning_rate',
                                          [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
        """
        schedule = [(int(a), float(b)) for a, b in schedule]
        self.schedule = sorted(schedule, key=operator.itemgetter(0))
        if interp is not None:
            assert interp == 'linear'
        self.interp = interp
        self._step = step_based
        super(ScheduledHyperParamSetter, self).__init__(param)

    def _get_value_to_set(self):
        refnum = self.global_step if self._step else self.epoch_num
        laste, lastv = None, None
        for e, v in self.schedule:
            if e == refnum:
                return v    # meet the exact boundary, return directly
            if e > refnum:
                break
            laste, lastv = e, v
        if laste is None or laste == e:
            # hasn't reached the first scheduled point, or reached the end of all scheduled points
            return None
        if self.interp is None:
            # If no interpolation, nothing to do.
            return None
        v = (refnum - laste) * 1. / (e - laste) * (v - lastv) + lastv
        return v

    def _trigger_epoch(self):
        if not self._step:
            self.trigger()

    def _trigger_step(self):
        if self._step:
            self.trigger()


class ProxyCallback(Callback):
    """ A callback which proxy all methods to another callback.
        It's useful as a base class of callbacks which decorate other callbacks.
    """

    def __init__(self, cb):
        """
        Args:
            cb(Callback): the underlying callback
        """
        assert isinstance(cb, Callback), type(cb)
        self.chief_only = cb.chief_only
        self.cb = cb

    def _before_train(self):
        self.cb.before_train()

    def _setup_graph(self):
        with tf.name_scope(None):
            self.cb.setup_graph(self.trainer)

    def _trigger_epoch(self):
        self.cb.trigger_epoch()

    def _trigger(self):
        self.cb.trigger()

    def _trigger_step(self):
        self.cb.trigger_step()

    def _after_train(self):
        self.cb.after_train()

    def _before_epoch(self):
        self.cb.before_epoch()

    def _after_epoch(self):
        self.cb.after_epoch()

    def _before_run(self, ctx):
        return self.cb._before_run(ctx)

    def _after_run(self, ctx, run_values):
        self.cb._after_run(ctx, run_values)

    def __str__(self):
        return "Proxy-" + str(self.cb)


class EnableCallbackIf(ProxyCallback):
    """
    Disable the ``{before,after}_epoch``, ``{before,after}_run``,
    ``trigger_{epoch,step}``
    methods of a callback, unless some condition satisfies.
    The other methods are unaffected.

    A more accurate name for this callback should be "DisableCallbackUnless", but that's too ugly.

    Note:
        If you use ``{before,after}_run``,
        ``pred`` will be evaluated only in ``before_run``.
    """

    def __init__(self, callback, pred):
        """
        Args:
            callback (Callback):
            pred (self -> bool): a callable predicate. Has to be a pure function.
                The callback is disabled unless this predicate returns True.
        """
        self._pred = pred
        super(EnableCallbackIf, self).__init__(callback)

    def _before_run(self, ctx):
        if self._pred(self):
            self._enabled = True
            return super(EnableCallbackIf, self)._before_run(ctx)
        else:
            self._enabled = False

    def _after_run(self, ctx, rv):
        if self._enabled:
            super(EnableCallbackIf, self)._after_run(ctx, rv)

    def _before_epoch(self):
        if self._pred(self):
            super(EnableCallbackIf, self)._before_epoch()

    def _after_epoch(self):
        if self._pred(self):
            super(EnableCallbackIf, self)._after_epoch()

    def _trigger_epoch(self):
        if self._pred(self):
            super(EnableCallbackIf, self)._trigger_epoch()

    def _trigger_step(self):
        if self._pred(self):
            super(EnableCallbackIf, self)._trigger_step()

    def __str__(self):
        return "EnableCallbackIf-" + str(self.cb)


class PeriodicCallback(EnableCallbackIf):
    """
    The ``{before,after}_epoch``, ``{before,after}_run``, ``trigger_{epoch,step}``
    methods of the given callback will be enabled only when ``global_step % every_k_steps == 0`
    or ``epoch_num % every_k_epochs == 0``. The other methods are unaffected.

    Note that this can only makes a callback **less** frequent than itself.
    If you have a callback that by default runs every epoch by its :meth:`trigger()` method,
    use :class:`PeriodicTrigger` to schedule it more frequent than itself.
    """

    def __init__(self, callback, every_k_steps=None, every_k_epochs=None):
        """
        Args:
            callback (Callback): a Callback instance.
            every_k_steps (int): enable the callback when ``global_step % k == 0``. Set to
                None to ignore.
            every_k_epochs (int): enable the callback when ``epoch_num % k == 0``.
                Also enable when the last step finishes (``epoch_num == max_epoch``
                and ``local_step == steps_per_epoch - 1``). Set to None to ignore.

        every_k_steps and every_k_epochs can be both set, but cannot be both None.
        """
        assert isinstance(callback, Callback), type(callback)
        assert (every_k_epochs is not None) or (every_k_steps is not None), \
            "every_k_steps and every_k_epochs cannot be both None!"
        self._step_k = every_k_steps
        self._epoch_k = every_k_epochs
        super(PeriodicCallback, self).__init__(callback, PeriodicCallback.predicate)

    def predicate(self):
        if self._step_k is not None and self.global_step % self._step_k == 0:
            return True
        if self._epoch_k is not None and self.epoch_num % self._epoch_k == 0:
            return True
        if self._epoch_k is not None:
            if self.local_step == self.trainer.steps_per_epoch - 1 and \
                    self.epoch_num == self.trainer.max_epoch:
                return True
        return False

    def __str__(self):
        return "PeriodicCallback-" + str(self.cb)


class CallbackToHook(tf.train.SessionRunHook):
    """ This is only for internal implementation of
        before_run/after_run callbacks.
        You shouldn't need to use this.
    """

    def __init__(self, cb):
        self._cb = cb

    def before_run(self, ctx):
        return self._cb.before_run(ctx)

    def after_run(self, ctx, vals):
        self._cb.after_run(ctx, vals)


class CallbackTimeLogger(object):
    def __init__(self):
        self.times = []
        self.tot = 0

    def add(self, name, time):
        self.tot += time
        self.times.append((name, time))

    @contextmanager
    def timed_callback(self, name):
        s = timer()
        yield
        self.add(name, timer() - s)

    def log(self):

        """ log the time of some heavy callbacks """
        if self.tot < 3:
            return
        msgs = []
        for name, t in self.times:
            if t / self.tot > 0.3 and t > 1:
                msgs.append(name + ": " + humanize_time_delta(t))
        logger.info(
            "Callbacks took {:.3f} sec in total. {}".format(
                self.tot, '; '.join(msgs)))


class Callbacks(Callback):
    """
    A container to hold all callbacks, and trigger them iteratively.
    Note that it does nothing to before_run/after_run.
    """

    def __init__(self, cbs):
        """
        Args:
            cbs(list): a list of :class:`Callback` instances.
        """
        # check type
        #for cb in cbs:
        #    assert isinstance(cb, Callback), cb.__class__
        self.cbs = cbs

    def _setup_graph(self):
        with tf.name_scope(None):   # clear the name scope
            for cb in self.cbs:
                cb.setup_graph(self.trainer)

    def _before_train(self):
        for cb in self.cbs:
            cb.before_train()

    def _after_train(self):
        for cb in self.cbs:
            # make sure callbacks are properly finalized
            try:
                cb.after_train()
            except Exception:
                traceback.print_exc()

    def get_hooks(self):
        return [CallbackToHook(cb) for cb in self.cbs]

    def trigger_step(self):
        for cb in self.cbs:
            cb.trigger_step()

    def _trigger_epoch(self):
        tm = CallbackTimeLogger()

        for cb in self.cbs:
            display_name = str(cb)
            with tm.timed_callback(display_name):
                cb.trigger_epoch()
        tm.log()

    def _before_epoch(self):
        for cb in self.cbs:
            cb.before_epoch()

    def _after_epoch(self):
        for cb in self.cbs:
            cb.after_epoch()


class MonitorBase(Callback):
    """
    Base class for monitors which monitor a training progress, by processing different types of
    summary/statistics from trainer.

    .. document private functions
    .. automethod:: _setup_graph
    """

    _chief_only = False

    def setup_graph(self, trainer):
        self.trainer = trainer
        self._setup_graph()

    def _setup_graph(self):
        """ Override this method to setup the monitor."""
        pass

    def process_summary(self, summary):
        """
        Process a tf.Summary.
        """
        pass

    def process(self, name, val):
        """
        Process a key-value pair.
        """
        pass

    def process_scalar(self, name, val):
        """
        Args:
            val: a scalar
        """
        pass

    def process_image(self, name, val):
        """
        Args:
            val (np.ndarray): 4D (NHWC) numpy array of images in range [0,255].
                If channel is 3, assumed to be RGB.
        """
        pass

    def process_event(self, evt):
        """
        Args:
            evt (tf.Event): the most basic format acceptable by tensorboard.
                It could include Summary, RunMetadata, LogMessage, and more.
        """
        pass


class NoOpMonitor(MonitorBase):
    def __init__(self, name=None):
        self._name = name

    def __str__(self):
        if self._name is None:
            return "NoOpMonitor"
        return "NoOpMonitor({})".format(self._name)


class ScalarHistory(MonitorBase):
    """
    Only internally used by monitors.
    """

    def __init__(self):
        self._dic = defaultdict(list)

    #@HIDE_DOC
    def process_scalar(self, name, val):
        self._dic[name].append((self.global_step, float(val)))

    def get_latest(self, name):
        hist = self._dic[name]
        if len(hist) == 0:
            raise KeyError("No available data for the key: {}".format(name))
        else:
            return hist[-1]

    def get_history(self, name):
        return self._dic[name]


class ScalarPrinter(MonitorBase):
    """
    Print scalar data into terminal.
    """

    def __init__(self, enable_step=False, enable_epoch=True,
                 whitelist=None, blacklist=None):
        """
        Args:
            enable_step, enable_epoch (bool): whether to print the
                monitor data (if any) between steps or between epochs.
            whitelist (list[str] or None): A list of regex. Only names
                matching some regex will be allowed for printing.
                Defaults to match all names.
            blacklist (list[str] or None): A list of regex. Names matching
                any regex will not be printed. Defaults to match no names.
        """
        def compile_regex(rs):
            if rs is None:
                return None
            rs = set([re.compile(r) for r in rs])
            return rs

        self._whitelist = compile_regex(whitelist)
        if blacklist is None:
            blacklist = []
        self._blacklist = compile_regex(blacklist)

        self._enable_step = enable_step
        self._enable_epoch = enable_epoch
        self._dic = {}

    # in case we have something to log here.
    def _before_train(self):
        self._trigger()

    def _trigger_step(self):
        if self._enable_step:
            if self.local_step != self.trainer.steps_per_epoch - 1:
                # not the last step
                self._trigger()
            else:
                if not self._enable_epoch:
                    self._trigger()
                # otherwise, will print them together

    def _trigger_epoch(self):
        if self._enable_epoch:
            self._trigger()

    @HIDE_DOC
    def process_scalar(self, name, val):
        self._dic[name] = float(val)

    def _trigger(self):
        # Print stats here
        def match_regex_list(regexs, name):
            for r in regexs:
                if r.search(name) is not None:
                    return True
            return False

        for k, v in sorted(self._dic.items(), key=operator.itemgetter(0)):
            if self._whitelist is None or \
                    match_regex_list(self._whitelist, k):
                if not match_regex_list(self._blacklist, k):
                    logger.info('{}: {:.5g}'.format(k, v))
        self._dic = {}


class TFEventWriter(MonitorBase):
    """
    Write summaries to TensorFlow event file.
    """
    def __init__(self, logdir=None, max_queue=10, flush_secs=120, split_files=False):
        """
        Args:
            logdir: ``logger.get_logger_dir()`` by default.
            max_queue, flush_secs: Same as in :class:`tf.summary.FileWriter`.
            split_files: if True, split events to multiple files rather than
                append to a single file. Useful on certain filesystems where append is expensive.
        """
        if logdir is None:
            logdir = logger.get_logger_dir()
        assert tf.gfile.IsDirectory(logdir), logdir
        self._logdir = logdir
        self._max_queue = max_queue
        self._flush_secs = flush_secs
        self._split_files = split_files

    def __new__(cls, logdir=None, max_queue=10, flush_secs=120, **kwargs):
        if logdir is None:
            logdir = logger.get_logger_dir()

        if logdir is not None:
            return super(TFEventWriter, cls).__new__(cls)
        else:
            logger.warn("logger directory was not set. Ignore TFEventWriter.")
            return NoOpMonitor("TFEventWriter")

    def _setup_graph(self):
        self._writer = tf.summary.FileWriter(
            self._logdir, graph=tf.get_default_graph(),
            max_queue=self._max_queue, flush_secs=self._flush_secs)

    @HIDE_DOC
    def process_summary(self, summary):
        self._writer.add_summary(summary, self.global_step)

    @HIDE_DOC
    def process_event(self, evt):
        self._writer.add_event(evt)

    def _trigger(self):     # flush every epoch
        self._writer.flush()
        if self._split_files:
            self._writer.close()
            self._writer.reopen()  # open new file

    def _after_train(self):
        self._writer.close()


class Monitors(Callback):
    """
    Merge monitors together for trainer to use.

    In training, each trainer will create a :class:`Monitors` instance,
    and you can access it through `trainer.monitors`.
    You should use `trainer.monitors` for logging and it will dispatch your
    logs to each sub-monitor.
    """

    _chief_only = False

    def __init__(self, monitors):
        self._scalar_history = ScalarHistory()
        self._monitors = monitors + [self._scalar_history]
        #for m in self._monitors:
        #    assert isinstance(m, MonitorBase), m

    def _setup_graph(self):
        # scalar_history's other methods were not called.
        # but they are not useful for now
        self._scalar_history.setup_graph(self.trainer)

    def _dispatch(self, func):
        for m in self._monitors:
            func(m)

    def put_summary(self, summary):
        """
        Put a `tf.Summary`.
        """
        if isinstance(summary, six.binary_type):
            summary = tf.Summary.FromString(summary)
        assert isinstance(summary, tf.Summary), type(summary)

        # TODO other types
        for val in summary.value:
            if val.WhichOneof('value') == 'simple_value':
                val.tag = re.sub('tower[0-9]+/', '', val.tag)   # TODO move to subclasses

                # TODO This hack is still needed, seem to disappear only when
                # compiled from source.
                suffix = '-summary'  # tensorflow#6150, tensorboard#59
                if val.tag.endswith(suffix):
                    val.tag = val.tag[:-len(suffix)]

                self._dispatch(lambda m: m.process_scalar(val.tag, val.simple_value))

        self._dispatch(lambda m: m.process_summary(summary))

    def put_scalar(self, name, val):
        """
        Put a scalar.
        """
        if isinstance(val, np.floating):
            val = float(val)
        if isinstance(val, np.integer):
            val = int(val)
        self._dispatch(lambda m: m.process_scalar(name, val))
        s = create_scalar_summary(name, val)
        self._dispatch(lambda m: m.process_summary(s))

    def put_image(self, name, val):
        """
        Put an image.

        Args:
            name (str):
            val (np.ndarray): 2D, 3D (HWC) or 4D (NHWC) numpy array of images
                in range [0,255]. If channel is 3, assumed to be RGB.
        """
        assert isinstance(val, np.ndarray)
        arr = image_to_nhwc(val)
        self._dispatch(lambda m: m.process_image(name, arr))
        s = create_image_summary(name, arr)
        self._dispatch(lambda m: m.process_summary(s))

    def put_event(self, evt):
        """
        Put an :class:`tf.Event`.
        `step` and `wall_time` fields of :class:`tf.Event` will be filled automatically.

        Args:
            evt (tf.Event):
        """
        evt.step = self.global_step
        evt.wall_time = time.time()
        self._dispatch(lambda m: m.process_event(evt))

    def get_latest(self, name):
        """
        Get latest scalar value of some data.

        If you run multiprocess training, keep in mind that
        the data is perhaps only available on chief process.

        Returns:
            scalar
        """
        return self._scalar_history.get_latest(name)[1]

    def get_history(self, name):
        """
        Get a history of the scalar value of some data.

        If you run multiprocess training, keep in mind that
        the data is perhaps only available on chief process.

        Returns:
            a list of (global_step, value) pairs: history data for this scalar
        """
        return self._scalar_history.get_history(name)


class MaintainStepCounter(Callback):
    """
    It maintains the global step in the graph, making sure it's increased by one.
    This callback is used internally by the trainer, you don't need to worry about it.
    """

    _chief_only = False
    """
    In distributed training, we let each worker maintain its local global_step.
    """

    def _setup_graph(self):
        # ensure it exists
        gs_var = get_global_step_var()
        with tf.name_scope(None):
            self.gs_incr_op = tf.assign_add(
                gs_var, 1,
                name=GLOBAL_STEP_INCR_OP_NAME).op
        self._fetches = tf.train.SessionRunArgs(self.gs_incr_op)

    def _before_train(self):
        if self.global_step != 0:
            logger.info("Start training with global_step={}".format(self.global_step))

    def _before_run(self, _):
        # always increase global_step when hooked_sess.run is called
        return self._fetches

    def _after_run(self, _, __):
        # Keep python-side global_step in agreement with TF-side
        self.trainer._global_step += 1


class CallbackFactory(Callback):
    """
    Create a callback with some lambdas.
    """
    def __init__(self, setup_graph=None, before_train=None, trigger=None,
                 after_train=None):
        """
        Each lambda takes ``self`` as the only argument.
        """

        self._cb_setup_graph = setup_graph
        self._cb_before_train = before_train
        self._cb_trigger = trigger
        self._cb_after_train = after_train

    def _setup_graph(self):
        if self._cb_setup_graph:
            self._cb_setup_graph(self)

    def _before_train(self):
        if self._cb_before_train:
            self._cb_before_train(self)

    def _trigger(self):
        if self._cb_trigger:
            self._cb_trigger(self)

    def _after_train(self):
        if self._cb_after_train:
            self._cb_after_train(self)


class StartProcOrThread(Callback):
    """
    Start some threads or processes before training.
    """

    _chief_only = False

    def __init__(self, startable, stop_at_last=True):
        """
        Args:
            startable (list): list of processes or threads which have ``start()`` method.
                Can also be a single instance of process of thread.
            stop_at_last (bool): whether to stop the processes or threads
                after training. It will use :meth:`Process.terminate()` or
                :meth:`StoppableThread.stop()`, but will do nothing on normal
                `threading.Thread` or other startable objects.
        """
        if not isinstance(startable, list):
            startable = [startable]
        self._procs_threads = startable
        self._stop_at_last = stop_at_last

    def _before_train(self):
        logger.info("Starting " +
                    ', '.join([k.name for k in self._procs_threads]) + ' ...')
        # avoid sigint get handled by other processes
        start_proc_mask_signal(self._procs_threads)

    def _after_train(self):
        if not self._stop_at_last:
            return
        for k in self._procs_threads:
            if not k.is_alive():
                continue
            if isinstance(k, mp.Process):
                logger.info("Stopping {} ...".format(k.name))
                k.terminate()
                k.join(5.0)
                if k.is_alive():
                    logger.error("Cannot join process {}.".format(k.name))
            elif isinstance(k, StoppableThread):
                logger.info("Stopping {} ...".format(k.name))
                k.stop()
                k.join(5.0)
                if k.is_alive():
                    logger.error("Cannot join thread {}.".format(k.name))


class RunOp(Callback):
    """ Run an Op. """

    _chief_only = False

    def __init__(self, op,
                 run_before=True, run_as_trigger=True,
                 run_step=False, verbose=False):
        """
        Args:
            op (tf.Operation or function): an Op, or a function that returns the Op in the graph.
                The function will be called after the main graph has been created (in the `setup_graph` callback).
            run_before (bool): run the Op before training
            run_as_trigger (bool): run the Op on every :meth:`trigger()` call.
            run_step (bool): run the Op every step (along with training)
            verbose (bool): print logs when the op is run.

        Example:
            The `DQN Example
            <https://github.com/tensorpack/tensorpack/blob/master/examples/DeepQNetwork/>`_
            uses this callback to update target network.
        """
        if not callable(op):
            self.setup_func = lambda: op  # noqa
        else:
            self.setup_func = op
        self.run_before = run_before
        self.run_as_trigger = run_as_trigger
        self.run_step = run_step
        self.verbose = verbose

    def _setup_graph(self):
        self._op = self.setup_func()
        if self.run_step:
            self._fetch = tf.train.SessionRunArgs(fetches=self._op)

    def _before_train(self):
        if self.run_before:
            self._print()
            self._op.run()

    def _trigger(self):
        if self.run_as_trigger:
            self._print()
            self._op.run()

    def _before_run(self, _):
        if self.run_step:
            self._print()
            return self._fetch

    def _print(self):
        if self.verbose:
            logger.info("Running Op {} ...".format(self._op.name))


class JSONWriter(MonitorBase):
    """
    Write all scalar data to a json file under ``logger.get_logger_dir()``, grouped by their global step.
    If found an earlier json history file, will append to it.
    """

    FILENAME = 'stats.json'
    """
    The name of the json file. Do not change it.
    """

    def __new__(cls):
        if logger.get_logger_dir():
            return super(JSONWriter, cls).__new__(cls)
        else:
            logger.warn("logger directory was not set. Ignore JSONWriter.")
            return NoOpMonitor("JSONWriter")

    @staticmethod
    def load_existing_json():
        """
        Look for an existing json under :meth:`logger.get_logger_dir()` named "stats.json",
        and return the loaded list of statistics if found. Returns None otherwise.
        """
        dir = logger.get_logger_dir()
        fname = os.path.join(dir, JSONWriter.FILENAME)
        if tf.gfile.Exists(fname):
            with open(fname) as f:
                stats = json.load(f)
                assert isinstance(stats, list), type(stats)
                return stats
        return None

    @staticmethod
    def load_existing_epoch_number():
        """
        Try to load the latest epoch number from an existing json stats file (if any).
        Returns None if not found.
        """
        stats = JSONWriter.load_existing_json()
        try:
            return int(stats[-1]['epoch_num'])
        except Exception:
            return None

    # initialize the stats here, because before_train from other callbacks may use it
    def _setup_graph(self):
        self._stats = []
        self._stat_now = {}
        self._last_gs = -1

    def _before_train(self):
        stats = JSONWriter.load_existing_json()
        self._fname = os.path.join(logger.get_logger_dir(), JSONWriter.FILENAME)
        if stats is not None:
            try:
                epoch = stats[-1]['epoch_num'] + 1
            except Exception:
                epoch = None

            # check against the current training settings
            # therefore this logic needs to be in before_train stage
            starting_epoch = self.trainer.loop.starting_epoch
            if epoch is None or epoch == starting_epoch:
                logger.info("Found existing JSON inside {}, will append to it.".format(logger.get_logger_dir()))
                self._stats = stats
            else:
                logger.warn(
                    "History epoch={} from JSON is not the predecessor of the current starting_epoch={}".format(
                        epoch - 1, starting_epoch))
                logger.warn("If you want to resume old training, either use `AutoResumeTrainConfig` "
                            "or correctly set the new starting_epoch yourself to avoid inconsistency. ")

                backup_fname = JSONWriter.FILENAME + '.' + datetime.now().strftime('%m%d-%H%M%S')
                backup_fname = os.path.join(logger.get_logger_dir(), backup_fname)

                logger.warn("Now, we will train with starting_epoch={} and backup old json to {}".format(
                    self.trainer.loop.starting_epoch, backup_fname))
                shutil.move(self._fname, backup_fname)

        # in case we have something to log here.
        self._trigger()

    def _trigger_step(self):
        # will do this in trigger_epoch
        if self.local_step != self.trainer.steps_per_epoch - 1:
            self._trigger()

    def _trigger_epoch(self):
        self._trigger()

    @HIDE_DOC
    def process_scalar(self, name, val):
        self._stat_now[name] = val

    def _trigger(self):
        """
        Add stats to json and dump to disk.
        Note that this method is idempotent.
        """
        if len(self._stat_now):
            self._stat_now['epoch_num'] = self.epoch_num
            self._stat_now['global_step'] = self.global_step

            self._stats.append(self._stat_now)
            self._stat_now = {}
            self._write_stat()

    def _write_stat(self):
        tmp_filename = self._fname + '.tmp'
        try:
            with open(tmp_filename, 'w') as f:
                json.dump(self._stats, f)
            shutil.move(tmp_filename, self._fname)
        except IOError:  # disk error sometimes..
            logger.exception("Exception in JSONWriter._write_stat()!")


class MovingAverageSummary(Callback):
    """
    This callback is enabled by default.
    Maintain the moving average of summarized tensors in every step,
    by ops added to the collection.
    Note that it only __maintains__ the moving averages by updating
    the relevant variables in the graph,
    the actual summary should be done in other callbacks.
    """
    def __init__(self, collection=MOVING_SUMMARY_OPS_KEY, train_op=None):
        """
        Args:
            collection(str): the collection of EMA-maintaining ops.
                The default value would work with
                the tensors you added by :func:`tfutils.summary.add_moving_summary()`,
                but you can use other collections as well.
            train_op (tf.Operation or str): the (name of) training op to associate the maintaing ops with.
                If not provided, the EMA-maintaining ops will be hooked to
                `trainer.hooked_session` and be executed in every iteration.
                Otherwise, the EMA-maintaining ops will be executed whenever
                the training op is executed.
        """
        self._collection = collection
        self._train_op = train_op

    def _setup_graph(self):
        ops = [k.op for k in tf.get_collection(self._collection)]
        if self._train_op is None:
            logger.info("[MovingAverageSummary] {} operations in collection '{}' "
                        "will be run with session hooks.".format(len(ops), self._collection))

            self.ema_op = tf.group(*ops, name='maintain_moving_average_summary')
            self._fetch = tf.train.SessionRunArgs(fetches=self.ema_op)
        else:
            if isinstance(self._train_op, tf.Tensor):
                self._train_op = self._train_op.op
            if not isinstance(self._train_op, tf.Operation):
                self._train_op = self.graph.get_operation_by_name(self._train_op)
            self._train_op._add_control_inputs(ops)
            logger.info("[MovingAverageSummary] {} operations in collection '{}'"
                        " will be run together with operation '{}'.".format(
                            len(ops), self._collection, self._train_op.name))

    def _before_run(self, _):
        if self._train_op is None:
            return self._fetch


class MergeAllSummaries_RunAlone(Callback):
    def __init__(self, period, key):
        self._period = period
        self._key = key

    def _setup_graph(self):
        size = len(tf.get_collection(self._key))
        logger.info("Summarizing collection '{}' of size {}.".format(self._key, size))
        self.summary_op = tf.summary.merge_all(self._key)

    def _trigger_step(self):
        if self._period:
            if (self.local_step + 1) % self._period == 0:
                self._trigger()

    def _trigger(self):
        if self.summary_op:
            summary = self.summary_op.eval()
            self.trainer.monitors.put_summary(summary)

class MergeAllSummaries_RunWithOp(Callback):
    def __init__(self, period, key):
        self._period = period
        self._key = key

    def _setup_graph(self):
        size = len(tf.get_collection(self._key))
        logger.info("Summarizing collection '{}' of size {}.".format(self._key, size))
        self.summary_op = tf.summary.merge_all(self._key)
        if self.summary_op is not None:
            self._fetches = tf.train.SessionRunArgs(self.summary_op)
        else:
            self._fetches = None

    def _need_run(self):
        if self.local_step == self.trainer.steps_per_epoch - 1:
            return True
        if self._period > 0 and (self.local_step + 1) % self._period == 0:
            return True
        return False

    def _before_run(self, ctx):
        if self._need_run():
            return self._fetches
        return None

    def _after_run(self, _, run_values):
        summary = run_values.results
        if summary is None:
            return
        self.trainer.monitors.put_summary(summary)

def MergeAllSummaries(period=0, run_alone=False, key=None):
    """
    This callback is enabled by default.
    Evaluate all summaries by `tf.summary.merge_all`, and write them to logs.

    Args:
        period (int): by default the callback summarizes once every epoch.
            This option (if not set to 0) makes it additionally summarize every ``period`` steps.
        run_alone (bool): whether to evaluate the summaries alone.
            If True, summaries will be evaluated after each epoch alone.
            If False, summaries will be evaluated together with the
            `sess.run` calls, in the last step of each epoch.
            For :class:`SimpleTrainer`, it needs to be False because summary may
            depend on inputs.
        key (str): the collection of summary tensors. Same as in `tf.summary.merge_all`.
            Default is ``tf.GraphKeys.SUMMARIES``.
    """
    if key is None:
        key = tf.GraphKeys.SUMMARIES
    period = int(period)
    if run_alone:
        return MergeAllSummaries_RunAlone(period, key)
    else:
        return MergeAllSummaries_RunWithOp(period, key)


class ProgressBar(Callback):
    """ A progress bar based on tqdm. Enabled by default. """

    _chief_only = False

    def __init__(self, names=[]):
        """
        Args:
            names(list): list of string, the names of the tensors to monitor
                on the progress bar.
        """
        super(ProgressBar, self).__init__()
        self._names = [get_op_tensor_name(n)[1] for n in names]
        self._tags = [get_op_tensor_name(n)[0].split("/")[-1] for n in names]
        self._bar = None

    def _before_train(self):
        self._last_updated = self.local_step

        self._total = self.trainer.steps_per_epoch
        self._tqdm_args = get_tqdm_kwargs(leave=True)

        if self._names:
            self._fetches = self.get_tensors_maybe_in_tower(self._names)
        else:
            self._fetches =  None
        if self._fetches:
            for t in self._fetches:
                assert t.shape.ndims == 0, "ProgressBar can only print scalars, not {}".format(t)
            self._fetches = tf.train.SessionRunArgs(self._fetches)
            self._tqdm_args['bar_format'] = self._tqdm_args['bar_format'] + "{postfix} "

    def _before_epoch(self):
        self._bar = tqdm.trange(self._total, **self._tqdm_args)

    def _after_epoch(self):
        self._bar.close()

    def _before_run(self, _):
        # update progress bar when local step changed (one step is finished)
        if self.local_step != self._last_updated:
            self._last_updated = self.local_step
            return self._fetches
        else:
            return None

    def _after_run(self, _, run_values):
        res = run_values.results
        if res:
            self._bar.set_postfix(zip(self._tags, res))

    def _trigger_step(self):
        self._bar.update()

    def _after_train(self):
        if self._bar:       # training may get killed before the first step
            self._bar.close()


class RunOp(Callback):
    """ Run an Op. """

    _chief_only = False

    def __init__(self, op,
                 run_before=True, run_as_trigger=True,
                 run_step=False, verbose=False):
        """
        Args:
            op (tf.Operation or function): an Op, or a function that returns the Op in the graph.
                The function will be called after the main graph has been created (in the `setup_graph` callback).
            run_before (bool): run the Op before training
            run_as_trigger (bool): run the Op on every :meth:`trigger()` call.
            run_step (bool): run the Op every step (along with training)
            verbose (bool): print logs when the op is run.

        Example:
            The `DQN Example
            <https://github.com/tensorpack/tensorpack/blob/master/examples/DeepQNetwork/>`_
            uses this callback to update target network.
        """
        if not callable(op):
            self.setup_func = lambda: op  # noqa
        else:
            self.setup_func = op
        self.run_before = run_before
        self.run_as_trigger = run_as_trigger
        self.run_step = run_step
        self.verbose = verbose

    def _setup_graph(self):
        self._op = self.setup_func()
        if self.run_step:
            self._fetch = tf.train.SessionRunArgs(fetches=self._op)

    def _before_train(self):
        if self.run_before:
            self._print()
            self._op.run()

    def _trigger(self):
        if self.run_as_trigger:
            self._print()
            self._op.run()

    def _before_run(self, _):
        if self.run_step:
            self._print()
            return self._fetch

    def _print(self):
        if self.verbose:
            logger.info("Running Op {} ...".format(self._op.name))

class RunUpdateOps(RunOp):
    """
    Run ops from the collection UPDATE_OPS every step.
    The ops will be hooked to `trainer.hooked_sess` and run along with
    each `sess.run` call.
    """

    def __init__(self, collection=None):
        """
        Args:
            collection (str): collection of ops to run. Defaults to ``tf.GraphKeys.UPDATE_OPS``
        """
        if collection is None:
            collection = tf.GraphKeys.UPDATE_OPS
        name = 'UPDATE_OPS' if collection == tf.GraphKeys.UPDATE_OPS else collection

        def f():
            ops = tf.get_collection(collection)
            if ops:
                logger.info("Applying collection {} of {} ops.".format(name, len(ops)))
                return tf.group(*ops, name='update_ops')
            else:
                return tf.no_op(name='empty_update_ops')

        super(RunUpdateOps, self).__init__(
            f, run_before=False, run_as_trigger=False, run_step=True)
