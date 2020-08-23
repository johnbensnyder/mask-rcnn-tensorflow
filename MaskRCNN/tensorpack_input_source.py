from abc import ABCMeta, abstractmethod
import six
from contextlib import contextmanager
from tensorpack_callbacks import CallbackFactory, RunOp
from tensorpack_dataflow import RepeatedData
from tensorpack_utils import ShareSessionThread, call_only_once, memoized_method
from tensorpack_tfutils import add_moving_summary
import tensorpack_logger as logger
import threading
import tensorflow.compat.v1 as tf

def _get_reset_callback(df):
    return CallbackFactory(setup_graph=lambda _: df.reset_state())

@six.add_metaclass(ABCMeta)
class InputSource(object):
    """ Base class for the abstract InputSource. """

    _name_scope = None
    _setup_done = False

    def get_input_tensors(self):
        """
        Returns:
            list[Tensor]: A list of tensors corresponding to the inputs of the model.
                Will be used as input for the tower function.
                This method should always create and return new tensors when called,
                unless it returns placeholders.
        """
        return self._get_input_tensors()

    @abstractmethod
    def _get_input_tensors(self):
        pass

    @call_only_once
    def setup(self, inputs_desc):
        """
        Args:
            inputs_desc (list[InputDesc]): list of input desc

        Returns:
            list[Callback]: extra callbacks needed by this InputSource.
            callbacks of InputSource cannot use any `trigger*()` method.
        """
        self._setup(inputs_desc)
        self._setup_done = True
        return self.get_callbacks()

    def _setup(self, inputs_desc):
        pass

    def setup_done(self):
        """
        Returns:
            bool: whether :meth:`setup()` has been called.
        """
        return self._setup_done

    @memoized_method
    def get_callbacks(self):
        """
        An InputSource might need some extra maintenance during training,
        which is done also through the Callback interface.
        This method returns the callbacks and the return value will be memoized.

        All callbacks will be automatically marked as `chief_only=False`,
        so they will run on all nodes.

        Callbacks returned by :class:`InputSource` only supports a subset of callback's functionalities:

        1. It cannot access the trainer, because an :class:`InputSource` can be used in pure inference.
        2. It cannot use the following methods: `trigger_{step,epoch}, {before,after}_epoch`.

        In other words, these callbacks should only have the basic functionality of `tf.train.SessionRunHooks`.

        Returns:
            list[Callback]: extra callbacks needed by this InputSource.
        """
        assert self.setup_done()
        ret = [CallbackFactory(
            before_train=lambda _: self.reset_state())] + self._get_callbacks()

        for r in ret:
            r.set_chief_only(False)    # no input callbacks should be chief-only
        return ret

    def _get_callbacks(self):
        return []

    def reset_state(self):
        """
        Initialize/reinitialize this InputSource.
        Must be called under a default session.

        For training, it will get called once by the trainer in `before_train` callbacks.
        For inference, the :class:`InferenceRunner` will call this method each time it is triggered.
        """
        self._reset_state()

    def _reset_state(self):
        pass

    def size(self):
        """
        Returns:
            int: epoch size of the InputSource
        """
        return self._size()

    def _size(self):
        raise NotImplementedError()

    @contextmanager
    def cached_name_scope(self):
        """
        Yield a context under a cached name scope, whose name is the name of
        this InputSource class.
        """
        if self._name_scope:
            with tf.name_scope(self._name_scope):
                yield self._name_scope
        else:
            name = type(self).__name__
            with tf.name_scope(name) as ns:
                self._name_scope = ns
                yield ns


class FeedfreeInput(InputSource):
    """ Abstract base for input without feed,
    e.g. by queue or other operations. """

    def _reset_state(self):
        pass


def _make_feeds(placeholders, datapoint):
    assert len(datapoint) == len(placeholders), \
        "Size of datapoint and placeholders are different: {} != {}".format(
            len(datapoint), len(placeholders))

    if isinstance(datapoint, (list, tuple)):
        return dict(zip(placeholders, datapoint))
    elif isinstance(datapoint, dict):
        ret = {p: datapoint[p.op.name] for p in placeholders}
        return ret
    else:
        raise TypeError("Got a datapoint of type {}!".format(type(datapoint)))


class EnqueueThread(ShareSessionThread):
    def __init__(self, queue, ds, placehdrs):
        super(EnqueueThread, self).__init__()
        self.name = 'EnqueueThread ' + queue.name
        self.daemon = True
        self.dataflow = ds
        self.queue = queue
        self.placehdrs = placehdrs

        self.op = self.queue.enqueue(self.placehdrs)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._running = threading.Event()
        self._running.set()
        # self._size = queue.size()

    def run(self):
        with self.default_sess():
            try:
                self.reinitialize_dataflow()
                while True:
                    # pausable loop
                    if not self._running.is_set():
                        self._running.wait()

                    dp = next(self._itr)
                    feed = _make_feeds(self.placehdrs, dp)
                    # _, sz = sess.run([self.op, self._sz], feed_dict=feed)
                    self.op.run(feed_dict=feed)
            except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
                pass
                # logger.exception("Exception in {}:".format(self.name))
            except Exception as e:
                if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                    pass
                else:
                    logger.exception("Exception in {}:".format(self.name))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logger.info("{} Exited.".format(self.name))

    def reinitialize_dataflow(self):
        self._itr = self.dataflow.__iter__()

    def pause(self):
        self._running.clear()

    def resume(self):
        self._running.set()


class QueueInput(FeedfreeInput):
    """ Enqueue datapoints from a DataFlow to a TF queue.
        And the model receives dequeued tensors.
    """

    def __init__(self, ds, queue=None):
        """
        Args:
            ds(DataFlow): the input DataFlow.
            queue (tf.QueueBase): A :class:`tf.QueueBase` whose type
                should match the corresponding InputDesc of the model.
                Defaults to a FIFO queue of size 50.
        """
        #if not isinstance(ds, DataFlow):
        #    raise ValueError("QueueInput takes a DataFlow! Got {}".format(ds))
        self.queue = queue
        self.ds = ds
        self._inf_ds = RepeatedData(ds, -1)
        self._started = False

    def _size(self):
        return len(self.ds)

    def _setup(self, inputs):
        self._input_placehdrs = [v.build_placeholder_reuse() for v in inputs]
        assert len(self._input_placehdrs) > 0, \
            "QueueInput has to be used with some inputs!"
        with self.cached_name_scope():
            if self.queue is None:
                self.queue = tf.FIFOQueue(
                    50, [x.dtype for x in self._input_placehdrs],
                    name='input_queue')
            logger.info("Setting up the queue '{}' for CPU prefetching ...".format(self.queue.name))
            self.thread = EnqueueThread(self.queue, self._inf_ds, self._input_placehdrs)

            self._dequeue_op = self.queue.dequeue(name='dequeue_for_reset')

    def refill_queue(self):
        """
        Clear the queue, then call dataflow.__iter__() again and fill into the queue.
        """
        self.thread.pause()     # pause enqueue

        opt = tf.RunOptions()
        # trying increasing from 2 to 50 for xla
        opt.timeout_in_ms = 2000   # 2s
        sess = tf.get_default_session()
        # dequeue until empty
        try:
            while True:
                sess.run(self._dequeue_op, options=opt)
        except tf.errors.DeadlineExceededError:
            pass

        # reset dataflow, start thread
        self.thread.reinitialize_dataflow()
        self.thread.resume()

    def _create_ema_callback(self):
        """
        Create a hook-only callback which maintain EMA of the queue size.
        Also tf.summary.scalar the EMA.
        """
        with self.cached_name_scope():
            # in TF there is no API to get queue capacity, so we can only summary the size
            size = tf.cast(self.queue.size(), tf.float32, name='queue_size')
        size_ema_op = add_moving_summary(size, collection=None, decay=0.5)[0].op
        return RunOp(
            lambda: size_ema_op,
            run_before=False,
            run_as_trigger=False,
            run_step=True)

    def _get_callbacks(self):
        #from ..callbacks.concurrency import StartProcOrThread
        from tensorpack_callbacks import StartProcOrThread
        cb = StartProcOrThread(self.thread)
        return [cb, self._create_ema_callback(), _get_reset_callback(self._inf_ds)]

    def _get_input_tensors(self):
        with tf.device('/cpu:0'), self.cached_name_scope():
            ret = self.queue.dequeue(name='input_deque')
            if isinstance(ret, tf.Tensor):  # only one input
                ret = [ret]
            assert len(ret) == len(self._input_placehdrs)
            for qv, v in zip(ret, self._input_placehdrs):
                qv.set_shape(v.get_shape())
            return ret
