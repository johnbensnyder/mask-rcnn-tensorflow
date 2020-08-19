import tensorflow.compat.v1 as tf
from tensorpack_callbacks import (
    JSONWriter, MergeAllSummaries, MovingAverageSummary, ProgressBar, RunUpdateOps, ScalarPrinter, TFEventWriter)


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


class NewSessionCreator(tf.train.SessionCreator):
    def __init__(self, target='', config=None):
        """
        Args:
            target, config: same as :meth:`Session.__init__()`.
            config: a :class:`tf.ConfigProto` instance, defaults to :func:`tfutils.get_default_sess_config()`
        """
        self.target = target

        if config is None:
            # distributed trainer doesn't support user-provided config
            # we set this attribute so that they can check
            self.user_provided_config = False
            config = get_default_sess_config()
        else:
            self.user_provided_config = True
            logger.warn(
                "User-provided custom session config may not work due to TF \
bugs. See https://github.com/tensorpack/tensorpack/issues/497 for workarounds.")
        self.config = config

    def create_session(self):
        sess = tf.Session(target=self.target, config=self.config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        return sess

class TrainConfig(object):
    """
    A collection of options to be used for single-cost trainers.

    Note that you do not have to use :class:`TrainConfig`.
    You can use the API of :class:`Trainer` directly, to have more fine-grained control of the training.
    """

    def __init__(self,
                 dataflow=None, data=None,
                 model=None,
                 callbacks=None, extra_callbacks=None, monitors=None,
                 session_creator=None, session_config=None, session_init=None,
                 starting_epoch=1, steps_per_epoch=None, max_epoch=99999,
                 **kwargs):
        """
        Args:
            dataflow (DataFlow):
            data (InputSource):
            model (ModelDesc):

            callbacks (list[Callback]): a list of :class:`Callback` to use during training.
            extra_callbacks (list[Callback]): This argument
                is only used to provide the defaults in addition to ``callbacks``.
                The list of callbacks that will be used in the end is simply ``callbacks + extra_callbacks``.

                It is usually left as None, and the default value for this argument is :func:`DEFAULT_CALLBACKS()`.
                You can override it when you don't like any of the default callbacks.
                For example, if you'd like to let the progress bar print tensors, you can use

                .. code-block:: none

                    extra_callbacks=[ProgressBar(names=['name']),
                                     MovingAverageSummary(),
                                     MergeAllSummaries(),
                                     RunUpdateOps()]

            monitors (list[MonitorBase]): Defaults to :func:`DEFAULT_MONITORS()`.

            session_creator (tf.train.SessionCreator): Defaults to :class:`sesscreate.NewSessionCreator()`
                with the config returned by :func:`tfutils.get_default_sess_config()`.
            session_config (tf.ConfigProto): when session_creator is None, use this to create the session.
            session_init (SessionInit): how to initialize variables of a session. Defaults to do nothing.

            starting_epoch (int): The index of the first epoch.
            steps_per_epoch (int): the number of steps (defined by :meth:`Trainer.run_step`) to run in each epoch.
                Defaults to the input data size.
            max_epoch (int): maximum number of epoch to run training.
        """

        # TODO type checker decorator
        def assert_type(v, tp, name):
            assert isinstance(v, tp), \
                "{} has to be type '{}', but an object of type '{}' found.".format(
                    name, tp.__name__, v.__class__.__name__)

        # process data & model
        assert data is None or dataflow is None, "dataflow and data cannot be both presented in TrainConfig!"
        #if dataflow is not None:
        #    assert_type(dataflow, DataFlow, 'dataflow')
        #if data is not None:
        #    assert_type(data, InputSource, 'data')
        self.dataflow = dataflow
        self.data = data

        #if model is not None:
        #    assert_type(model, ModelDescBase, 'model')
        self.model = model

        #if callbacks is not None:
        #    assert_type(callbacks, list, 'callbacks')
        self.callbacks = callbacks
        #if extra_callbacks is not None:
        #    assert_type(extra_callbacks, list, 'extra_callbacks')
        self.extra_callbacks = extra_callbacks
        #if monitors is not None:
        #    assert_type(monitors, list, 'monitors')
        self.monitors = monitors
        #if session_init is not None:
        #    assert_type(session_init, SessionInit, 'session_init')
        self.session_init = session_init

        if session_creator is None:
            if session_config is not None:
                self.session_creator = NewSessionCreator(config=session_config)
            else:
                self.session_creator = NewSessionCreator(config=None)
        else:
            self.session_creator = session_creator
            assert session_config is None, "Cannot set both session_creator and session_config!"

        if steps_per_epoch is None:
            try:
                if dataflow is not None:
                    steps_per_epoch = len(dataflow)
                elif data is not None:
                    steps_per_epoch = data.size()
                else:
                    raise NotImplementedError()
            except NotImplementedError:
                logger.error("You must set `TrainConfig(steps_per_epoch)` if the size of your input is not available.")
                raise
        else:
            steps_per_epoch = int(steps_per_epoch)
        self.steps_per_epoch = steps_per_epoch

        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)

        if 'nr_tower' in kwargs:
            self.nr_tower = kwargs.pop('nr_tower')
        if 'tower' in kwargs:
            self.tower = kwargs.pop('tower')
        else:
            self.tower = [0]
        assert len(kwargs) == 0, "Unknown arguments: {}".format(kwargs.keys())


def DEFAULT_MONITORS():
    """
    Return the default monitors,
    which will be used in :class:`TrainConfig` and :meth:`Trainer.train_with_defaults`.
    They are:

    1. TFEventWriter()
    2. JSONWriter()
    3. ScalarPrinter()
    """
    return [TFEventWriter(), JSONWriter(), ScalarPrinter()]

def DEFAULT_CALLBACKS():
    """
    Return the default callbacks,
    which will be used in :class:`TrainConfig` and :meth:`Trainer.train_with_defaults`.
    They are:

    1. MovingAverageSummary()
    2. ProgressBar()
    3. MergeAllSummaries()
    4. RunUpdateOps()
    """
    return [
        MovingAverageSummary(),
        ProgressBar(),
        MergeAllSummaries(),
        RunUpdateOps()]
