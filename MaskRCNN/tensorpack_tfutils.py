import tensorflow.compat.v1 as tf
from six.moves import map
from collections import defaultdict
from contextlib import contextmanager
from copy import copy
import six
import functools
import re
import cv2
import os
import numpy as np
from abc import ABCMeta, abstractmethod

from tensorflow.python.training import moving_averages

from tensorpack_utils import graph_memoized, get_data_format, shape2d

import tensorpack_logger as logger

def gpu_available_in_session():
    sess = tf.get_default_session()
    for dev in sess.list_devices():
        if dev.device_type.lower() == 'gpu':
            return True
    return False

def get_tf_version_tuple():
    """
    Return TensorFlow version as a 2-element tuple (for comparison).
    """
    return tuple(map(int, tf.__version__.split('.')[:2]))

def auto_reuse_variable_scope(func):
    """
    A decorator which automatically reuses the current variable scope if the
    function has been called with the same variable scope before.

    Example:

    .. code-block:: python

        @auto_reuse_variable_scope
        def myfunc(x):
            return tf.layers.conv2d(x, 128, 3)

        myfunc(x1)  # will inherit parent scope reuse
        myfunc(x2)  # will reuse
        with tf.variable_scope('newscope'):
            myfunc(x3)  # will inherit parent scope reuse
            myfunc(x4)  # will reuse
    """
    used_scope = set()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        scope = tf.get_variable_scope()
        h = hash((tf.get_default_graph(), scope.name))
        # print("Entering " + scope.name + " reuse: " + str(h in used_scope))
        if h in used_scope:
            if get_tf_version_tuple() >= (1, 5):
                with tf.variable_scope(scope, reuse=True, auxiliary_name_scope=False):
                    return func(*args, **kwargs)
            else:
                ns = tf.get_default_graph().get_name_scope()
                with tf.variable_scope(scope, reuse=True), \
                        tf.name_scope(ns + '/' if ns else ''):
                    return func(*args, **kwargs)
        else:
            used_scope.add(h)
            return func(*args, **kwargs)

    return wrapper

def under_name_scope(name_scope=None):
    """
    Args:
        name_scope(str): the default scope to use. If None, will use the name of the function.

    Returns:
        A decorator which makes the function run under a name scope.
        The name scope is obtained by the following:
        1. The 'name_scope' keyword argument when the decorated function is called.
        2. The 'name_scope' argument of the decorator.
        3. (default) The name of the decorated function itself.

    Example:

    .. code-block:: python

        @under_name_scope()
        def rms(x):
            return tf.sqrt(
                tf.reduce_mean(tf.square(x)))

        rms(tensor)  # will be called under name scope 'rms'
        rms(tensor, name_scope='scope')  # will be called under name scope 'scope'


    Todo:
        Add a reuse option.
    """

    def _impl(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            scopename = kwargs.pop('name_scope', name_scope)
            if scopename is None:
                scopename = func.__name__

            with tf.name_scope(scopename):
                return func(*args, **kwargs)
        return wrapper
    return _impl

@contextmanager
def custom_getter_scope(custom_getter):
    """
    Args:
        custom_getter: the same as in :func:`tf.get_variable`

    Returns:
        The current variable scope with a custom_getter.
    """
    scope = tf.get_variable_scope()
    if get_tf_version_tuple() >= (1, 5):
        with tf.variable_scope(
                scope, custom_getter=custom_getter,
                auxiliary_name_scope=False):
            yield
    else:
        ns = tf.get_default_graph().get_name_scope()
        with tf.variable_scope(
                scope, custom_getter=custom_getter):
            with tf.name_scope(ns + '/' if ns else ''):
                yield

def freeze_variables(stop_gradient=True, skip_collection=False):
    """
    Return a context to freeze variables,
    by wrapping ``tf.get_variable`` with a custom getter.
    It works by either applying ``tf.stop_gradient`` on the variables,
    or by keeping them out of the ``TRAINABLE_VARIABLES`` collection, or
    both.

    Example:
        .. code-block:: python

            with varreplace.freeze_variable(stop_gradient=False, skip_collection=True):
                x = FullyConnected('fc', x, 1000)   # fc/* will not be trained

    Args:
        stop_gradient (bool): if True, variables returned from `get_variable`
            will be wrapped with `tf.stop_gradient` and therefore has no
            gradient when used later.
            Note that the created variables may still have gradient when accessed
            by other approaches (e.g. by name, or by collection).
            Also note that this makes `tf.get_variable` returns a Tensor instead of a Variable,
            which may break existing code.
            Therefore, it's recommended to use the `skip_collection` option instead.
        skip_collection (bool): if True, do not add the variable to
            ``TRAINABLE_VARIABLES`` collection, but to ``MODEL_VARIABLES``
            collection. As a result they will not be trained by default.
    """
    def custom_getter(getter, *args, **kwargs):
        trainable = kwargs.get('trainable', True)
        name = args[0] if len(args) else kwargs.get('name')
        if skip_collection:
            kwargs['trainable'] = False
        v = getter(*args, **kwargs)
        if skip_collection:
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)
        if trainable and stop_gradient:
            v = tf.stop_gradient(v, name='freezed_' + name)
        return v
    return custom_getter_scope(custom_getter)

_ArgScopeStack = []

@contextmanager
def argscope(layers, **kwargs):
    """
    Args:
        layers (list or layer): layer or list of layers to apply the arguments.

    Returns:
        a context where all appearance of these layer will by default have the
        arguments specified by kwargs.

    Example:
        .. code-block:: python

            with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):
                x = Conv2D('conv0', x)
                x = Conv2D('conv1', x)
                x = Conv2D('conv2', x, out_channel=64)  # override argscope

    """
    if not isinstance(layers, list):
        layers = [layers]

    # def _check_args_exist(l):
    #     args = inspect.getargspec(l).args
    #     for k, v in six.iteritems(kwargs):
    #         assert k in args, "No argument {} in {}".format(k, l.__name__)

    for l in layers:
        assert hasattr(l, 'symbolic_function'), "{} is not a registered layer".format(l.__name__)
        # _check_args_exist(l.symbolic_function)

    new_scope = copy(get_arg_scope())
    for l in layers:
        new_scope[l.__name__].update(kwargs)
    _ArgScopeStack.append(new_scope)
    yield
    del _ArgScopeStack[-1]

def get_arg_scope():
    """
    Returns:
        dict: the current argscope.

    An argscope is a dict of dict: ``dict[layername] = {arg: val}``
    """
    if len(_ArgScopeStack) > 0:
        return _ArgScopeStack[-1]
    else:
        return defaultdict(dict)

def get_shape_str(tensors):
    """
    Internally used by layer registry, to print shapes of inputs/outputs of layers.

    Args:
        tensors (list or tf.Tensor): a tensor or a list of tensors
    Returns:
        str: a string to describe the shape
    """
    if isinstance(tensors, (list, tuple)):
        for v in tensors:
            assert isinstance(v, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(v))
        shape_str = ",".join(
            map(lambda x: str(x.get_shape().as_list()), tensors))
    else:
        assert isinstance(tensors, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(tensors))
        shape_str = str(tensors.get_shape().as_list())
    return shape_str

_CurrentTowerContext = None

class TrainContext(object):
    """
    Create the context to reuse the variables from training graph and add namescope to the tensors in predict graph
    """
    def __init__(self, ns_name, vs_name='', is_training=True):
        self._name = ns_name
        self._vs_name = vs_name
        self.is_training = is_training
        self.is_main_training_tower = is_training

        if len(vs_name):
            assert len(ns_name), "vs_name cannot be used with an empty name!"

    @property
    def has_own_variables(self):
        if self.is_training:
            return True
        else:
            return not tf.get_variable_scope().reuse

    @property
    def vs_name(self):
        return self._vs_name

    def _get_scopes(self):
        """
        Returns the ns and vs for this tower.
        """
        if not len(self._name):
            # work around https://github.com/tensorflow/tensorflow/issues/14703
            return [tf.variable_scope(tf.get_variable_scope())]

        ret = []

        if len(self._vs_name):
            ret.append(tf.variable_scope(self._vs_name))
        else:
            # caller should have handled reuse outside of TowerContext
            ret.append(tf.variable_scope(tf.get_variable_scope()))

        # always clear existing ns  # TODO check existing ns
        if len(self._name):
            ret.append(tf.name_scope(self._name + '/'))
        return ret

    def __enter__(self):
        global _CurrentTowerContext
        _CurrentTowerContext = self
        self._ctxs = self._get_scopes()
        for c in self._ctxs:
            c.__enter__()

        # check that ns_name is always the same as _name
        ns = tf.get_default_graph().get_name_scope()
        assert ns == self._name, \
            "Name conflict: name_scope inside tower '{}' becomes '{}'!".format(self._name, ns) \
            + " You may need a different name for the tower!"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CurrentTowerContext
        _CurrentTowerContext = None
        for c in self._ctxs[::-1]:
            c.__exit__(exc_type, exc_val, exc_tb)
        return False

    def __str__(self):
        return "TowerContext(name={}, is_training={})".format(
            self._name, "True" if self.is_training else "False")

def get_current_tower_context():
    """
    When called inside a TowerContext, returns the TowerContext.

    Returns:
        a :class:`BaseTowerContext` instance or None, if not called under a TowerContext.
    """
    return _CurrentTowerContext


def backup_collection(keys=None):
    """
    Args:
        keys (list): list of collection keys to backup.
            Defaults to all keys in the graph.

    Returns:
        dict: the backup
    """
    if keys is None:
        keys = tf.get_default_graph().get_all_collection_keys()
    ret = {}
    assert isinstance(keys, (list, tuple, set))
    for k in keys:
        ret[k] = copy(tf.get_collection(k))
    return ret


def restore_collection(backup):
    """
    Restore from a collection backup.

    Args:
        backup (dict):
    """
    for k, v in six.iteritems(backup):
        del tf.get_collection_ref(k)[:]
        tf.get_collection_ref(k).extend(v)


MOVING_SUMMARY_OPS_KEY = 'MOVING_SUMMARY_OPS'
# some scope stuff to use internally...
@graph_memoized
def _get_cached_vs(name):
    with tf.variable_scope(name) as scope:
        return scope


@contextmanager
def _enter_vs_reuse_ns(name):
    vs = _get_cached_vs(name)
    # XXX Not good to enter the cached vs directly, because this will clean-up custom getter
    # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):    # available in 1.4 only
    with tf.variable_scope(vs):
        with tf.name_scope(vs.original_name_scope):
            yield vs

def add_moving_summary(*args, **kwargs):
    """
    Summarize the moving average for scalar tensors.
    This function is a no-op if not calling from main training tower.

    Args:
        args: scalar tensors to summarize
        decay (float): the decay rate. Defaults to 0.95.
        collection (str or None): the name of the collection to add EMA-maintaining ops.
            The default will work together with the default
            :class:`MovingAverageSummary` callback.
        summary_collections ([str]): the names of collections to add the
            summary op. Default is TF's default (`tf.GraphKeys.SUMMARIES`).

    Returns:
        [tf.Tensor]: list of tensors returned by assign_moving_average,
            which can be used to maintain the EMA.
    """
    decay = kwargs.pop('decay', 0.95)
    coll = kwargs.pop('collection', MOVING_SUMMARY_OPS_KEY)
    summ_coll = kwargs.pop('summary_collections', None)
    assert len(kwargs) == 0, "Unknown arguments: " + str(kwargs)

    ctx = get_current_tower_context()
    # allow ctx to be none
    if ctx is not None and not ctx.is_main_training_tower:
        return []

    graph = tf.get_default_graph()
    try:
        control_flow_ctx = graph._get_control_flow_context()
        # XLA does not support summaries anyway
        # However, this function will generate unnecessary dependency edges,
        # which makes the tower function harder to compile under XLA, so we skip it
        if control_flow_ctx is not None and control_flow_ctx.IsXLAContext():
            return
    except Exception:
        pass

    if tf.get_variable_scope().reuse is True:
        logger.warn("add_moving_summary() called under reuse=True scope, ignored.")
        return []

    for x in args:
        assert isinstance(x, (tf.Tensor, tf.Variable)), x
        assert x.get_shape().ndims == 0, \
            "add_moving_summary() only accepts scalar tensor! Got one with {}".format(x.get_shape())

    ema_ops = []
    for c in args:
        name = re.sub('tower[0-9]+/', '', c.op.name)
        with tf.name_scope(None):
            if not c.dtype.is_floating:
                c = tf.cast(c, tf.float32)
            # assign_moving_average creates variables with op names, therefore clear ns first.
            with _enter_vs_reuse_ns('EMA') as vs:
                ema_var = tf.get_variable(name, shape=c.shape, dtype=c.dtype,
                                          initializer=tf.constant_initializer(),
                                          trainable=False)
                ns = vs.original_name_scope
            with tf.name_scope(ns):     # reuse VS&NS so that EMA_1 won't appear
                ema_op = moving_averages.assign_moving_average(
                    ema_var, c, decay,
                    zero_debias=True, name=name + '_EMA_apply')
            ema_ops.append(ema_op)
        with tf.name_scope(None):
            tf.summary.scalar(
                name + '-summary', ema_op,
                collections=summ_coll)    # write the EMA value as a summary
    if coll is not None:
        for op in ema_ops:
            tf.add_to_collection(coll, op)
    return ema_ops

def create_scalar_summary(name, v):
    """
    Args:
        name (str):
        v (float): scalar value
    Returns:
        tf.Summary: a tf.Summary object with name and simple scalar value v.
    """
    assert isinstance(name, six.string_types), type(name)
    v = float(v)
    s = tf.Summary()
    s.value.add(tag=name, simple_value=v)
    return s

def create_image_summary(name, val):
    """
    Args:
        name(str):
        val(np.ndarray): 4D tensor of NHWC. assume RGB if C==3.
            Can be either float or uint8. Range has to be [0,255].

    Returns:
        tf.Summary:
    """
    assert isinstance(name, six.string_types), type(name)
    n, h, w, c = val.shape
    val = val.astype('uint8')
    s = tf.Summary()
    imparams = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    for k in range(n):
        arr = val[k]
        # CV2 will only write correctly in BGR chanel order
        if c == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif c == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        tag = name if n == 1 else '{}/{}'.format(name, k)
        retval, img_str = cv2.imencode('.png', arr, imparams)
        if not retval:
            # Encoding has failed.
            continue
        img_str = img_str.tostring()

        img = tf.Summary.Image()
        img.height = h
        img.width = w
        # 1 - grayscale 3 - RGB 4 - RGBA
        img.colorspace = c
        img.encoded_image_string = img_str
        s.value.add(tag=tag, image=img)
    return s


def DynamicLazyAxis(shape, idx):
    return lambda: shape[idx]

def StaticLazyAxis(dim):
    return lambda: dim

class StaticDynamicShape(object):
    def __init__(self, tensor):
        assert isinstance(tensor, tf.Tensor), tensor
        ndims = tensor.shape.ndims
        self.static = tensor.shape.as_list()
        if tensor.shape.is_fully_defined():
            self.dynamic = self.static[:]
        else:
            dynamic = tf.shape(tensor)
            self.dynamic = [DynamicLazyAxis(dynamic, k) for k in range(ndims)]

        for k in range(ndims):
            if self.static[k] is not None:
                self.dynamic[k] = StaticLazyAxis(self.static[k])

    def apply(self, axis, f):
        if self.static[axis] is not None:
            try:
                st = f(self.static[axis])
                self.static[axis] = st
                self.dynamic[axis] = StaticLazyAxis(st)
                return
            except TypeError:
                pass
        self.static[axis] = None
        dyn = self.dynamic[axis]
        self.dynamic[axis] = lambda: f(dyn())

    def get_static(self):
        return self.static

    @property
    def ndims(self):
        return len(self.static)

    def get_dynamic(self, axis=None):
        if axis is None:
            return [self.dynamic[k]() for k in range(self.ndims)]
        return self.dynamic[axis]()


def get_op_tensor_name(name):
    """
    Will automatically determine if ``name`` is a tensor name (ends with ':x')
    or a op name.
    If it is an op name, the corresponding tensor name is assumed to be ``op_name + ':0'``.

    Args:
        name(str): name of an op or a tensor
    Returns:
        tuple: (op_name, tensor_name)
    """
    if len(name) >= 3 and name[-2] == ':':
        return name[:-2], name
    else:
        return name, name + ':0'


def get_op_or_tensor_by_name(name):
    """
    Get either tf.Operation of tf.Tensor from names.

    Args:
        name (list[str] or str): names of operations or tensors.

    Raises:
        KeyError, if the name doesn't exist
    """
    G = tf.get_default_graph()

    def f(n):
        if len(n) >= 3 and n[-2] == ':':
            return G.get_tensor_by_name(n)
        else:
            return G.get_operation_by_name(n)

    if not isinstance(name, list):
        return f(name)
    else:
        return list(map(f, name))


class MismatchLogger(object):
    def __init__(self, exists, nonexists):
        self._exists = exists
        self._nonexists = nonexists
        self._names = []

    def add(self, name):
        self._names.append(get_op_tensor_name(name)[0])

    def log(self):
        if len(self._names):
            logger.warn("The following variables are in the {}, but not found in the {}: {}".format(
                self._exists, self._nonexists, ', '.join(self._names)))


@graph_memoized
def get_global_step_var():
    """
    Returns:
        tf.Tensor: the global_step variable in the current graph. Create if
            doesn't exist.
    """
    scope = tf.VariableScope(reuse=False, name='')  # the root vs
    with tf.variable_scope(scope):
        var = tf.train.get_or_create_global_step()
    return var


@six.add_metaclass(ABCMeta)
class GradientProcessor(object):
    """
    Base class for all gradient processors.
    Gradient processors can be applied to optimizers by
    :func:`optimizer.apply_grad_processors`.

    Subclass should override the ``_process()`` method.
    """
    _name_scope = None

    def process(self, grads):
        """
        Process the symbolic gradients.

        Args:
            grads (list): list of (grad, var).
        Returns:
            list: processed gradients, with the same type as input.
        """

        # reuse the old name_scope, if process() is called multiple times
        if self._name_scope is None:
            with tf.name_scope(type(self).__name__) as scope:
                self._name_scope = scope
                return self._process(grads)
        else:
            with tf.name_scope(self._name_scope):
                return self._process(grads)

    @abstractmethod
    def _process(self, grads):
        pass


class FilterNoneGrad(GradientProcessor):
    """
    Skip the update and print a warning (instead of crashing),
    when the gradient of certain variable is None.
    """
    def __init__(self, verbose=True):
        """
        Args:
            verbose (bool): whether to print warning about None gradients.
        """
        super(FilterNoneGrad, self).__init__()
        self._verbose = verbose

    def _process(self, grads):
        g = []
        to_print = []
        for grad, var in grads:
            if grad is None:
                to_print.append(var.op.name)
            else:
                g.append((grad, var))
        if self._verbose and len(to_print):
            message = ', '.join(to_print)
            logger.warn("No gradient w.r.t {} trainable variables: {}".format(len(to_print), message))
        return g


class SessionUpdate(object):
    """ Update the variables in a session """

    def __init__(self, sess, vars_to_update):
        """
        Args:
            sess (tf.Session): a session object
            vars_to_update: a collection of variables to update
        """
        self.sess = sess
        self.name_map = {v.name: v for v in vars_to_update}

    @staticmethod
    def load_value_to_var(var, val, strict=False):
        """
        Call `var.load(val)` with the default session, with some type checks.

        Args:
            var (tf.Variable):
            strict (bool): Behave less strict if set to False.
        """
        if strict:
            var.load(val)
            return
        name = var.op.name

        # check incompatible shape
        varshape = tuple(var.get_shape().as_list())
        if varshape != val.shape:
            # TODO only allow reshape when shape different by empty axis
            if np.prod(varshape) != np.prod(val.shape):
                raise ValueError(
                    "Trying to load a tensor of shape {} into the variable '{}' whose shape is {}.".format(
                        val.shape, name, varshape))
            logger.warn("The tensor is reshaped from {} to {} when assigned to '{}'".format(
                val.shape, varshape, name))
            val = val.reshape(varshape)

        # fix some common type incompatibility problems, but not all
        def upcast(vartype, valtype):
            # allow up-casting
            if vartype == tf.float64 and valtype == np.float32:
                return np.float64
            if vartype in [tf.int64, tf.int32] and valtype in [np.int32, np.int16, np.int8]:
                return np.int64 if vartype == tf.int64 else np.int32
            return None

        def downcast(vartype, valtype):
            # allow down-casting
            if vartype == tf.float16 and valtype == np.float32:
                return np.float16
            return None

        if hasattr(val, 'dtype'):
            vartype = var.value().dtype
            if vartype != val.dtype:
                msg = "Variable {} has dtype {} but was given a value of dtype {}.".format(name, vartype, val.dtype)
                newtype = upcast(var.dtype.base_dtype, val.dtype)
                if newtype is not None:
                    val = newtype(val)
                    logger.warn(msg + " Load it after casting!")
                else:
                    newtype = downcast(var.dtype.base_dtype, val.dtype)
                    if newtype is not None:
                        val = newtype(val)
                        logger.warn(msg + " Load it after downcasting!")
                    else:
                        assert vartype == val.dtype, msg
        try:
            # change for tf2 assign is around 1000x faster
            # var.load(val)
            var.assign(val)
        except tf.errors.InvalidArgumentError:
            logger.exc("Cannot load this value to the variable {}".format(name))

    def update(self, prms):
        """
        Args:
            prms(dict): dict of {variable name: value}
                Any name in prms must be in the graph and in vars_to_update.
        """
        with self.sess.as_default():
            for name, value in six.iteritems(prms):
                assert name in self.name_map
                v = self.name_map[name]
                SessionUpdate.load_value_to_var(v, value)


class SessionInit(object):
    """ Base class for utilities to load variables to a (existing) session. """
    def init(self, sess):
        """
        Initialize a session

        Args:
            sess (tf.Session): the session
        """
        self._setup_graph()
        self._run_init(sess)

    def _setup_graph(self):
        pass

    def _run_init(self, sess):
        pass


class JustCurrentSession(SessionInit):
    """ This is a no-op placeholder"""
    pass


def get_checkpoint_path(model_path):
    """
    Work around TF problems in checkpoint path handling.

    Args:
        model_path: a user-input path
    Returns:
        str: the argument that can be passed to NewCheckpointReader
    """
    if os.path.basename(model_path) == model_path:
        model_path = os.path.join('.', model_path)  # avoid #4921 and #6142
    if os.path.basename(model_path) == 'checkpoint':
        assert tf.gfile.Exists(model_path), model_path
        model_path = tf.train.latest_checkpoint(os.path.dirname(model_path))
        # to be consistent with either v1 or v2

    # fix paths if provided a wrong one
    new_path = model_path
    if '00000-of-00001' in model_path:
        new_path = model_path.split('.data')[0]
    elif model_path.endswith('.index'):
        new_path = model_path.split('.index')[0]
    if new_path != model_path:
        logger.info(
            "Checkpoint path {} is auto-corrected to {}.".format(model_path, new_path))
        model_path = new_path
    assert tf.gfile.Exists(model_path) or tf.gfile.Exists(model_path + '.index'), model_path
    return model_path


def get_savename_from_varname(
        varname, varname_prefix=None,
        savename_prefix=None):
    """
    Args:
        varname(str): a variable name in the graph
        varname_prefix(str): an optional prefix that may need to be removed in varname
        savename_prefix(str): an optional prefix to append to all savename
    Returns:
        str: the name used to save the variable
    """
    name = varname
    if varname_prefix is not None \
            and name.startswith(varname_prefix):
        name = name[len(varname_prefix) + 1:]
    if savename_prefix is not None:
        name = savename_prefix + '/' + name
    return name


def is_training_name(name):
    """
    **Guess** if this variable is only used in training.
    Only used internally to avoid too many logging. Do not use it.
    """
    # TODO: maybe simply check against TRAINABLE_VARIABLES and MODEL_VARIABLES?
    # TODO or use get_slot_names()
    name = get_op_tensor_name(name)[0]
    if name.endswith('/Adam') or name.endswith('/Adam_1'):
        return True
    if name.endswith('/Momentum'):
        return True
    if name.endswith('/Adadelta') or name.endswith('/Adadelta_1'):
        return True
    if name.endswith('/RMSProp') or name.endswith('/RMSProp_1'):
        return True
    if name.endswith('/Adagrad'):
        return True
    if name.startswith('EMA/'):  # all the moving average summaries
        return True
    if name.startswith('AccumGrad') or name.endswith('/AccumGrad'):
        return True
    if name.startswith('apply_gradients'):
        return True
    return False


class DictRestore(SessionInit):
    """
    Restore variables from a dictionary.
    """

    def __init__(self, variable_dict):
        """
        Args:
            variable_dict (dict): a dict of {name: value}
        """
        assert isinstance(variable_dict, dict), type(variable_dict)
        # use varname (with :0) for consistency
        self._prms = {get_op_tensor_name(n)[1]: v for n, v in six.iteritems(variable_dict)}

    def _run_init(self, sess):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        variable_names = set([k.name for k in variables])
        param_names = set(six.iterkeys(self._prms))

        intersect = variable_names & param_names

        logger.info("Variables to restore from dict: {}".format(', '.join(map(str, intersect))))

        mismatch = MismatchLogger('graph', 'dict')
        for k in sorted(variable_names - param_names):
            if not is_training_name(k):
                mismatch.add(k)
        mismatch.log()
        mismatch = MismatchLogger('dict', 'graph')
        for k in sorted(param_names - variable_names):
            mismatch.add(k)
        mismatch.log()

        upd = SessionUpdate(sess, [v for v in variables if v.name in intersect])
        logger.info("Restoring {} variables from dict ...".format(len(intersect)))
        upd.update({name: value for name, value in six.iteritems(self._prms) if name in intersect})


class SaverRestore(SessionInit):
    """
    Restore a tensorflow checkpoint saved by :class:`tf.train.Saver` or :class:`ModelSaver`.
    """
    def __init__(self, model_path, prefix=None, ignore=[]):
        """
        Args:
            model_path (str): a model name (model-xxxx) or a ``checkpoint`` file.
            prefix (str): during restore, add a ``prefix/`` for every variable in this checkpoint.
            ignore (list[str]): list of tensor names that should be ignored during loading, e.g. learning-rate
        """
        if model_path.endswith('.npy') or model_path.endswith('.npz'):
            logger.warn("SaverRestore expect a TF checkpoint, but got a model path '{}'.".format(model_path) +
                        " To load from a dict, use 'DictRestore'.")
        model_path = get_checkpoint_path(model_path)
        self.path = model_path  # attribute used by AutoResumeTrainConfig!
        self.prefix = prefix
        self.ignore = [i if i.endswith(':0') else i + ':0' for i in ignore]

    def _setup_graph(self):
        dic = self._get_restore_dict()
        self.saver = tf.train.Saver(var_list=dic, name=str(id(dic)))

    def _run_init(self, sess):
        logger.info("Restoring checkpoint from {} ...".format(self.path))
        self.saver.restore(sess, self.path)

    @staticmethod
    def _read_checkpoint_vars(model_path):
        """ return a set of strings """
        reader = tf.train.NewCheckpointReader(model_path)
        reader = CheckpointReaderAdapter(reader)    # use an adapter to standardize the name
        ckpt_vars = reader.get_variable_to_shape_map().keys()
        return reader, set(ckpt_vars)

    def _match_vars(self, func):
        reader, chkpt_vars = SaverRestore._read_checkpoint_vars(self.path)
        graph_vars = tf.global_variables()
        chkpt_vars_used = set()

        mismatch = MismatchLogger('graph', 'checkpoint')
        for v in graph_vars:
            name = get_savename_from_varname(v.name, varname_prefix=self.prefix)
            if name in self.ignore and reader.has_tensor(name):
                logger.info("Variable {} in the graph will not be loaded from the checkpoint!".format(name))
            else:
                if reader.has_tensor(name):
                    func(reader, name, v)
                    chkpt_vars_used.add(name)
                else:
                    # use tensor name (instead of op name) for logging, to be consistent with the reverse case
                    if not is_training_name(v.name):
                        mismatch.add(v.name)
        mismatch.log()
        mismatch = MismatchLogger('checkpoint', 'graph')
        if len(chkpt_vars_used) < len(chkpt_vars):
            unused = chkpt_vars - chkpt_vars_used
            for name in sorted(unused):
                if not is_training_name(name):
                    mismatch.add(name)
        mismatch.log()

    def _get_restore_dict(self):
        var_dict = {}

        def f(reader, name, v):
            name = reader.get_real_name(name)
            assert name not in var_dict, "Restore conflict: {} and {}".format(v.name, var_dict[name].name)
            var_dict[name] = v
        self._match_vars(f)
        return var_dict


def get_model_loader(filename):
    """
    Get a corresponding model loader by looking at the file name.

    Returns:
        SessInit: either a :class:`DictRestore` (if name ends with 'npy/npz') or
        :class:`SaverRestore` (otherwise).
    """
    assert isinstance(filename, six.string_types), filename
    if filename.endswith('.npy'):
        assert tf.gfile.Exists(filename), filename
        return DictRestore(np.load(filename, encoding='latin1').item())
    elif filename.endswith('.npz'):
        assert tf.gfile.Exists(filename), filename
        obj = np.load(filename)
        return DictRestore(dict(obj))
    else:
        return SaverRestore(filename)
