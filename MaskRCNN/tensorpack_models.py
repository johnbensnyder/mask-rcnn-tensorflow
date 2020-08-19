import tensorflow.compat.v1 as tf
from contextlib import contextmanager
import six
from functools import wraps
import functools
import copy
import re
import numpy as np
from tensorflow.python.training import moving_averages


from tensorpack_tfutils import get_tf_version_tuple, get_arg_scope, get_shape_str, custom_getter_scope, get_current_tower_context
from tensorpack_tfutils import backup_collection, restore_collection, StaticDynamicShape
from tensorpack_utils import get_data_format, shape2d, shape4d, graph_memoized
import tensorpack_logger as logger

from utils.mixed_precision import mixed_precision_scope


def rename_get_variable(mapping):
    """
    Args:
        mapping(dict): an old -> new mapping for variable basename. e.g. {'kernel': 'W'}

    Returns:
        A context where the variables are renamed.
    """
    def custom_getter(getter, name, *args, **kwargs):
        splits = name.split('/')
        basename = splits[-1]
        if basename in mapping:
            basename = mapping[basename]
            splits[-1] = basename
            name = '/'.join(splits)
        return getter(name, *args, **kwargs)
    return custom_getter_scope(custom_getter)

def map_common_tfargs(kwargs):
    df = kwargs.pop('data_format', None)
    if df is not None:
        df = get_data_format(df, tfmode=True)
        kwargs['data_format'] = df

    old_nl = kwargs.pop('nl', None)
    if old_nl is not None:
        kwargs['activation'] = lambda x, name=None: old_nl(x, name=name)

    if 'W_init' in kwargs:
        kwargs['kernel_initializer'] = kwargs.pop('W_init')

    if 'b_init' in kwargs:
        kwargs['bias_initializer'] = kwargs.pop('b_init')
    return kwargs


# -

def convert_to_tflayer_args(args_names, name_mapping):
    """
    After applying this decorator:
    1. data_format becomes tf.layers style
    2. nl becomes activation
    3. initializers are renamed
    4. positional args are transformed to corresponding kwargs, according to args_names
    5. kwargs are mapped to tf.layers names if needed, by name_mapping
    """

    def decorator(func):
        @functools.wraps(func)
        def decorated_func(inputs, *args, **kwargs):
            kwargs = map_common_tfargs(kwargs)

            posarg_dic = {}
            assert len(args) <= len(args_names), \
                "Please use kwargs instead of positional args to call this model, " \
                "except for the following arguments: {}".format(', '.join(args_names))
            for pos_arg, name in zip(args, args_names):
                posarg_dic[name] = pos_arg

            ret = {}
            for name, arg in six.iteritems(kwargs):
                newname = name_mapping.get(name, None)
                if newname is not None:
                    assert newname not in kwargs, \
                        "Argument {} and {} conflicts!".format(name, newname)
                else:
                    newname = name
                ret[newname] = arg
            ret.update(posarg_dic)  # Let pos arg overwrite kw arg, for argscope to work

            return func(inputs, **ret)

        return decorated_func

    return decorator

# make sure each layer is only logged once
_LAYER_LOGGED = set()
_LAYER_REGISTRY = {}

def _register(name, func):
    if name in _LAYER_REGISTRY:
        raise ValueError("Layer named {} is already registered!".format(name))
    if name in ['tf']:
        raise ValueError(logger.error("A layer cannot be named {}".format(name)))
    _LAYER_REGISTRY[name] = func

    # handle alias
    if name == 'Conv2DTranspose':
        _register('Deconv2D', func)

def layer_register(
        log_shape=False,
        use_scope=True):
    """
    Args:
        log_shape (bool): log input/output shape of this layer
        use_scope (bool or None):
            Whether to call this layer with an extra first argument as scope.
            When set to None, it can be called either with or without
            the scope name argument.
            It will try to figure out by checking if the first argument
            is string or not.

    Returns:
        A decorator used to register a layer.

    Example:

    .. code-block:: python

        @layer_register(use_scope=True)
        def add10(x):
            return x + tf.get_variable('W', shape=[10])
    """

    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            assert args[0] is not None, args
            if use_scope:
                name, inputs = args[0], args[1]
                args = args[1:]  # actual positional args used to call func
                assert isinstance(name, six.string_types), "First argument for \"{}\" should be a string. ".format(
                    func.__name__) + "Did you forget to specify the name of the layer?"
            else:
                assert not log_shape
                if isinstance(args[0], six.string_types):
                    if use_scope is False:
                        logger.warn(
                            "Please call layer {} without the first scope name argument, "
                            "or register the layer with use_scope=None to allow "
                            "two calling methods.".format(func.__name__))
                    name, inputs = args[0], args[1]
                    args = args[1:]  # actual positional args used to call func
                else:
                    inputs = args[0]
                    name = None
            if not (isinstance(inputs, (tf.Tensor, tf.Variable)) or
                    (isinstance(inputs, (list, tuple)) and
                        isinstance(inputs[0], (tf.Tensor, tf.Variable)))):
                raise ValueError("Invalid inputs to layer: " + str(inputs))

            # use kwargs from current argument scope
            actual_args = copy.copy(get_arg_scope()[func.__name__])
            # explicit kwargs overwrite argscope
            actual_args.update(kwargs)
            # if six.PY3:
            #     # explicit positional args also override argscope. only work in PY3
            #     posargmap = inspect.signature(func).bind_partial(*args).arguments
            #     for k in six.iterkeys(posargmap):
            #         if k in actual_args:
            #             del actual_args[k]

            if name is not None:        # use scope
                with tf.variable_scope(name) as scope:
                    # this name is only used to surpress logging, doesn't hurt to do some heuristics
                    scope_name = re.sub('tower[0-9]+/', '', scope.name)
                    do_log_shape = log_shape and scope_name not in _LAYER_LOGGED
                    if do_log_shape:
                        logger.info("{} input: {}".format(scope.name, get_shape_str(inputs)))

                    # run the actual function
                    outputs = func(*args, **actual_args)

                    if do_log_shape:
                        # log shape info and add activation
                        logger.info("{} output: {}".format(
                            scope.name, get_shape_str(outputs)))
                        _LAYER_LOGGED.add(scope_name)
            else:
                # run the actual function
                outputs = func(*args, **actual_args)
            return outputs

        wrapped_func.symbolic_function = func   # attribute to access the underlying function object
        wrapped_func.use_scope = use_scope
        _register(func.__name__, wrapped_func)
        return wrapped_func

    return wrapper

class VariableHolder(object):
    """ A proxy to access variables defined in a layer. """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: {name:variable}
        """
        self._vars = {}
        for k, v in six.iteritems(kwargs):
            self._add_variable(k, v)

    def _add_variable(self, name, var):
        assert name not in self._vars
        self._vars[name] = var

    def __setattr__(self, name, var):
        if not name.startswith('_'):
            self._add_variable(name, var)
        else:
            # private attributes
            super(VariableHolder, self).__setattr__(name, var)

    def __getattr__(self, name):
        return self._vars[name]

    def all(self):
        """
        Returns:
            list of all variables
        """
        return list(six.itervalues(self._vars))


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['units'],
    name_mapping={'out_dim': 'units'})
def FullyConnected(
        inputs,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    A wrapper around `tf.layers.Dense`.
    One difference to maintain backward-compatibility:
    Default weight initializer is variance_scaling_initializer(2.0).

    Variable Names:

    * ``W``: weights of shape [in_dim, out_dim]
    * ``b``: bias
    """
    def batch_flatten(x):
        """
        Flatten the tensor except the first dimension.
        """
        shape = x.get_shape().as_list()[1:]
        if None not in shape:
            return tf.reshape(x, [-1, int(np.prod(shape))])
        return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

    if kernel_initializer is None:
        if get_tf_version_tuple() <= (1, 12):
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)
        else:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')

    inputs = batch_flatten(inputs)
    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            _reuse=tf.get_variable_scope().reuse)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())
        ret = tf.identity(ret, name='output')

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return ret

@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['pool_size', 'strides'],
    name_mapping={'shape': 'pool_size', 'stride': 'strides'})
def MaxPooling(
        inputs,
        pool_size,
        strides=None,
        padding='valid',
        data_format='channels_last'):
    """
    Same as `tf.layers.MaxPooling2D`. Default strides is equal to pool_size.
    """
    if strides is None:
        strides = pool_size
    layer = tf.layers.MaxPooling2D(pool_size, strides, padding=padding, data_format=data_format)
    ret = layer.apply(inputs, scope=tf.get_variable_scope())
    return tf.identity(ret, name='output')

def UnPooling2x2ZeroFilled(x):
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
        return ret

@layer_register(log_shape=True)
def FixedUnPooling(x, shape, unpool_mat=None, data_format='channels_last'):
    """
    Unpool the input with a fixed matrix to perform kronecker product with.

    Args:
        x (tf.Tensor): a 4D image tensor
        shape: int or (h, w) tuple
        unpool_mat: a tf.Tensor or np.ndarray 2D matrix with size=shape.
            If is None, will use a matrix with 1 at top-left corner.

    Returns:
        tf.Tensor: a 4D image tensor.
    """
    data_format = get_data_format(data_format, tfmode=False)
    shape = shape2d(shape)

    output_shape = StaticDynamicShape(x)
    output_shape.apply(1 if data_format == 'NHWC' else 2, lambda x: x * shape[0])
    output_shape.apply(2 if data_format == 'NHWC' else 3, lambda x: x * shape[1])

    # a faster implementation for this special case
    if shape[0] == 2 and shape[1] == 2 and unpool_mat is None and data_format == 'NHWC':
        ret = UnPooling2x2ZeroFilled(x)
    else:
        # check unpool_mat
        if unpool_mat is None:
            mat = np.zeros(shape, dtype='float32')
            mat[0][0] = 1
            unpool_mat = tf.constant(mat, name='unpool_mat')
        elif isinstance(unpool_mat, np.ndarray):
            unpool_mat = tf.constant(unpool_mat, name='unpool_mat')
        assert unpool_mat.shape.as_list() == list(shape)

        if data_format == 'NHWC':
            x = tf.transpose(x, [0, 3, 1, 2])
        # perform a tensor-matrix kronecker product
        x = tf.expand_dims(x, -1)       # bchwx1
        mat = tf.expand_dims(unpool_mat, 0)  # 1xshxsw
        #ret = tf.tensordot(x, mat, axes=1)  # bxcxhxwxshxsw
        x = tf.transpose(x, [4, 0, 1, 2, 3])
        mat = tf.transpose(mat, [1, 2, 0])
        ret = tf.tensordot(mat, x, axes=1) # shxswxbxcxhxw
        ret = tf.transpose(ret, [2, 3, 4, 5, 0, 1]) # bxcxhxwxshxsw

        if data_format == 'NHWC':
            ret = tf.transpose(ret, [0, 2, 4, 3, 5, 1])
        else:
            ret = tf.transpose(ret, [0, 1, 2, 4, 3, 5])

        shape3_dyn = [output_shape.get_dynamic(k) for k in range(1, 4)]
        ret = tf.reshape(ret, tf.stack([-1] + shape3_dyn))

    ret.set_shape(tf.TensorShape(output_shape.get_static()))
    return ret

@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv2D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        split=1,
        seed=None):
    """
    A wrapper around `tf.layers.Conv2D`.
    Some differences to maintain backward-compatibility:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.
    3. Support 'split' argument to do group conv. Note that this is not efficient.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    print(f"Use {data_format} data format")
    if kernel_initializer is None:
        if get_tf_version_tuple() <= (1, 12):
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0, seed=seed)
        else:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal', seed=seed)
    dilation_rate = shape2d(dilation_rate)

    if split == 1 and dilation_rate == [1, 1]:
        # tf.layers.Conv2D has bugs with dilations (https://github.com/tensorflow/tensorflow/issues/26797)
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv2D(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                _reuse=tf.get_variable_scope().reuse)
            ret = layer.apply(inputs, scope=tf.get_variable_scope())
            ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=layer.kernel)
        if use_bias:
            ret.variables.b = layer.bias

    else:
        # group conv implementation
        data_format = get_data_format(data_format, tfmode=False)
        in_shape = inputs.get_shape().as_list()
        channel_axis = 3 if data_format == 'NHWC' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
        assert in_channel % split == 0

        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Not supported by group conv or dilated conv!"

        out_channel = filters
        assert out_channel % split == 0
        assert dilation_rate == [1, 1] or get_tf_version_tuple() >= (1, 5), 'TF>=1.5 required for dilated conv.'

        kernel_shape = shape2d(kernel_size)
        filter_shape = kernel_shape + [in_channel / split, out_channel]
        stride = shape4d(strides, data_format=data_format)

        kwargs = dict(data_format=data_format)
        if get_tf_version_tuple() >= (1, 5):
            kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)

        W = tf.get_variable(
            'W', filter_shape, initializer=kernel_initializer)

        if use_bias:
            b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

        if split == 1:
            conv = tf.nn.conv2d(inputs, W, stride, padding.upper(), **kwargs)
        else:
            conv = None
            if get_tf_version_tuple() >= (1, 13):
                try:
                    conv = tf.nn.conv2d(inputs, W, stride, padding.upper(), **kwargs)
                except ValueError:
                    log_once("CUDNN group convolution support is only available with "
                             "https://github.com/tensorflow/tensorflow/pull/25818 . "
                             "Will fall back to a loop-based slow implementation instead!", 'warn')
            if conv is None:
                inputs = tf.split(inputs, split, channel_axis)
                kernels = tf.split(W, split, 3)
                outputs = [tf.nn.conv2d(i, k, stride, padding.upper(), **kwargs)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

        ret = tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv
        if activation is not None:
            ret = activation(ret)
        ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b
    return ret


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size', 'strides'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv2DTranspose(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        seed=None):
    """
    A wrapper around `tf.layers.Conv2DTranspose`.
    Some differences to maintain backward-compatibility:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    if kernel_initializer is None:
        if get_tf_version_tuple() <= (1, 12):
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0, seed=seed)
        else:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal', seed=seed)

    if get_tf_version_tuple() <= (1, 12):
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv2DTranspose(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                _reuse=tf.get_variable_scope().reuse)
            ret = layer.apply(inputs, scope=tf.get_variable_scope())
            ret = tf.identity(ret, name='output')
        ret.variables = VariableHolder(W=layer.kernel)
        if use_bias:
            ret.variables.b = layer.bias
    else:
        # Update from tensorpack: https://github.com/tensorpack/tensorpack/blob/master/tensorpack/models/conv2d.py#L204
        # To avoid Keras bugs in TF > 1.14. https://github.com/tensorflow/tensorflow/issues/25946
        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Unsupported arguments due to Keras bug in TensorFlow 1.13"
        data_format = get_data_format(data_format, tfmode=False)
        shape_dyn = tf.shape(inputs)
        strides2d = shape2d(strides)
        channels_in = inputs.shape[1 if data_format == 'NCHW' else 3]
        if data_format == 'NCHW':
            channels_in = inputs.shape[1]
            out_shape_dyn = tf.stack(
                [shape_dyn[0], filters,
                 shape_dyn[2] * strides2d[0],
                 shape_dyn[3] * strides2d[1]])
            out_shape3_sta = [filters,
                              None if inputs.shape[2] is None else inputs.shape[2] * strides2d[0],
                              None if inputs.shape[3] is None else inputs.shape[3] * strides2d[1]]
        else:
            channels_in = inputs.shape[-1]
            out_shape_dyn = tf.stack(
                [shape_dyn[0],
                 shape_dyn[1] * strides2d[0],
                 shape_dyn[2] * strides2d[1],
                 filters])
            out_shape3_sta = [None if inputs.shape[1] is None else inputs.shape[1] * strides2d[0],
                              None if inputs.shape[2] is None else inputs.shape[2] * strides2d[1],
                              filters]

        kernel_shape = shape2d(kernel_size)
        import os
        fp16 = True if os.getenv("TENSORPACK_FP16") else False
        with mixed_precision_scope(mixed=fp16):
            W = tf.get_variable('W', kernel_shape + [filters, channels_in], initializer=kernel_initializer, dtype=tf.float16 if fp16 else tf.float32)
            if use_bias:
                b = tf.get_variable('b', [filters], initializer=bias_initializer, dtype=tf.float16 if fp16 else tf.float32)
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            conv = tf.nn.conv2d_transpose(
                inputs, W, out_shape_dyn,
                shape4d(strides, data_format=data_format),
                padding=padding.upper(),
                data_format=data_format)
            conv.set_shape(tf.TensorShape([None] + out_shape3_sta))

            ret = tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv
            if activation is not None:
                ret = activation(ret)
            ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b

    return ret

Deconv2D = Conv2DTranspose


@graph_memoized
def _log_once(msg):
    logger.info(msg)

def regularize_cost(regex, func, name='regularize_cost'):
    """
    Apply a regularizer on trainable variables matching the regex, and print
    the matched variables (only print once in multi-tower training).
    In replicated mode, it will only regularize variables within the current tower.

    If called under a TowerContext with `is_training==False`, this function returns a zero constant tensor.

    Args:
        regex (str): a regex to match variable names, e.g. "conv.*/W"
        func: the regularization function, which takes a tensor and returns a scalar tensor.
            E.g., ``tf.nn.l2_loss, tf.contrib.layers.l1_regularizer(0.001)``.

    Returns:
        tf.Tensor: a scalar, the total regularization cost.

    Example:
        .. code-block:: python

            cost = cost + regularize_cost("fc.*/W", l2_regularizer(1e-5))
    """
    assert len(regex)
    ctx = get_current_tower_context()
    if not ctx.is_training:
        # Currently cannot build the wd_cost correctly at inference,
        # because ths vs_name used in inference can be '', therefore the
        # variable filter will fail
        return tf.constant(0, dtype=tf.float32, name='empty_' + name)

    # If vars are shared, regularize all of them
    # If vars are replicated, only regularize those in the current tower
    if ctx.has_own_variables:
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    else:
        params = tf.trainable_variables()

    names = []

    with tf.name_scope(name + '_internals'):
        costs = []
        for p in params:
            para_name = p.op.name
            if re.search(regex, para_name):
                regloss = func(p)
                assert regloss.dtype.is_floating, regloss
                # Some variables may not be fp32, but it should
                # be fine to assume regularization in fp32
                if regloss.dtype != tf.float32:
                    regloss = tf.cast(regloss, tf.float32)
                costs.append(regloss)
                names.append(p.name)
        if not costs:
            return tf.constant(0, dtype=tf.float32, name='empty_' + name)

    # remove tower prefix from names, and print
    if len(ctx.vs_name):
        prefix = ctx.vs_name + '/'
        prefixlen = len(prefix)

        def f(name):
            if name.startswith(prefix):
                return name[prefixlen:]
            return name
        names = list(map(f, names))
    logger.info("regularize_cost() found {} variables to regularize.".format(len(names)))
    _log_once("The following tensors will be regularized: {}".format(', '.join(names)))

    return tf.add_n(costs, name=name)

def get_bn_variables(n_out, use_scale, use_bias, beta_init, gamma_init):
    if use_bias:
        beta = tf.get_variable('beta', [n_out], initializer=beta_init)
    else:
        beta = tf.zeros([n_out], name='beta')
    if use_scale:
        gamma = tf.get_variable('gamma', [n_out], initializer=gamma_init)
    else:
        gamma = tf.ones([n_out], name='gamma')
    # x * gamma + beta

    moving_mean = tf.get_variable('mean/EMA', [n_out],
                                  initializer=tf.constant_initializer(), trainable=False)
    moving_var = tf.get_variable('variance/EMA', [n_out],
                                 initializer=tf.constant_initializer(1.0), trainable=False)
    return beta, gamma, moving_mean, moving_var


def update_bn_ema(xn, batch_mean, batch_var,
                  moving_mean, moving_var, decay, internal_update):
    update_op1 = moving_averages.assign_moving_average(
        moving_mean, batch_mean, decay, zero_debias=False,
        name='mean_ema_op')
    update_op2 = moving_averages.assign_moving_average(
        moving_var, batch_var, decay, zero_debias=False,
        name='var_ema_op')

    if internal_update:
        with tf.control_dependencies([update_op1, update_op2]):
            return tf.identity(xn, name='output')
    else:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
        return tf.identity(xn, name='output')

try:
    # When BN is used as an activation, keras layers try to autograph.convert it
    # This leads to massive warnings
    from tensorflow.python.autograph.impl.api import do_not_convert as disable_autograph
except ImportError:
    def disable_autograph():
        return lambda x: x

@layer_register()
@convert_to_tflayer_args(
    args_names=[],
    name_mapping={
        'use_bias': 'center',
        'use_scale': 'scale',
        'gamma_init': 'gamma_initializer',
        'decay': 'momentum',
        'use_local_stat': 'training'
    })
@disable_autograph()
def BatchNorm(inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,
              center=True, scale=True,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              virtual_batch_size=None,
              data_format='channels_last',
              internal_update=False,
              sync_statistics=None):
    """
    Almost equivalent to `tf.layers.batch_normalization`, but different (and more powerful)
    in the following:

    1. Accepts an alternative `data_format` option when `axis` is None. For 2D input, this argument will be ignored.
    2. Default value for `momentum` and `epsilon` is different.
    3. Default value for `training` is automatically obtained from tensorpack's `TowerContext`, but can be overwritten.
    4. Support the `internal_update` option, which enables the use of BatchNorm layer inside conditionals.
    5. Support the `sync_statistics` option, which is very useful in small-batch models.

    Args:
        internal_update (bool): if False, add EMA update ops to
          `tf.GraphKeys.UPDATE_OPS`. If True, update EMA inside the layer by control dependencies.
          They are very similar in speed, but `internal_update=True` can be used
          when you have conditionals in your model, or when you have multiple networks to train.
          Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/14699
        sync_statistics (str or None): one of None, "nccl", or "horovod".

          By default (None), it uses statistics of the input tensor to normalize.
          This is the standard way BatchNorm was done in most frameworks.

          When set to "nccl", this layer must be used under tensorpack's multi-GPU trainers.
          It uses the aggregated statistics of the whole batch (across all GPUs) to normalize.

          When set to "horovod", this layer must be used under tensorpack's :class:`HorovodTrainer`.
          It uses the aggregated statistics of the whole batch (across all MPI ranks) to normalize.
          Note that on single machine this is significantly slower than the "nccl" implementation.

          If not None, per-GPU E[x] and E[x^2] among all GPUs are averaged to compute
          global mean & variance. Therefore each GPU needs to have the same batch size.

          The BatchNorm layer on each GPU needs to use the same name (`BatchNorm('name', input)`), so that
          statistics can be reduced. If names do not match, this layer will hang.

          This option only has effect in standard training mode.

          This option is also known as "Cross-GPU BatchNorm" as mentioned in:
          `MegDet: A Large Mini-Batch Object Detector <https://arxiv.org/abs/1711.07240>`_.
          Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/18222.

    Variable Names:

    * ``beta``: the bias term. Will be zero-inited by default.
    * ``gamma``: the scale term. Will be one-inited by default.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.

    Note:
        Combinations of ``training`` and ``ctx.is_training``:

        * ``training == ctx.is_training``: standard BN, EMA are maintained during training
          and used during inference. This is the default.
        * ``training and not ctx.is_training``: still use batch statistics in inference.
        * ``not training and ctx.is_training``: use EMA to normalize in
          training. This is useful when you load a pre-trained BN and
          don't want to fine tune the EMA. EMA will not be updated in
          this case.
    """
    # parse shapes
    data_format = get_data_format(data_format, tfmode=False)
    shape = inputs.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4], ndims
    if sync_statistics is not None:
        sync_statistics = sync_statistics.lower()
    assert sync_statistics in [None, 'nccl', 'horovod'], sync_statistics

    if axis is None:
        if ndims == 2:
            axis = 1
        else:
            axis = 1 if data_format == 'NCHW' else 3
    assert axis in [1, 3], axis
    num_chan = shape[axis]

    # parse training/ctx
    ctx = get_current_tower_context()
    if training is None:
        training = ctx.is_training
    training = bool(training)
    TF_version = get_tf_version_tuple()
    freeze_bn_backward = not training and ctx.is_training
    if freeze_bn_backward:
        assert TF_version >= (1, 4), \
            "Fine tuning a BatchNorm model with fixed statistics needs TF>=1.4!"
        if ctx.is_main_training_tower:  # only warn in first tower
            logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
        # Using moving_mean/moving_variance in training, which means we
        # loaded a pre-trained BN and only fine-tuning the affine part.

    if sync_statistics is None or not (training and ctx.is_training):
        coll_bk = backup_collection([tf.GraphKeys.UPDATE_OPS])
        with rename_get_variable(
                {'moving_mean': 'mean/EMA',
                    'moving_variance': 'variance/EMA'}):
            tf_args = dict(
                axis=axis,
                momentum=momentum, epsilon=epsilon,
                center=center, scale=scale,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                # https://github.com/tensorflow/tensorflow/issues/10857#issuecomment-410185429
                fused=(ndims == 4 and axis in [1, 3] and not freeze_bn_backward),
                _reuse=tf.get_variable_scope().reuse)
            if TF_version >= (1, 5):
                tf_args['virtual_batch_size'] = virtual_batch_size
            else:
                assert virtual_batch_size is None, "Feature not supported in this version of TF!"
            layer = tf.layers.BatchNormalization(**tf_args)
            xn = layer.apply(inputs, training=training, scope=tf.get_variable_scope())

        # maintain EMA only on one GPU is OK, even in replicated mode.
        # because during training, EMA isn't used
        if ctx.is_main_training_tower:
            for v in layer.non_trainable_variables:
                if isinstance(v, tf.Variable):
                    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)
        if not ctx.is_main_training_tower or internal_update:
            restore_collection(coll_bk)

        if training and internal_update:
            assert layer.updates
            with tf.control_dependencies(layer.updates):
                ret = tf.identity(xn, name='output')
        else:
            ret = tf.identity(xn, name='output')

        vh = ret.variables = VariableHolder(
            moving_mean=layer.moving_mean,
            mean=layer.moving_mean,  # for backward-compatibility
            moving_variance=layer.moving_variance,
            variance=layer.moving_variance)  # for backward-compatibility
        if scale:
            vh.gamma = layer.gamma
        if center:
            vh.beta = layer.beta
    else:
        red_axis = [0] if ndims == 2 else ([0, 2, 3] if axis == 1 else [0, 1, 2])

        new_shape = None  # don't need to reshape unless ...
        if ndims == 4 and axis == 1:
            new_shape = [1, num_chan, 1, 1]

        batch_mean = tf.reduce_mean(inputs, axis=red_axis)
        batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axis)

        if sync_statistics == 'nccl':
            num_dev = ctx.total
            if num_dev == 1:
                logger.warn("BatchNorm(sync_statistics='nccl') is used with only one tower!")
            else:
                assert six.PY2 or TF_version >= (1, 10), \
                    "Cross-GPU BatchNorm is only supported in TF>=1.10 ." \
                    "Upgrade TF or apply this patch manually: https://github.com/tensorflow/tensorflow/pull/20360"

                if TF_version <= (1, 12):
                    try:
                        from tensorflow.contrib.nccl.python.ops.nccl_ops import _validate_and_load_nccl_so
                    except Exception:
                        pass
                    else:
                        _validate_and_load_nccl_so()
                    from tensorflow.contrib.nccl.ops import gen_nccl_ops
                else:
                    from tensorflow.python.ops import gen_nccl_ops
                shared_name = re.sub('tower[0-9]+/', '', tf.get_variable_scope().name)
                batch_mean = gen_nccl_ops.nccl_all_reduce(
                    input=batch_mean,
                    reduction='sum',
                    num_devices=num_dev,
                    shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
                batch_mean_square = gen_nccl_ops.nccl_all_reduce(
                    input=batch_mean_square,
                    reduction='sum',
                    num_devices=num_dev,
                    shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
        elif sync_statistics == 'horovod':
            # Require https://github.com/uber/horovod/pull/331
            import horovod.tensorflow as hvd
            if hvd.size() == 1:
                logger.warn("BatchNorm(sync_statistics='horovod') is used with only one process!")
            else:
                import horovod
                hvd_version = tuple(map(int, horovod.__version__.split('.')))
                assert hvd_version >= (0, 13, 6), "sync_statistics=horovod needs horovod>=0.13.6 !"

                batch_mean = hvd.allreduce(batch_mean, average=True)
                batch_mean_square = hvd.allreduce(batch_mean_square, average=True)
        batch_var = batch_mean_square - tf.square(batch_mean)
        batch_mean_vec = batch_mean
        batch_var_vec = batch_var

        beta, gamma, moving_mean, moving_var = get_bn_variables(
            num_chan, scale, center, beta_initializer, gamma_initializer)
        if new_shape is not None:
            batch_mean = tf.reshape(batch_mean, new_shape)
            batch_var = tf.reshape(batch_var, new_shape)
            # Using fused_batch_norm(is_training=False) is actually slightly faster,
            # but hopefully this call will be JITed in the future.
            xn = tf.nn.batch_normalization(
                inputs, batch_mean, batch_var,
                tf.reshape(beta, new_shape),
                tf.reshape(gamma, new_shape), epsilon)
        else:
            xn = tf.nn.batch_normalization(
                inputs, batch_mean, batch_var,
                beta, gamma, epsilon)

        if ctx.is_main_training_tower:
            ret = update_bn_ema(
                xn, batch_mean_vec, batch_var_vec, moving_mean, moving_var,
                momentum, internal_update)
        else:
            ret = tf.identity(xn, name='output')

        vh = ret.variables = VariableHolder(
            moving_mean=moving_mean,
            mean=moving_mean,  # for backward-compatibility
            moving_variance=moving_var,
            variance=moving_var)  # for backward-compatibility
        if scale:
            vh.gamma = gamma
        if center:
            vh.beta = beta
    return ret
