# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
# File: serialize.py

import os
import sys

import msgpack
import msgpack_numpy

import tensorpack_logger as logger

msgpack_numpy.patch()
assert msgpack.version >= (0, 5, 2)

__all__ = ['loads', 'dumps']


MAX_MSGPACK_LEN = 1000000000


def create_dummy_func(func, dependency):
    """
    When a dependency of a function is not available, create a dummy function which throws ImportError when used.

    Args:
        func (str): name of the function.
        dependency (str or list[str]): name(s) of the dependency.

    Returns:
        function: a function object
    """
    assert not building_rtfd()

    if isinstance(dependency, (list, tuple)):
        dependency = ','.join(dependency)

    def _dummy(*args, **kwargs):
        raise ImportError("Cannot import '{}', therefore '{}' is not available".format(dependency, func))
    return _dummy


def building_rtfd():
    """
    Returns:
        bool: if tensorpack is being imported to generate docs now.
    """
    return os.environ.get('READTHEDOCS') == 'True' \
        or os.environ.get('DOC_BUILDING')


def dumps_msgpack(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object.
    """
    return msgpack.dumps(obj, use_bin_type=True)


def loads_msgpack(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    # Since 0.6, the default max size was set to 1MB.
    # We change it to approximately 1G.
    return msgpack.loads(buf, raw=False,
                         max_bin_len=MAX_MSGPACK_LEN,
                         max_array_len=MAX_MSGPACK_LEN,
                         max_map_len=MAX_MSGPACK_LEN,
                         max_str_len=MAX_MSGPACK_LEN)


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object.
        May not be compatible across different versions of pyarrow.
    """
    return pa.serialize(obj).to_buffer()


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


# import pyarrow has a lot of side effect:
# https://github.com/apache/arrow/pull/2329
# https://groups.google.com/a/tensorflow.org/forum/#!topic/developers/TMqRaT-H2bI
# So we use msgpack as default.
if os.environ.get('TENSORPACK_SERIALIZE', 'msgpack') == 'pyarrow':
    try:
        import pyarrow as pa
    except ImportError:
        loads_pyarrow = create_dummy_func('loads_pyarrow', ['pyarrow'])  # noqa
        dumps_pyarrow = create_dummy_func('dumps_pyarrow', ['pyarrow'])  # noqa

    if 'horovod' in sys.modules:
        logger.warn("Horovod and pyarrow may have symbol conflicts. "
                    "Uninstall pyarrow and use msgpack instead.")
    loads = loads_pyarrow
    dumps = dumps_pyarrow
else:
    loads = loads_msgpack
    dumps = dumps_msgpack
