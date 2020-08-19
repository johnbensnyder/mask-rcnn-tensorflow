import functools
from time import time
from datetime import datetime, timedelta
import tensorpack_logger as logger
from tqdm import tqdm
import signal
import platform
import threading
from contextlib import contextmanager
import os
import numpy as np
import tensorflow.compat.v1 as tf
import inspect
import six
import sys
import subprocess
from six.moves import queue
from ctypes import CDLL, POINTER, Structure, byref, c_uint, c_ulonglong

GLOBAL_STEP_INCR_OP_NAME = 'global_step_incr'

def get_tqdm(*args, **kwargs):
    """ Similar to :func:`tqdm.tqdm()`,
    but use tensorpack's default options to have consistent style. """
    return tqdm(*args, **get_tqdm_kwargs(**kwargs))

def shape2d(a):
    """
    Ensure a 2D shape.

    Args:
        a: a int or tuple/list of length 2

    Returns:
        list: of length 2. if ``a`` is a int, return ``[a, a]``.
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))

def get_data_format(data_format, tfmode=True):
    if tfmode:
        dic = {'NCHW': 'channels_first', 'NHWC': 'channels_last'}
    else:
        dic = {'channels_first': 'NCHW', 'channels_last': 'NHWC'}
    ret = dic.get(data_format, data_format)
    if ret not in dic.values():
        raise ValueError("Unknown data_format: {}".format(data_format))
    return ret

def shape4d(a, data_format='channels_last'):
    """
    Ensuer a 4D shape, to use with 4D symbolic functions.

    Args:
        a: a int or tuple/list of length 2

    Returns:
        list: of length 4. if ``a`` is a int, return ``[1, a, a, 1]``
            or ``[1, 1, a, a]`` depending on data_format.
    """
    s2d = shape2d(a)
    if get_data_format(data_format) == 'channels_last':
        return [1] + s2d + [1]
    else:
        return [1, 1] + s2d

memoized = functools.lru_cache(maxsize=None)
""" Alias to :func:`functools.lru_cache`
WARNING: memoization will keep keys and values alive!
"""

def graph_memoized(func):
    """
    Like memoized, but keep one cache per default graph.
    """

    # TODO it keeps the graph alive
    import tensorflow.compat.v1 as tf
    GRAPH_ARG_NAME = '__IMPOSSIBLE_NAME_FOR_YOU__'

    @memoized
    def func_with_graph_arg(*args, **kwargs):
        kwargs.pop(GRAPH_ARG_NAME)
        return func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert GRAPH_ARG_NAME not in kwargs, "No Way!!"
        graph = tf.get_default_graph()
        kwargs[GRAPH_ARG_NAME] = graph
        return func_with_graph_arg(*args, **kwargs)
    return wrapper

def memoized_method(func):
    """
    A decorator that performs memoization on methods. It stores the cache on the object instance itself.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        assert func.__name__ in dir(self), "memoized_method can only be used on method!"

        if not hasattr(self, '_MEMOIZED_CACHE'):
            cache = self._MEMOIZED_CACHE = {}
        else:
            cache = self._MEMOIZED_CACHE

        key = (func, ) + args[1:] + tuple(kwargs)
        ret = cache.get(key, None)
        if ret is not None:
            return ret
        value = func(*args, **kwargs)
        cache[key] = value
        return value

    return wrapper

_RNG_SEED = None
def fix_rng_seed(seed):
    """
    Call this function at the beginning of program to fix rng seed within tensorpack.

    Args:
        seed (int):

    Note:
        See https://github.com/tensorpack/tensorpack/issues/196.

    Example:

        Fix random seed in both tensorpack and tensorflow.

    .. code-block:: python

            import tensorpack.utils.utils as utils

            seed = 42
            utils.fix_rng_seed(seed)
            tesnorflow.set_random_seed(seed)
            # run trainer
    """
    global _RNG_SEED
    _RNG_SEED = int(seed)


def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)


def humanize_time_delta(sec):
    """Humanize timedelta given in seconds

    Args:
        sec (float): time difference in seconds. Must be positive.

    Returns:
        str - time difference as a readable string

    Example:

    .. code-block:: python

        print(humanize_time_delta(1))                                   # 1 second
        print(humanize_time_delta(60 + 1))                              # 1 minute 1 second
        print(humanize_time_delta(87.6))                                # 1 minute 27 seconds
        print(humanize_time_delta(0.01))                                # 0.01 seconds
        print(humanize_time_delta(60 * 60 + 1))                         # 1 hour 1 second
        print(humanize_time_delta(60 * 60 * 24 + 1))                    # 1 day 1 second
        print(humanize_time_delta(60 * 60 * 24 + 60 * 2 + 60*60*9 + 3)) # 1 day 9 hours 2 minutes 3 seconds
    """
    if sec < 0:
        logger.warn("humanize_time_delta() obtains negative seconds!")
        return "{:.3g} seconds".format(sec)
    if sec == 0:
        return "0 second"
    time = datetime(2000, 1, 1) + timedelta(seconds=int(sec))
    units = ['day', 'hour', 'minute', 'second']
    vals = [int(sec // 86400), time.hour, time.minute, time.second]
    if sec < 60:
        vals[-1] = sec

    def _format(v, u):
        return "{:.3g} {}{}".format(v, u, "s" if v > 1 else "")

    ans = []
    for v, u in zip(vals, units):
        if v > 0:
            ans.append(_format(v, u))
    return " ".join(ans)

def _pick_tqdm_interval(file):
    # Heuristics to pick a update interval for progress bar that's nice-looking for users.
    isatty = file.isatty()
    # Jupyter notebook should be recognized as tty.
    # Wait for https://github.com/ipython/ipykernel/issues/268
    try:
        from ipykernel import iostream
        if isinstance(file, iostream.OutStream):
            isatty = True
    except ImportError:
        pass

    if isatty:
        return 0.5
    else:
        # When run under mpirun/slurm, isatty is always False.
        # Here we apply some hacky heuristics for slurm.
        if 'SLURM_JOB_ID' in os.environ:
            if int(os.environ.get('SLURM_JOB_NUM_NODES', 1)) > 1:
                # multi-machine job, probably not interactive
                return 60
            else:
                # possibly interactive, so let's be conservative
                return 15

        if 'OMPI_COMM_WORLD_SIZE' in os.environ:
            return 60

        # If not a tty, don't refresh progress bar that often
        return 180


def get_tqdm_kwargs(**kwargs):
    """
    Return default arguments to be used with tqdm.

    Args:
        kwargs: extra arguments to be used.
    Returns:
        dict:
    """
    default = dict(
        smoothing=0.5,
        dynamic_ncols=True,
        ascii=True,
        bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]'
    )

    try:
        # Use this env var to override the refresh interval setting
        interval = float(os.environ['TENSORPACK_PROGRESS_REFRESH'])
    except KeyError:
        interval = _pick_tqdm_interval(kwargs.get('file', sys.stderr))

    default['mininterval'] = interval
    default.update(kwargs)
    return default

class StoppableThread(threading.Thread):
    """
    A thread that has a 'stop' event.
    """

    def __init__(self, evt=None):
        """
        Args:
            evt(threading.Event): if None, will create one.
        """
        super(StoppableThread, self).__init__()
        if evt is None:
            evt = threading.Event()
        self._stop_evt = evt

    def stop(self):
        """ Stop the thread"""
        self._stop_evt.set()

    def stopped(self):
        """
        Returns:
            bool: whether the thread is stopped or not
        """
        return self._stop_evt.isSet()

    def queue_put_stoppable(self, q, obj):
        """ Put obj to queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                q.put(obj, timeout=5)
                break
            except queue.Full:
                pass

    def queue_get_stoppable(self, q):
        """ Take obj from queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                return q.get(timeout=5)
            except queue.Empty:
                pass


@memoized
def log_once(message, func='info'):
    """
    Log certain message only once. Call this function more than one times with
    the same message will result in no-op.

    Args:
        message(str): message to log
        func(str): the name of the logger method. e.g. "info", "warn", "error".
    """
    getattr(logger, func)(message)


@contextmanager
def timed_operation(msg, log_start=False):
    """
    Surround a context with a timer.

    Args:
        msg(str): the log to print.
        log_start(bool): whether to print also at the beginning.

    Example:
        .. code-block:: python

            with timed_operation('Good Stuff'):
                time.sleep(1)

        Will print:

        .. code-block:: python

            Good stuff finished, time:1sec.
    """
    if log_start:
        logger.info('Start {} ...'.format(msg))
    start = time()
    yield
    logger.info('{} finished, time:{:.4f}sec.'.format(
        msg, time() - start))


class ShareSessionThread(threading.Thread):
    """ A wrapper around thread so that the thread
        uses the default session at "start()" time.
    """
    def __init__(self, th=None):
        """
        Args:
            th (threading.Thread or None):
        """
        super(ShareSessionThread, self).__init__()
        if th is not None:
            assert isinstance(th, threading.Thread), th
            self._th = th
            self.name = th.name
            self.daemon = th.daemon

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield self._sess
        else:
            logger.warn("ShareSessionThread {} wasn't under a default session!".format(self.name))
            yield None

    def start(self):
        import tensorflow.compat.v1 as tf
        self._sess = tf.get_default_session()
        super(ShareSessionThread, self).start()

    def run(self):
        if not self._th:
            raise NotImplementedError()
        with self._sess.as_default():
            self._th.run()


def call_only_once(func):
    """
    Decorate a method or property of a class, so that this method can only
    be called once for every instance.
    Calling it more than once will result in exception.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # cannot use hasattr here, because hasattr tries to getattr, which
        # fails if func is a property
        assert func.__name__ in dir(self), "call_only_once can only be used on method or property!"

        if not hasattr(self, '_CALL_ONLY_ONCE_CACHE'):
            cache = self._CALL_ONLY_ONCE_CACHE = set()
        else:
            cache = self._CALL_ONLY_ONCE_CACHE

        cls = type(self)
        # cannot use ismethod(), because decorated method becomes a function
        is_method = inspect.isfunction(getattr(cls, func.__name__))
        assert func not in cache, \
            "{} {}.{} can only be called once per object!".format(
                'Method' if is_method else 'Property',
                cls.__name__, func.__name__)
        cache.add(func)

        return func(*args, **kwargs)

    return wrapper


def is_main_thread():
    if six.PY2:
        return isinstance(threading.current_thread(), threading._MainThread)
    else:
        # a nicer solution with py3
        return threading.current_thread() == threading.main_thread()


@contextmanager
def mask_sigint():
    """
    Returns:
        If called in main thread, returns a context where ``SIGINT`` is ignored, and yield True.
        Otherwise yield False.
    """
    if is_main_thread():
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        yield True
        signal.signal(signal.SIGINT, sigint_handler)
    else:
        yield False


def start_proc_mask_signal(proc):
    """
    Start process(es) with SIGINT ignored.

    Args:
        proc: (multiprocessing.Process or list)

    Note:
        The signal mask is only applied when called from main thread.
    """
    if not isinstance(proc, list):
        proc = [proc]

    with mask_sigint():
        for p in proc:
            p.start()


def enable_death_signal(_warn=True):
    """
    Set the "death signal" of the current process, so that
    the current process will be cleaned with guarantee
    in case the parent dies accidentally.
    """
    if platform.system() != 'Linux':
        return
    try:
        import prctl    # pip install python-prctl
    except ImportError:
        if _warn:
            log_once('"import prctl" failed! Install python-prctl so that processes can be cleaned with guarantee.',
                     'warn')
        return
    else:
        assert hasattr(prctl, 'set_pdeathsig'), \
            "prctl.set_pdeathsig does not exist! Note that you need to install 'python-prctl' instead of 'prctl'."
        # is SIGHUP a good choice?
        prctl.set_pdeathsig(signal.SIGHUP)


def HIDE_DOC(func):
    func.__HIDE_SPHINX_DOC__ = True
    return func


def building_rtfd():
    """
    Returns:
        bool: if tensorpack is being imported to generate docs now.
    """
    return os.environ.get('READTHEDOCS') == 'True' \
        or os.environ.get('DOC_BUILDING')


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


def mkdir_p(dirname):
    """ Like "mkdir -p", make a dir recursively, but do nothing if the dir exists

    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


# copied from https://stackoverflow.com/questions/2328339/how-to-generate-n-different-colors-for-any-natural-number-n
PALETTE_HEX = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94",
    "#7ED379", "#012C58"]


def _parse_hex_color(s):
    r = int(s[1:3], 16)
    g = int(s[3:5], 16)
    b = int(s[5:7], 16)
    return (r, g, b)


PALETTE_RGB = np.asarray(
    list(map(_parse_hex_color, PALETTE_HEX)),
    dtype='int32')

def subproc_call(cmd, timeout=None):
    """
    Execute a command with timeout, and return both STDOUT/STDERR.

    Args:
        cmd(str): the command to execute.
        timeout(float): timeout in seconds.

    Returns:
        output(bytes), retcode(int). If timeout, retcode is -1.
    """
    try:
        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT,
            shell=True, timeout=timeout)
        return output, 0
    except subprocess.TimeoutExpired as e:
        logger.warn("Command '{}' timeout!".format(cmd))
        logger.warn(e.output.decode('utf-8'))
        return e.output, -1
    except subprocess.CalledProcessError as e:
        logger.warn("Command '{}' failed, return code={}".format(cmd, e.returncode))
        logger.warn(e.output.decode('utf-8'))
        return e.output, e.returncode
    except Exception:
        logger.warn("Command '{}' failed to run.".format(cmd))
        return "", -2

class NvmlException(Exception):
    def __init__(self, error_code):
        super(NvmlException, self).__init__(error_code)
        self.error_code = error_code

    def __str__(self):
        return NvmlErrorCodes[str(self.error_code)]


def _check_return(ret):
    if (ret != 0):
        raise NvmlException(ret)
    return ret


class NVML(object):
    """
    Loader for libnvidia-ml.so
    """

    _nvmlLib = None
    _lib_lock = threading.Lock()

    def load(self):
        with self._lib_lock:
            if self._nvmlLib is None:
                self._nvmlLib = CDLL("libnvidia-ml.so.1")

                function_pointers = ["nvmlDeviceGetName", "nvmlDeviceGetUUID", "nvmlDeviceGetMemoryInfo",
                                     "nvmlDeviceGetUtilizationRates", "nvmlInit_v2", "nvmlShutdown",
                                     "nvmlDeviceGetCount_v2", "nvmlDeviceGetHandleByIndex_v2"]

                self.func_ptr = {n: self._function_pointer(n) for n in function_pointers}

    def _function_pointer(self, name):
        try:
            return getattr(self._nvmlLib, name)
        except AttributeError:
            raise NvmlException(NVML_ERROR_FUNCTION_NOT_FOUND)

    def get_function(self, name):
        if name in self.func_ptr.keys():
            return self.func_ptr[name]


_NVML = NVML()

class NVMLContext(object):
    """Creates a context to query information

    Example:

        with NVMLContext() as ctx:
            num_gpus = ctx.num_devices()
            for device in ctx.devices():
                print(device.memory())
                print(device.utilization())

    """
    def __enter__(self):
        """Create a new context """
        _NVML.load()
        _check_return(_NVML.get_function("nvmlInit_v2")())
        return self

    def __exit__(self, type, value, tb):
        """Destroy current context"""
        _check_return(_NVML.get_function("nvmlShutdown")())

    def num_devices(self):
        """Get number of devices """
        c_count = c_uint()
        _check_return(_NVML.get_function(
            "nvmlDeviceGetCount_v2")(byref(c_count)))
        return c_count.value

    def devices(self):
        """
        Returns:
            [NvidiaDevice]: a list of devices
        """
        return [self.device(i) for i in range(self.num_devices())]

    def device(self, idx):
        """Get a specific GPU device

        Args:
            idx: index of device

        Returns:
            NvidiaDevice: single GPU device
        """

        class GpuDevice(Structure):
            pass

        c_nvmlDevice_t = POINTER(GpuDevice)

        c_index = c_uint(idx)
        device = c_nvmlDevice_t()
        _check_return(_NVML.get_function(
            "nvmlDeviceGetHandleByIndex_v2")(c_index, byref(device)))
        return NvidiaDevice(device)


def get_num_gpu():
    """
    Returns:
        int: #available GPUs in CUDA_VISIBLE_DEVICES, or in the system.
    """

    def warn_return(ret, message):
        try:
            import tensorflow.compat.v1 as tf
        except ImportError:
            return ret

        built_with_cuda = tf.test.is_built_with_cuda()
        if not built_with_cuda and ret > 0:
            logger.warn(message + "But TensorFlow was not built with CUDA support!")
        return ret

    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env is not None:
        return warn_return(len(env.split(',')), "Found non-empty CUDA_VISIBLE_DEVICES. ")
    output, code = subproc_call("nvidia-smi -L", timeout=5)
    if code == 0:
        output = output.decode('utf-8')
        return warn_return(len(output.strip().split('\n')), "Found nvidia-smi. ")
    try:
        # Use NVML to query device properties
        with NVMLContext() as ctx:
            return warn_return(ctx.num_devices(), "NVML found nvidia devices. ")
    except Exception:
        # Fallback
        # Note this will initialize all GPUs and therefore has side effect
        # https://github.com/tensorflow/tensorflow/issues/8136
        logger.info("Loading local devices by TensorFlow ...")
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return len([x.name for x in local_device_protos if x.device_type == 'GPU'])
