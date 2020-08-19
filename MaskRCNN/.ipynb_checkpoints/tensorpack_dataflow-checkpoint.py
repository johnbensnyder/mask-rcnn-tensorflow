import sys
from contextlib import contextmanager
import inspect
import pprint
import threading
import atexit
import uuid
from abc import ABCMeta, abstractmethod
import six
from copy import copy
import multiprocessing as mp
import zmq
import tqdm
import numpy as np
from six.moves import zip, queue
import cv2
import os
import weakref

from tensorpack_utils import get_rng, StoppableThread, log_once, start_proc_mask_signal, enable_death_signal
from tensorpack_serialize import loads, dumps

def del_weakref(x):
    o = x()
    if o is not None:
        o.__del__()
        
def _bind_guard(sock, name):
    try:
        sock.bind(name)
    except zmq.ZMQError:
        logger.error(
            "ZMQError in socket.bind('{}'). Perhaps you're \
using pipes on a non-local file system. See documentation of PrefetchDataZMQ \
for more information.".format(name))
        raise

        
@contextmanager
def _zmq_catch_error(name):
    try:
        yield
    except zmq.ContextTerminated:
        logger.info("[{}] Context terminated.".format(name))
        raise DataFlowTerminated()
    except zmq.ZMQError as e:
        if e.errno == errno.ENOTSOCK:       # socket closed
            logger.info("[{}] Socket closed.".format(name))
            raise DataFlowTerminated()
        else:
            raise
    except Exception:
        raise

def _get_pipe_name(name):
    if sys.platform.startswith('linux'):
        # linux supports abstract sockets: http://api.zeromq.org/4-1:zmq-ipc
        pipename = "ipc://@{}-pipe-{}".format(name, str(uuid.uuid1())[:8])
        pipedir = os.environ.get('TENSORPACK_PIPEDIR', None)
        if pipedir is not None:
            logger.warn("TENSORPACK_PIPEDIR is not used on Linux any more! Abstract sockets will be used.")
    else:
        pipedir = os.environ.get('TENSORPACK_PIPEDIR', None)
        if pipedir is not None:
            logger.info("ZMQ uses TENSORPACK_PIPEDIR={}".format(pipedir))
        else:
            pipedir = '.'
        assert os.path.isdir(pipedir), pipedir
        filename = '{}/{}-pipe-{}'.format(pipedir.rstrip('/'), name, str(uuid.uuid1())[:6])
        assert not os.path.exists(filename), "Pipe {} exists! You may be unlucky.".format(filename)
        pipename = "ipc://{}".format(filename)
    return pipename        
        

def check_dtype(img):
    assert isinstance(img, np.ndarray), "[Augmentor] Needs an numpy array, but got a {}!".format(type(img))
    assert not isinstance(img.dtype, np.integer) or (img.dtype == np.uint8), \
        "[Augmentor] Got image of type {}, use uint8 or floating points instead!".format(img.dtype)

    
class DataFlowReentrantGuard(object):
    """
    A tool to enforce non-reentrancy.
    Mostly used on DataFlow whose :meth:`get_data` is stateful,
    so that multiple instances of the iterator cannot co-exist.
    """
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._succ = self._lock.acquire(False)
        if not self._succ:
            raise threading.ThreadError("This DataFlow is not reentrant!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False
    

class DataFlowMeta(ABCMeta):
    """
    DataFlow uses "__iter__()" and "__len__()" instead of
    "get_data()" and "size()". This add back-compatibility.
    """
    def __new__(mcls, name, bases, namespace, **kwargs):

        def hot_patch(required, existing):
            if required not in namespace and existing in namespace:
                namespace[required] = namespace[existing]

        hot_patch('__iter__', 'get_data')
        hot_patch('__len__', 'size')

        return ABCMeta.__new__(mcls, name, bases, namespace, **kwargs)

@six.add_metaclass(DataFlowMeta)
class DataFlow(object):
    """ Base class for all DataFlow """

    @abstractmethod
    def __iter__(self):
        """
        * A dataflow is an iterable. The :meth:`__iter__` method should yield a list each time.
          Each element in the list should be either a number or a numpy array.
          For now, tensorpack also **partially** supports dict instead of list.

        * The :meth:`__iter__` method can be either finite (will stop iteration) or infinite
          (will not stop iteration). For a finite dataflow, :meth:`__iter__` can be called
          again after the previous call returned.

        * For many dataflow, the :meth:`__iter__` method is non-reentrant, which means for an dataflow
          instance ``df``, :meth:`df.__iter__` cannot be called before the previous
          :meth:`df.__iter__` call has finished (iteration has stopped).
          If a dataflow is non-reentrant, :meth:`df.__iter__` should throw an exception if
          called before the previous call has finished.
          If you need to use the same dataflow in two places, you can simply create two dataflow instances.

        Yields:
            list: The datapoint, i.e. list of components.
        """

    def get_data(self):
        return self.__iter__()

    def __len__(self):
        """
        * A dataflow can optionally implement :meth:`__len__`. If not implemented, it will
          throw :class:`NotImplementedError`.

        * It returns an integer representing the size of the dataflow.
          The return value **may not be accurate or meaningful** at all.
          When it's accurate, it means that :meth:`__iter__` will always yield this many of datapoints.

        * There could be many reasons why :meth:`__len__` is inaccurate.
          For example, some dataflow has dynamic size.
          Some dataflow mixes the datapoints between consecutive passes over
          the dataset, due to parallelism and buffering.
          In this case it does not make sense to stop the iteration anywhere.

        * Due to the above reasons, the length is only a rough guidance. Inside
          tensorpack it's only used in these places:

          + A default ``steps_per_epoch`` in training, but you probably want to customize
            it yourself, especially when using data-parallel trainer.
          + The length of progress bar when processing a dataflow.
          + Used by :class:`InferenceRunner` to get the number of iterations in inference.
            In this case users are **responsible** for making sure that :meth:`__len__` is accurate.
            This is to guarantee that inference is run on a fixed set of images.

        Returns:
            int: rough size of this dataflow.

        Raises:
            :class:`NotImplementedError` if this DataFlow doesn't have a size.
        """
        raise NotImplementedError()

    def size(self):
        return self.__len__()

    def reset_state(self):
        """
        * It's guaranteed that :meth:`reset_state` should be called **once and only once**
          by the **process that uses the dataflow** before :meth:`__iter__` is called.
          The caller thread of this method should stay alive to keep this dataflow alive.

        * It is meant for certain initialization that involves processes,
          e.g., initialize random number generators (RNG), create worker processes.

          Because it's very common to use RNG in data processing,
          developers of dataflow can also subclass :class:`RNGDataFlow` to have easier access to an RNG.

        * A dataflow is not fork-safe after :meth:`reset_state` is called (because this will violate the guarantee).
          A few number of dataflow is not fork-safe anytime, which will be mentioned in the docs.

        * You should follow the above guarantee if you're using a dataflow yourself
          (either outside of tensorpack, or writing a wrapper dataflow)
        """
        pass

    
class RNGDataFlow(DataFlow):
    """ A DataFlow with RNG"""

    rng = None
    """
    ``self.rng`` is a ``np.random.RandomState`` instance that is initialized
    correctly in ``RNGDataFlow.reset_state()``.
    """

    def reset_state(self):
        """ Reset the RNG """
        self.rng = get_rng(self)


class DataFromList(RNGDataFlow):
    """ Wrap a list of datapoints to a DataFlow"""

    def __init__(self, lst, shuffle=True):
        """
        Args:
            lst (list): input list. Each element is a datapoint.
            shuffle (bool): shuffle data.
        """
        super(DataFromList, self).__init__()
        self.lst = lst
        self.shuffle = shuffle

    def __len__(self):
        return len(self.lst)

    def __iter__(self):
        if not self.shuffle:
            for k in self.lst:
                yield k
        else:
            idxs = np.arange(len(self.lst))
            self.rng.shuffle(idxs)
            for k in idxs:
                yield self.lst[k]


class ProxyDataFlow(DataFlow):
    """ Base class for DataFlow that proxies another.
        Every method is proxied to ``self.ds`` unless overriden by a subclass.
    """

    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): DataFlow to proxy.
        """
        self.ds = ds

    def reset_state(self):
        self.ds.reset_state()

    def __len__(self):
        return self.ds.__len__()

    def __iter__(self):
        return self.ds.__iter__()


class MapData(ProxyDataFlow):
    """
    Apply a mapper/filter on the datapoints of a DataFlow.

    Note:
        1. Please make sure func doesn't modify its arguments in place,
           unless you're certain it's safe.
        2. If you discard some datapoints, ``len(ds)`` will be incorrect.

    Example:

        .. code-block:: none

            ds = Mnist('train)
            ds = MapData(ds, lambda dp: [dp[0] * 255, dp[1]])
    """

    def __init__(self, ds, func):
        """
        Args:
            ds (DataFlow): input DataFlow
            func (datapoint -> datapoint | None): takes a datapoint and returns a new
                datapoint. Return None to discard/skip this datapoint.
        """
        super(MapData, self).__init__(ds)
        self.func = func

    def __iter__(self):
        for dp in self.ds:
            ret = self.func(copy(dp))  # shallow copy the list
            if ret is not None:
                yield ret


class MapDataComponent(MapData):
    """
    Apply a mapper/filter on a datapoint component.

    Note:
        1. This dataflow itself doesn't modify the datapoints.
           But please make sure func doesn't modify its arguments in place,
           unless you're certain it's safe.
        2. If you discard some datapoints, ``len(ds)`` will be incorrect.

    Example:

        .. code-block:: none

            ds = Mnist('train)
            ds = MapDataComponent(ds, lambda img: img * 255, 0)
    """
    def __init__(self, ds, func, index=0):
        """
        Args:
            ds (DataFlow): input DataFlow which produces either list or dict.
            func (TYPE -> TYPE|None): takes ``dp[index]``, returns a new value for ``dp[index]``.
                Return None to discard/skip this datapoint.
            index (int or str): index or key of the component.
        """
        self._index = index
        self._func = func
        super(MapDataComponent, self).__init__(ds, self._mapper)

    def _mapper(self, dp):
        r = self._func(dp[self._index])
        if r is None:
            return None
        dp = copy(dp)   # shallow copy to avoid modifying the datapoint
        if isinstance(dp, tuple):
            dp = list(dp)  # to be able to modify it in the next line
        dp[self._index] = r
        return dp
    
class _ParallelMapData(ProxyDataFlow):
    def __init__(self, ds, buffer_size, strict=False):
        super(_ParallelMapData, self).__init__(ds)
        assert buffer_size > 0, buffer_size
        self._buffer_size = buffer_size
        self._buffer_occupancy = 0  # actual #elements in buffer, only useful in strict mode
        self._strict = strict

    def reset_state(self):
        super(_ParallelMapData, self).reset_state()
        if not self._strict:
            ds = RepeatedData(self.ds, -1)
        else:
            ds = self.ds
        self._iter = ds.__iter__()

    def _recv(self):
        pass

    def _send(self, dp):
        pass

    def _recv_filter_none(self):
        ret = self._recv()
        assert ret is not None, \
            "[{}] Map function cannot return None when strict mode is used.".format(type(self).__name__)
        return ret

    def _fill_buffer(self, cnt=None):
        if cnt is None:
            cnt = self._buffer_size - self._buffer_occupancy
        try:
            for _ in range(cnt):
                dp = next(self._iter)
                self._send(dp)
        except StopIteration:
            raise RuntimeError(
                "[{}] buffer_size cannot be larger than the size of the DataFlow when strict=True!".format(
                    type(self).__name__))
        self._buffer_occupancy += cnt

    def get_data_non_strict(self):
        for dp in self._iter:
            self._send(dp)
            ret = self._recv()
            if ret is not None:
                yield ret

    def get_data_strict(self):
        self._fill_buffer()
        for dp in self._iter:
            self._send(dp)
            yield self._recv_filter_none()
        self._iter = self.ds.__iter__()   # refresh

        # first clear the buffer, then fill
        for k in range(self._buffer_size):
            dp = self._recv_filter_none()
            self._buffer_occupancy -= 1
            if k == self._buffer_size - 1:
                self._fill_buffer()
            yield dp

    def __iter__(self):
        if self._strict:
            for dp in self.get_data_strict():
                yield dp
        else:
            for dp in self.get_data_non_strict():
                yield dp
                
class MultiThreadMapData(_ParallelMapData):
    """
    Same as :class:`MapData`, but start threads to run the mapping function.
    This is useful when the mapping function is the bottleneck, but you don't
    want to start processes for the entire dataflow pipeline.

    Note:
        1. There is tiny communication overhead with threads, but you
           should avoid starting many threads in your main process to reduce GIL contention.

           The threads will only start in the process which calls :meth:`reset_state()`.
           Therefore you can use ``PrefetchDataZMQ(MultiThreadMapData(...), 1)``
           to reduce GIL contention.

        2. Threads run in parallel and can take different time to run the
           mapping function. Therefore the order of datapoints won't be
           preserved, and datapoints from one pass of `df.__iter__()` might get
           mixed with datapoints from the next pass.

           You can use **strict mode**, where `MultiThreadMapData.__iter__()`
           is guaranteed to produce the exact set which `df.__iter__()`
           produces. Although the order of data still isn't preserved.

           The behavior of strict mode is undefined if the dataflow is infinite.
    """
    class _Worker(StoppableThread):
        def __init__(self, inq, outq, evt, map_func):
            super(MultiThreadMapData._Worker, self).__init__(evt)
            self.inq = inq
            self.outq = outq
            self.func = map_func
            self.daemon = True

        def run(self):
            try:
                while True:
                    dp = self.queue_get_stoppable(self.inq)
                    if self.stopped():
                        return
                    # cannot ignore None here. will lead to unsynced send/recv
                    obj = self.func(dp)
                    self.queue_put_stoppable(self.outq, obj)
            except Exception:
                if self.stopped():
                    pass        # skip duplicated error messages
                else:
                    raise
            finally:
                self.stop()

    def __init__(self, ds, nr_thread, map_func, buffer_size=200, strict=False):
        """
        Args:
            ds (DataFlow): the dataflow to map
            nr_thread (int): number of threads to use
            map_func (callable): datapoint -> datapoint | None. Return None to
                discard/skip the datapoint.
            buffer_size (int): number of datapoints in the buffer
            strict (bool): use "strict mode", see notes above.
        """
        super(MultiThreadMapData, self).__init__(ds, buffer_size, strict)

        self._strict = strict
        self.nr_thread = nr_thread
        self.map_func = map_func
        self._threads = []
        self._evt = None

    def reset_state(self):
        super(MultiThreadMapData, self).reset_state()
        if self._threads:
            self._threads[0].stop()
            for t in self._threads:
                t.join()

        self._in_queue = queue.Queue()
        self._out_queue = queue.Queue()
        self._evt = threading.Event()
        self._threads = [MultiThreadMapData._Worker(
            self._in_queue, self._out_queue, self._evt, self.map_func)
            for _ in range(self.nr_thread)]
        for t in self._threads:
            t.start()

        self._guard = DataFlowReentrantGuard()

        # Call once at the beginning, to ensure inq+outq has a total of buffer_size elements
        self._fill_buffer()

    def _recv(self):
        return self._out_queue.get()

    def _send(self, dp):
        self._in_queue.put(dp)

    def __iter__(self):
        with self._guard:
            for dp in super(MultiThreadMapData, self).__iter__():
                yield dp

    def __del__(self):
        if self._evt is not None:
            self._evt.set()
        for p in self._threads:
            p.stop()
            p.join(timeout=5.0)
            # if p.is_alive():
            #     logger.warn("Cannot join thread {}.".format(p.name))

class _MultiProcessZMQDataFlow(DataFlow):
    def __init__(self):
        assert os.name != 'nt', "ZMQ IPC doesn't support windows!"
        self._reset_done = False
        self._procs = []

    def reset_state(self):
        """
        All forked dataflows should only be reset **once and only once** in spawned processes.
        Subclasses should call this method with super.
        """
        assert not self._reset_done, "reset_state() was called twice! This violates the API of DataFlow!"
        self._reset_done = True

        # __del__ not guaranteed to get called at exit
        atexit.register(del_weakref, weakref.ref(self))

    def _start_processes(self):
        start_proc_mask_signal(self._procs)

    def __del__(self):
        try:
            if not self._reset_done:
                return
            if not self.context.closed:
                self.socket.close(0)
                self.context.destroy(0)
            for x in self._procs:
                x.terminate()
                x.join(5)
            print("{} successfully cleaned-up.".format(type(self).__name__))
        except Exception:
            pass


class MultiProcessMapDataZMQ(_ParallelMapData, _MultiProcessZMQDataFlow):
    """
    Same as :class:`MapData`, but start processes to run the mapping function,
    and communicate with ZeroMQ pipe.

    Note:
        1. Processes run in parallel and can take different time to run the
           mapping function. Therefore the order of datapoints won't be
           preserved, and datapoints from one pass of `df.__iter__()` might get
           mixed with datapoints from the next pass.

           You can use **strict mode**, where `MultiProcessMapData.__iter__()`
           is guaranteed to produce the exact set which `df.__iter__()`
           produces. Although the order of data still isn't preserved.

           The behavior of strict mode is undefined if the dataflow is infinite.
    """
    class _Worker(mp.Process):
        def __init__(self, identity, map_func, pipename, hwm):
            super(MultiProcessMapDataZMQ._Worker, self).__init__()
            self.identity = identity
            self.map_func = map_func
            self.pipename = pipename
            self.hwm = hwm

        def run(self):
            enable_death_signal(_warn=self.identity == b'0')
            ctx = zmq.Context()
            socket = ctx.socket(zmq.REP)
            socket.setsockopt(zmq.IDENTITY, self.identity)
            socket.set_hwm(self.hwm)
            socket.connect(self.pipename)

            while True:
                dp = loads(socket.recv(copy=False))
                dp = self.map_func(dp)
                socket.send(dumps(dp), copy=False)

    def __init__(self, ds, nr_proc, map_func, buffer_size=200, strict=False):
        """
        Args:
            ds (DataFlow): the dataflow to map
            nr_proc(int): number of threads to use
            map_func (callable): datapoint -> datapoint | None. Return None to
                discard/skip the datapoint.
            buffer_size (int): number of datapoints in the buffer
            strict (bool): use "strict mode", see notes above.
        """
        _ParallelMapData.__init__(self, ds, buffer_size, strict)
        _MultiProcessZMQDataFlow.__init__(self)
        self.nr_proc = nr_proc
        self.map_func = map_func
        self._strict = strict
        self._procs = []
        self._guard = DataFlowReentrantGuard()

    def reset_state(self):
        _MultiProcessZMQDataFlow.reset_state(self)
        _ParallelMapData.reset_state(self)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.set_hwm(self._buffer_size * 2)
        pipename = _get_pipe_name('dataflow-map')
        _bind_guard(self.socket, pipename)

        self._proc_ids = [u'{}'.format(k).encode('utf-8') for k in range(self.nr_proc)]
        worker_hwm = int(self._buffer_size * 2 // self.nr_proc)
        self._procs = [MultiProcessMapDataZMQ._Worker(
            self._proc_ids[k], self.map_func, pipename, worker_hwm)
            for k in range(self.nr_proc)]

        self._start_processes()
        self._fill_buffer()     # pre-fill the bufer

    def _send(self, dp):
        msg = [b"", dumps(dp)]
        self.socket.send_multipart(msg, copy=False)
    
    def _recv(self):
        msg = self.socket.recv_multipart(copy=False)
        dp = loads(msg[1])
        return dp
    
    def __iter__(self):
        with self._guard, _zmq_catch_error('MultiProcessMapData'):
            for dp in super(MultiProcessMapDataZMQ, self).__iter__():
                yield dp


class TestDataSpeed(ProxyDataFlow):
    """ Test the speed of some DataFlow """
    def __init__(self, ds, size=5000, warmup=0):
        """
        Args:
            ds (DataFlow): the DataFlow to test.
            size (int): number of datapoints to fetch.
            warmup (int): warmup iterations
        """
        super(TestDataSpeed, self).__init__(ds)
        self.test_size = int(size)
        self.warmup = int(warmup)

    def __iter__(self):
        """ Will run testing at the beginning, then produce data normally. """
        self.start_test()
        for dp in self.ds:
            yield dp

    def start_test(self):
        log_deprecated("TestDataSpeed.start_test() was renamed to start()", "2019-03-30")
        self.start()

    def start(self):
        """
        Start testing with a progress bar.
        """
        self.ds.reset_state()
        itr = self.ds.__iter__()
        if self.warmup:
            for _ in tqdm.trange(self.warmup, **get_tqdm_kwargs()):
                next(itr)
        # add smoothing for speed benchmark
        with get_tqdm(total=self.test_size,
                      leave=True, smoothing=0.2) as pbar:
            for idx, dp in enumerate(itr):
                pbar.update()
                if idx == self.test_size - 1:
                    break


@six.add_metaclass(ABCMeta)
class Augmentor(object):
    """ Base class for an augmentor"""

    def __init__(self):
        self.reset_state()

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self' and not k.startswith('_'):
                    setattr(self, k, v)

    def reset_state(self):
        """ reset rng and other state """
        self.rng = get_rng(self)

    def augment(self, d):
        """
        Perform augmentation on the data.

        Args:
            d: input data

        Returns:
            augmented data
        """
        d, params = self._augment_return_params(d)
        return d

    def augment_return_params(self, d):
        """
        Augment the data and return the augmentation parameters.
        If the augmentation is non-deterministic (random),
        the returned parameters can be used to augment another data with the identical transformation.
        This can be used for, e.g. augmenting image, masks, keypoints altogether with the
        same transformation.

        Returns:
            (augmented data, augmentation params)
        """
        return self._augment_return_params(d)

    def _augment_return_params(self, d):
        """
        Augment the image and return both image and params
        """
        prms = self._get_augment_params(d)
        return (self._augment(d, prms), prms)

    def augment_with_params(self, d, param):
        """
        Augment the data with the given param.

        Args:
            d: input data
            param: augmentation params returned by :meth:`augment_return_params`

        Returns:
            augmented data
        """
        return self._augment(d, param)

    @abstractmethod
    def _augment(self, d, param):
        """
        Augment with the given param and return the new data.
        The augmentor is allowed to modify data in-place.
        """

    def _get_augment_params(self, d):
        """
        Get the augmentor parameters.
        """
        return None

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return self.rng.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "imgaug.MyAugmentor(field1={self.field1}, field2={self.field2})"
        """
        try:
            argspec = inspect.getargspec(self.__init__)
            assert argspec.varargs is None, "The default __repr__ doesn't work for varargs!"
            assert argspec.keywords is None, "The default __repr__ doesn't work for kwargs!"
            fields = argspec.args[1:]
            index_field_has_default = len(fields) - (0 if argspec.defaults is None else len(argspec.defaults))

            classname = type(self).__name__
            argstr = []
            for idx, f in enumerate(fields):
                assert hasattr(self, f), \
                    "Attribute {} not found! Default __repr__ only works if attributes match the constructor.".format(f)
                attr = getattr(self, f)
                if idx >= index_field_has_default:
                    if attr is argspec.defaults[idx - index_field_has_default]:
                        continue
                argstr.append("{}={}".format(f, pprint.pformat(attr)))
            return "imgaug.{}({})".format(classname, ', '.join(argstr))
        except AssertionError as e:
            log_once(e.args[0], 'warn')
            return super(Augmentor, self).__repr__()

    __str__ = __repr__


class ImageAugmentor(Augmentor):
    """
    ImageAugmentor should take images of type uint8 in range [0, 255], or
    floating point images in range [0, 1] or [0, 255].
    """
    def augment_coords(self, coords, param):
        """
        Augment the coordinates given the param.

        By default, an augmentor keeps coordinates unchanged.
        If a subclass of :class:`ImageAugmentor` changes coordinates but couldn't implement this method,
        it should ``raise NotImplementedError()``.

        Args:
            coords: Nx2 floating point numpy array where each row is (x, y)
            param: augmentation params returned by :meth:`augment_return_params`

        Returns:
            new coords
        """
        return self._augment_coords(coords, param)

    def _augment_coords(self, coords, param):
        return coords


class AugmentorList(ImageAugmentor):
    """
    Augment an image by a list of augmentors
    """

    def __init__(self, augmentors):
        """
        Args:
            augmentors (list): list of :class:`ImageAugmentor` instance to be applied.
        """
        assert isinstance(augmentors, (list, tuple)), augmentors
        self.augmentors = augmentors
        super(AugmentorList, self).__init__()

    def _get_augment_params(self, img):
        # the next augmentor requires the previous one to finish
        raise RuntimeError("Cannot simply get all parameters of a AugmentorList without running the augmentation!")

    def _augment_return_params(self, img):
        check_dtype(img)
        assert img.ndim in [2, 3], img.ndim

        prms = []
        for a in self.augmentors:
            img, prm = a._augment_return_params(img)
            prms.append(prm)
        return img, prms

    def _augment(self, img, param):
        check_dtype(img)
        assert img.ndim in [2, 3], img.ndim
        for aug, prm in zip(self.augmentors, param):
            img = aug._augment(img, prm)
        return img

    def _augment_coords(self, coords, param):
        for aug, prm in zip(self.augmentors, param):
            coords = aug._augment_coords(coords, prm)
        return coords

    def reset_state(self):
        """ Will reset state of each augmentor """
        for a in self.augmentors:
            a.reset_state()


class Flip(ImageAugmentor):
    """
    Random flip the image either horizontally or vertically.
    """
    def __init__(self, horiz=False, vert=False, prob=0.5):
        """
        Args:
            horiz (bool): use horizontal flip.
            vert (bool): use vertical flip.
            prob (float): probability of flip.
        """
        super(Flip, self).__init__()
        if horiz and vert:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        elif horiz:
            self.code = 1
        elif vert:
            self.code = 0
        else:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        return (do, h, w)

    def _augment(self, img, param):
        do, _, _ = param
        if do:
            ret = cv2.flip(img, self.code)
            if img.ndim == 3 and ret.ndim == 2:
                ret = ret[:, :, np.newaxis]
        else:
            ret = img
        return ret

    def _augment_coords(self, coords, param):
        do, h, w = param
        if do:
            if self.code == 0:
                coords[:, 1] = h - coords[:, 1]
            elif self.code == 1:
                coords[:, 0] = w - coords[:, 0]
        return coords 
    
    
class RepeatedData(ProxyDataFlow):
    """ Take data points from another DataFlow and produce them until
        it's exhausted for certain amount of times. i.e.:
        dp1, dp2, .... dpn, dp1, dp2, ....dpn
    """

    def __init__(self, ds, nr):
        """
        Args:
            ds (DataFlow): input DataFlow
            nr (int): number of times to repeat ds.
                Set to -1 to repeat ``ds`` infinite times.
        """
        self.nr = nr
        super(RepeatedData, self).__init__(ds)

    def __len__(self):
        """
        Raises:
            :class:`ValueError` when nr == -1.
        """
        if self.nr == -1:
            raise NotImplementedError("__len__() is unavailable for infinite dataflow")
        return len(self.ds) * self.nr

    def __iter__(self):
        if self.nr == -1:
            while True:
                for dp in self.ds:
                    yield dp
        else:
            for _ in range(self.nr):
                for dp in self.ds:
                    yield dp