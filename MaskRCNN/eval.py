# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
# File: eval.py

import itertools
import sys
import os
import json
import numpy as np
from numba import jit
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import cv2
import pycocotools.mask as cocomask
import tqdm
import tensorflow.compat.v1 as tf
from scipy import interpolate

from tensorpack_callbacks import Callback
from tensorpack_tfutils import get_tf_version_tuple, TrainContext
import tensorpack_logger as logger
from tensorpack_utils import get_tqdm

from common import CustomResize, clip_boxes
from data import get_eval_dataflow, get_batched_eval_dataflow
from dataset import DetectionDataset
from config import config as cfg

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass


DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""

def _scale_box(box, scale):
    w_half = (box[2] - box[0]) * 0.5
    h_half = (box[3] - box[1]) * 0.5
    x_c = (box[2] + box[0]) * 0.5
    y_c = (box[3] + box[1]) * 0.5

    w_half *= scale
    h_half *= scale

    scaled_box = np.zeros_like(box)
    scaled_box[0] = x_c - w_half
    scaled_box[2] = x_c + w_half
    scaled_box[1] = y_c - h_half
    scaled_box[3] = y_c + h_half
    return scaled_box

#@jit
def _paste_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    Returns:
        A uint8 binary image of hxw.
    """
    assert mask.shape[0] == mask.shape[1], mask.shape
    if not cfg.RPN.SLOW_ACCURATE_MASK:
        # This method (inspired by Detectron) is less accurate but fast.
        # int() is floor
        # box fpcoor=0.0 -> intcoor=0.0
        x0, y0 = list(map(int, box[:2] + 0.5))
        # box fpcoor=h -> intcoor=h-1, inclusive
        x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
        x1 = max(x0, x1)    # require at least 1x1
        y1 = max(y0, y1)

        w = x1 + 1 - x0
        h = y1 + 1 - y0

        # rounding errors could happen here, because masks were not originally computed for this shape.
        # but it's hard to do better, because the network does not know the "original" scale
        mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
        ret = np.zeros(shape, dtype='uint8')
        ret[y0:y1 + 1, x0:x1 + 1] = mask
        return ret
    else:
        # This method is accurate but much slower.
        mask = np.pad(mask, [(1, 1), (1, 1)], mode='constant')
        box = _scale_box(box, float(mask.shape[0]) / (mask.shape[0] - 2))

        mask_pixels = np.arange(0.0, mask.shape[0]) + 0.5
        mask_continuous = interpolate.interp2d(mask_pixels, mask_pixels, mask, fill_value=0.0)
        h, w = shape
        ys = np.arange(0.0, h) + 0.5
        xs = np.arange(0.0, w) + 0.5
        ys = (ys - box[1]) / (box[3] - box[1]) * mask.shape[0]
        xs = (xs - box[0]) / (box[2] - box[0]) * mask.shape[1]
        res = mask_continuous(xs, ys)
        return (res >= 0.5).astype('uint8')


def predict_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, *masks = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    if masks:
        # has mask
        full_masks = [_paste_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    return results

def predict_image_batch(img_batch, model_func, resized_sizes, scales, orig_sizes):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    resized_sizes = np.stack(resized_sizes)
    resized_sizes_in = np.concatenate((resized_sizes, 3*np.ones((resized_sizes.shape[0], 1))), axis=1)

    indices, boxes, probs, labels, *masks = model_func(img_batch, resized_sizes_in)

    results = []
    for i in range(len(scales)):
        ind = np.where(indices.astype(np.int32) == i)[0]

        if len(ind) > 0:
            boxes[ind, :] = boxes[ind, :]/scales[i]

            # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
            boxes[ind, :] = clip_boxes(boxes[ind, :], orig_sizes[i])

        if masks and len(ind) > 0:
           # has mask
           full_masks = [_paste_mask(box, mask, orig_sizes[i])
                      for box, mask in zip(boxes[ind,:], masks[0][ind,:])]
           masks = full_masks
        else:
           # fill with none
           masks = [None] * len(boxes[ind,:])

    results.append([DetectionResult(*args) for args in zip(boxes, probs, labels, masks)])
    return results


def predict_dataflow(df, model_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.
    Returns:
        list of dict, in the format used by
        `DetectionDataset.eval_or_save_inference_results`
    """
    df.reset_state()
    all_results = []
    with ExitStack() as stack:
        # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(get_tqdm(total=df.size()))
        for img, img_id in df:
            results = predict_image(img, model_func)
            for r in results:

                img_id = int(img_id)
                class_id = int(r.class_id)
                bbox = list([float(b) for b in r.box])
                score = round(float(r.score), 4)

#                 print("A result")
#                 print(f'image_id [{type(img_id)}] {img_id}')
#                 print(f'class_id [{type(class_id)}] {class_id}')
#                 print(f'bbox [{type(bbox)}] {bbox}')
#                 print(f'bbox[0] [{type(bbox[0])}] {bbox[0]}')
#                 print(f'score [{type(score)}] {score}')

                res = {
                    'image_id': img_id,
                    'category_id': class_id,
                    'bbox': bbox,
                    'score': score,
                }

                # also append segmentation to results
                if r.mask is not None:
                    rle = cocomask.encode(
                        np.array(r.mask[:, :, None], order='F'))[0]
                    rle['counts'] = rle['counts'].decode('ascii')
                    res['segmentation'] = rle
                all_results.append(res)
            tqdm_bar.update(1)
    return all_results



def predict_dataflow_batch(df, model_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.

    Returns:
        list of dict, in the format used by
        `DetectionDataset.eval_or_save_inference_results`
    """
    df.reset_state()
    all_results = []
    with ExitStack() as stack:
        # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(get_tqdm(total=df.size()))
        for imgs, img_ids, resized_sizes, scales, orig_sizes in df:
            results = predict_image_batch(imgs, model_func, resized_sizes, scales, orig_sizes)
            batch_id = 0
            for img_results in results:
                for r in img_results:

                    img_id = int(img_ids[batch_id])
                    class_id = int(r.class_id)
                    bbox = list([float(b) for b in r.box])
                    score = round(float(r.score), 4)

#                     print("A result")
#                     print(f'image_id [{type(img_id)}] {img_id}')
#                     print(f'class_id [{type(class_id)}] {class_id}')
#                     print(f'bbox [{type(bbox)}] {bbox}')
#                     print(f'bbox[0] [{type(bbox[0])}] {bbox[0]}')
#                     print(f'score [{type(score)}] {score}')

                    res = {
                        'image_id': img_id,
                        'category_id': class_id,
                        'bbox': bbox,
                        'score': score,
                    }

                    # also append segmentation to results
                    if r.mask is not None:
                        rle = cocomask.encode(
                            np.array(r.mask[:, :, None], order='F'))[0]
                        rle['counts'] = rle['counts'].decode('ascii')
                        res['segmentation'] = rle
                    all_results.append(res)
                batch_id += 1
            tqdm_bar.update(1)
    return all_results


def multithread_predict_dataflow(dataflows, model_funcs):
    """
    Running multiple `predict_dataflow` in multiple threads, and aggregate the results.

    Args:
        dataflows: a list of DataFlow to be used in :func:`predict_dataflow`
        model_funcs: a list of callable to be used in :func:`predict_dataflow`

    Returns:
        list of dict, in the format used by
        `DetectionDataset.eval_or_save_inference_results`
    """
    num_worker = len(model_funcs)
    assert len(dataflows) == num_worker
    if num_worker == 1:
        return predict_dataflow(dataflows[0], model_funcs[0])
    kwargs = {'thread_name_prefix': 'EvalWorker'} if sys.version_info.minor >= 6 else {}
    with ThreadPoolExecutor(max_workers=num_worker, **kwargs) as executor, \
            tqdm.tqdm(total=sum([df.size() for df in dataflows])) as pbar:
        futures = []
        for dataflow, pred in zip(dataflows, model_funcs):
            futures.append(executor.submit(predict_dataflow, dataflow, pred, pbar))
        all_results = list(itertools.chain(*[fut.result() for fut in futures]))
        return all_results

def gather_result_from_all_processes(local_results, root=0):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    res = comm.gather(local_results,root=root)
    return res


class AsyncEvaluator():
    '''
    An async evaluator used to submit coco evaluation job to a background thread

    Usage:
    1. create the worker with: worker = AsyncEvaluator()
    2. submit the job: work.submit_task(tag, background_task_fn, fn_inputs)
    '''
    def __init__(self, num_threads=1, device=None):
        self.num_threads = num_threads
        self.pool = ThreadPoolExecutor(num_threads)
        self.events = {}

    def submit_task(self, tag, fn, *args, **kwargs):
        e = self.pool.submit(fn, *args, **kwargs)
        self.events[tag] = e

    def task_done(self, tag):
        if tag in self.events.keys():
            return self.events[tag].done()
        else:
            return False

class OnlinePredictor(object):
    """ A predictor which directly use an existing session and given tensors.
    """

    ACCEPT_OPTIONS = False
    """ See Session.make_callable """

    sess = None
    """
    The tf.Session object associated with this predictor.
    """

    def __init__(self, input_tensors, output_tensors,
                 return_input=False, sess=None):
        """
        Args:
            input_tensors (list): list of names.
            output_tensors (list): list of names.
            return_input (bool): same as :attr:`PredictorBase.return_input`.
            sess (tf.Session): the session this predictor runs in. If None,
                will use the default session at the first call.
                Note that in TensorFlow, default session is thread-local.
        """
        self.return_input = return_input
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.sess = sess

        if sess is not None:
            self._callable = sess.make_callable(
                fetches=output_tensors,
                feed_list=input_tensors,
                accept_options=self.ACCEPT_OPTIONS)
        else:
            self._callable = None

    def __call__(self, *dp):
        """
        Call the predictor on some inputs.

        Example:
            When you have a predictor defined with two inputs, call it with:

            .. code-block:: python

                predictor(e1, e2)
        """
        output = self._do_call(dp)
        if self.return_input:
            return (dp, output)
        else:
            return output

    def _do_call(self, dp):
        assert len(dp) == len(self.input_tensors), \
            "{} != {}".format(len(dp), len(self.input_tensors))
        if self.sess is None:
            self.sess = tf.get_default_session()
            assert self.sess is not None, "Predictor isn't called under a default session!"

        if self._callable is None:
            self._callable = self.sess.make_callable(
                fetches=self.output_tensors,
                feed_list=self.input_tensors,
                accept_options=self.ACCEPT_OPTIONS)
        # run_metadata = tf.RunMetadata()
        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        return self._callable(*dp)

class EvalCallback(Callback):
    """
    A callback that runs evaluation once a while.
    It supports multi-gpu evaluation.

    Supoort the async evaluation:
    1. Running the graph on all gpus and gather the result on the master node
    2. Running a background thread to do the coco evaluation
    """

    _chief_only = False
    build_graph_func = None
    inputs_desc = None

    def __init__(self, eval_dataset, in_names, out_names, output_dir, batch_size, a_sync=False):
        self._eval_dataset = eval_dataset
        self._in_names, self._out_names = in_names, out_names
        self._output_dir = output_dir
        self.batched = batch_size > 0
        self.batch_size = batch_size
        self.async = a_sync

    def _setup_graph(self):
        num_gpu = cfg.TRAIN.NUM_GPUS
        if cfg.TRAINER == 'replicated':
            # TF bug in version 1.11, 1.12: https://github.com/tensorflow/tensorflow/issues/22750
            buggy_tf = get_tf_version_tuple() in [(1, 11), (1, 12)]

            # Use two predictor threads per GPU to get better throughput
            self.num_predictor = num_gpu if buggy_tf else num_gpu * 2
            self.predictors = [self._build_predictor(k % num_gpu) for k in range(self.num_predictor)]
            self.dataflows = [get_eval_dataflow(self._eval_dataset,
                                                shard=k, num_shards=self.num_predictor)
                              for k in range(self.num_predictor)]
        else:
            # Eval on all ranks and use gather
            self.predictor = self._build_predictor(0)

            if self.batched:
                self.dataflow = get_batched_eval_dataflow(self._eval_dataset,
                                              shard=hvd.rank(), num_shards=hvd.size(), batch_size=self.batch_size)
            else:
                self.dataflow = get_eval_dataflow(self._eval_dataset,
                                              shard=hvd.rank(), num_shards=hvd.size())


    def _build_predictor(self, idx):
        return self.get_predictor(self._in_names, self._out_names, device=idx)

    def get_predictor(self, input_names, output_names, device=0):
        pred_name = 'pred-{}'.format(device) if device >= 0 else 'tower-pred-cpu'
        device_id = device
        device = '/gpu:{}'.format(device_id) if device_id >= 0 else '/cpu:0'

        def get_tensor(name):
            name_with_ns = pred_name + "/" + name + ":0"
            G = tf.get_default_graph()
            return G.get_tensor_by_name(name_with_ns)

        input_tensors = [v.build_placeholder_reuse() for v in self.inputs_desc]
        input_tensor_names = {x.name: y for x, y in zip(self.inputs_desc, input_tensors)}
        with tf.variable_scope(tf.get_variable_scope(), reuse=True), tf.device(device), \
               TrainContext(pred_name, is_training=False):
            logger.info("Building pred graph on device {}...".format(device))
            self.build_graph_func(*input_tensors)
        eval_input_tensors = [input_tensor_names[x] for x in input_names]
        eval_output_tensors = [get_tensor(name) for name in output_names]
        predictor = OnlinePredictor(eval_input_tensors, eval_output_tensors)
        return predictor

    def _before_train(self):
        if hvd.rank() == 0 and self.async:
            self.worker = AsyncEvaluator()
        eval_period = cfg.TRAIN.EVAL_PERIOD
        self.epochs_to_eval = set()
        for k in itertools.count(1):
            if k * eval_period > self.trainer.max_epoch:
                break
            self.epochs_to_eval.add(k * eval_period)
        self.epochs_to_eval.add(self.trainer.max_epoch)
        logger.info("[EvalCallback] Will evaluate every {} epochs".format(eval_period))

    def _eval(self):
        logdir = self._output_dir
        if cfg.TRAINER == 'replicated':
            all_results = multithread_predict_dataflow(self.dataflows, self.predictors)
        else:
            if self.batched:
                local_results = predict_dataflow_batch(self.dataflow, self.predictor)
            else:
                local_results = predict_dataflow(self.dataflow, self.predictor)

            results = gather_result_from_all_processes(local_results)
            if hvd.rank() > 0:
                return
            all_results = []
            for item in results:
                if item is not None:
                    all_results.extend(item)

        if self.async:
            # define the async eval task
            def background_coco(all_results):
                output_file = os.path.join(
                    logdir, '{}-outputs{}'.format(self._eval_dataset, self.global_step))
                scores = DetectionDataset().eval_or_save_inference_results(
                    all_results, self._eval_dataset, output_file)
                cfg.TRAIN.SHOULD_STOP = scores['mAP(bbox)/IoU=0.5:0.95'] >= cfg.TEST.BOX_TARGET and scores['mAP(segm)/IoU=0.5:0.95'] >= cfg.TEST.MASK_TARGET
                for k, v in scores.items():
                    self.trainer.monitors.put_scalar(k, v)
                return

            self.worker.submit_task(f"eval_{self.epoch_num}", background_coco, all_results)
        else:
            output_file = os.path.join(
                logdir, '{}-outputs{}'.format(self._eval_dataset, self.global_step))
            scores = DetectionDataset().eval_or_save_inference_results(
                all_results, self._eval_dataset, output_file)
            for k, v in scores.items():
                self.trainer.monitors.put_scalar(k, v)

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            logger.info("Running evaluation ...")
            self._eval()
