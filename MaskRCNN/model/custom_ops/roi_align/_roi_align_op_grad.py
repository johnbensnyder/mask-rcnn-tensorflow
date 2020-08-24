import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
import os

module = '/mask-rcnn-tensorflow/MaskRCNN/model/custom_ops/roi_align/roi_align_op.so'

gen_roi_align_op = tf.load_op_library(module)

@ops.RegisterGradient("ROIAlign")
def _ROIAlignGrad(op, grad):
    """The derivatives for ROIAlign.
    
    
      Args:
        op: The ROIAlign op.
        grad: The tensor representing the gradient w.r.t. the output.
    
      Returns:
        The gradients w.r.t. the input image, ROIS, as well as the always-None
        gradients w.r.t. box_ind and crop_size.
    """
    original_input = op.inputs[0]
    rois = op.inputs[1]
    allowed_types = [dtypes.float16, dtypes.float32, dtypes.float64]
    #allowed_types = [dtypes.float32]
    if op.inputs[0].dtype in allowed_types:
        # pylint: disable=protected-access
        grad0 = gen_roi_align_op.roi_align_grad(
            grad, original_input, rois,
            spatial_scale=op.get_attr("spatial_scale"),
            pooled_height=op.get_attr("pooled_height"),
            pooled_width=op.get_attr("pooled_width"),
            sampling_ratio=op.get_attr("sampling_ratio"))
        # pylint: enable=protected-access
    else:
        grad0 = None
    # gradient wrt rois is 0
    return [grad0, None]

@ops.RegisterGradient("ROIAlignV2")
def _ROIAlignV2Grad(op, grad):
    """The derivatives for ROIAlign.
    
    
      Args:
        op: The ROIAlign op.
        grad: The tensor representing the gradient w.r.t. the output.
    
      Returns:
        The gradients w.r.t. the input image, ROIS, as well as the always-None
        gradients w.r.t. box_ind and crop_size.
    """
    original_input = op.inputs[0]
    rois = op.inputs[1]
    allowed_types = [dtypes.float16, dtypes.float32, dtypes.float64]
    #allowed_types = [dtypes.float32]
    if op.inputs[0].dtype in allowed_types:
        # pylint: disable=protected-access
        grad0 = gen_roi_align_op.roi_align_v2_grad(
            grad, original_input, rois,
            spatial_scale=op.get_attr("spatial_scale"),
            pooled_height=op.get_attr("pooled_height"),
            pooled_width=op.get_attr("pooled_width"),
            sampling_ratio=op.get_attr("sampling_ratio"),
            min_level=op.get_attr("min_level"),
            max_level=op.get_attr("max_level"),
            canonical_scale=op.get_attr("canonical_scale"),
            canonical_level=op.get_attr("canonical_level"),
            debug=op.get_attr("debug"),
            )
        # pylint: enable=protected-access
    else:
        grad0 = None
    # gradient wrt rois is 0
    return [grad0, None]
