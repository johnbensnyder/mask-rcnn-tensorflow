A few small changes are need to TF to compile these

In 

/usr/local/lib/python3.6/dist-packages/tensorflow/include/tensorflow/core/util/gpu_device_functions.h

on lines 34 and 35, change

#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cuda.h"

to

#include "cuda/include/cuComplex.h"
#include "cuda/include/cuda.h"

In 

/usr/local/lib/python3.6/dist-packages/tensorflow/include/tensorflow/core/util/gpu_kernel_helper.h

Change line 22

#include "third_party/gpus/cuda/include/cuda_fp16.h"

to

#include "cuda/include/cuda_fp16.h"

Finally, create a symbolic link for TF libraries

ln -s /usr/local/lib/python3.6/dist-packages/tensorflow/libtensorflow_framework.so.2 \
       /usr/local/lib/python3.6/dist-packages/tensorflow/libtensorflow_framework.so
       
Then run make.

functions are imported in 3 files, _roi_align_op_grad.py, rpn.py and fpn.py using

module = '/workspace/shared_workspace/mask-rcnn-tensorflow/MaskRCNN/model/custom_ops/roi_align/roi_align_op.so'

gen_custom_op = tf.load_op_library(module)

Then can be called like normal python TF function,

rois, rois_probs = gen_custom_op.generate_bounding_box_proposals_v1(scores,
                                bbox_deltas,
                                im_info,
                                single_level_anchor_boxes,
                                spatial_scale=1.0 / cfg.FPN.ANCHOR_STRIDES[lvl],
                                pre_nms_topn=fpn_nms_topk,
                                post_nms_topn=fpn_nms_topk,
                                nms_threshold=cfg.RPN.PROPOSAL_NMS_THRESH,
                                min_size=cfg.RPN.MIN_SIZE) 