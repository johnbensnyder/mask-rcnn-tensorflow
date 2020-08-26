NUM_GPU=8
BATCH_SIZE_PER_GPU=6
THROUGHPUT_LOG_FREQ=2000

cd /workspace/shared_workspace/mask-rcnn-tensorflow

mpirun -np ${NUM_GPU} \
--H localhost:${NUM_GPU} \
--allow-run-as-root \
--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_exclude lo,docker0 \
-mca btl_vader_single_copy_mechanism none \
-x LD_LIBRARY_PATH \
-x PATH \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_MIN_NRINGS=8 \
-x NCCL_ALGO=Ring \
-x NCCL_DEBUG=INFO \
-x TENSORPACK_FP16=1 \
-x TF_CUDNN_USE_AUTOTUNE=0 \
-x TF_ENABLE_NHWC=0 \
-x HOROVOD_CYCLE_TIME=0.5 \
-x HOROVOD_FUSION_THRESHOLD=67108864 \
--output-filename /workspace/shared_workspace/logs/mpirun_logs \
python3 /workspace/shared_workspace/mask-rcnn-tensorflow/MaskRCNN/train.py \
--logdir /workspace/shared_workspace/logs/train_log \
--fp16 \
--throughput_log_freq ${THROUGHPUT_LOG_FREQ} \
--config \
MODE_MASK=True \
MODE_FPN=True \
DATA.BASEDIR=/workspace/shared_workspace/data/coco/coco \
DATA.TRAIN='["train2017"]' \
DATA.VAL='("val2017",)' \
TRAIN.BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU} \
TRAIN.LR_EPOCH_SCHEDULE='[(8, 0.1), (10, 0.01), (12, None)]' \
TRAIN.EVAL_PERIOD=1 \
TRAIN.BACKBONE_NCHW=True \
TRAIN.FPN_NCHW=True \
TRAIN.RPN_NCHW=True \
TRAIN.MASK_NCHW=True \
RPN.TOPK_PER_IMAGE=True \
PREPROC.PREDEFINED_PADDING=False \
BACKBONE.WEIGHTS=/workspace/shared_workspace/data/weights/ImageNet-R50-AlignPadding.npz \
BACKBONE.NORM=FreezeBN \
TRAIN.WARMUP_INIT_LR=0.000416666666667 \
FRCNN.BBOX_REG_WEIGHTS='[20., 20., 10., 10.]' \
TRAINER=horovod