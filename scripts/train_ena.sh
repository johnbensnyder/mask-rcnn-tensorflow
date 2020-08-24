NUM_GPU=16
BATCH_SIZE_PER_GPU=4
THROUGHPUT_LOG_FREQ=2000

cd /workspace/shared_workspace/mask-rcnn-tensorflow

mpirun --allow-run-as-root \
    --hostfile /root/.ssh/hosts \
    --mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 \
    --mca btl_vader_single_copy_mechanism none \
    --mca oob_tcp_if_include ens5 \
    --mca btl_tcp_if_include ens5 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_ALGO=Tree,Ring,Collnet \
    -x NCCL_MIN_NRINGS=8 \
    -x TENSORPACK_FP16=1 \
    -x TF_CUDNN_USE_AUTOTUNE=0 \
    -x TF_ENABLE_NHWC=1 \
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
    TRAIN.BACKBONE_NCHW=False \
    TRAIN.FPN_NCHW=False \
    TRAIN.RPN_NCHW=False \
    TRAIN.MASK_NCHW=False \
    RPN.TOPK_PER_IMAGE=True \
    PREPROC.PREDEFINED_PADDING=False \
    BACKBONE.WEIGHTS=/workspace/shared_workspace/data/weights/ImageNet-R50-AlignPadding.npz \
    BACKBONE.NORM=FreezeBN \
    TRAIN.WARMUP_INIT_LR=0.000416666666667 \
    FRCNN.BBOX_REG_WEIGHTS='[20., 20., 10., 10.]' \
    TRAINER=horovod


