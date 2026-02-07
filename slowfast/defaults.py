# Based on https://github.com/facebookresearch/SlowFast/blob/main/slowfast/config/defaults.py#L1291

import math
from fvcore.common.config import CfgNode
#------------------
# Config definition 
#------------------
_C = CfgNode()

#------------------
# Data option
#------------------
_C.DATA = CfgNode()
_C.DATA.COLOR_RND_GRAYSCALE = 0.0 # color random percentage for grayscale conversion
_C.DATA.NUM_FRAMES=32 # number of frames per clip
_C.DATA.SAMPLING_RATE=2 # video sampling rate of the input clip
_C.DATA.TRAIN_JITTER_SCALES=[256,320] # spatial augmentation: jitter scales for training
_C.DATA.TRAIN_CROP_SIZE=224 # spatial crop size for training
_C.DATA.TEST_CROP_SIZE=256 # spatial crop size for testing
_C.DATA.INPUT_CHANNEL_NUM=[3,3]
_C.DATA.TIME_DIFF_PROB=0. # augmentation probability to convert raw decoded video to grayscale temporal difference
_C.DATA.SKIP_ROWS=0 # for chunked reading, dataloader can skip rows in large training csv file
_C.DATA.LOADER_CHUNK_SIZE=0 # loader can read .csv file in chunks of his chunk
_C.DATA.LOADER_CHUNK_OVERALL_SIZE=0 # if LOADER_CHUNK_SIZE>0, define overall length of .csv file
_C.DATA.TRAIN_CROP_NUM_TEMPORAL=1 # how many samples (=clips) to decode from a single video
_C.DATA.DECODING_BACKEND = "torchvision" # decoding backend `pyav` or `torchvision`
_C.DATA.NUM_FRAMES = 8 # number of frames of input clips
_C.DATA.TARGET_FPS = 30 # Input videos may has different fps, convert it to the target video fps before frame sampling.
_C.DATA.USE_OFFSET_SAMPLING=False # if True, perform stride length uniform temporal sampling
#--------------
# SlowFast options
#--------------
_C.SLOWFAST=CfgNode()
_C.SLOWFAST.ALPHA=8 # frame rate reduction ratio, $\alpha$ between the Slow and Fast pathways
_C.SLOWFAST.BETA_INV=8 # inverse of the channel reduction ratio, $\beta$ between the Slow and Fast pathways
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO=2 # ration of channel dimensiosn between slow and fast pathways
_C.SLOWFAST.FUSION_KERNEL_SZ=5 # kernel dimension used for fusing info. from Fast to slow pathway

#-----------------
# ResNet options
#-----------------
_C.RESNET=CfgNode()
_C.RESNET.ZERO_INIT_FINAL_BN=True
_C.RESNET.WIDTH_PER_GROUP=64
_C.RESNET.NUM_GROUPS=1
_C.RESNET.DEPTH=50
_C.RESNET.TRANS_FUNC="bottleneck_transform"
_C.RESNET.STRIDE_1X1=False
_C.RESNET.NUM_BLOCK_TEMP_KERNEL=[[3, 3], [4, 4], [6, 6], [3, 3]]
_C.RESNET.SPATIAL_STRIDES=[[1, 1], [2, 2], [2, 2], [2, 2]]
_C.RESNET.SPATIAL_DILATIONS=[[1, 1], [1, 1], [1, 1], [1, 1]]
#------------------
# Training options
#------------------
_C.TRAIN = CfgNode()
_C.TRAIN.ENABLE = True # if True, train a model; else, skip training
_C.TRAIN.DATASET="ucf101"
_C.TRAIN.BATCH_SIZE=64 # total minibatch size
_C.TRAIN.EVAL_PERIOD=10 # evaluate model on test data every eval period epochs
_C.TRAIN.CHECKPOINT_PERIOD=1 # save model checkpoint every checkpoint period epochs
_C.TRAIN.AUTO_RESUME=True # resume training from the latest checkpoint

#------------------
# Nonlocal options
#------------------
_C.NONLOCAL=CfgNode()
_C.NONLOCAL.LOCATION=[[[], []], [[], []], [[], []], [[], []]]
_C.NONLOCAL.GROUP=[[1, 1], [1, 1], [1, 1], [1, 1]]
_C.NONLOCAL.INSTANTIATION="dot_product"

#------------------
# Batch norm options
#------------------
_C.BN=CfgNode()
_C.BN.USE_PRECISE_STATS=True
_C.BN.NUM_BATCHES_PRECISE=200 # number of samples used to compute precise bn

#------------------
# Optimizer options
#------------------
_C.SOLVER=CfgNode()
_C.SOLVER.BASE_LR=0.1 # base learning rate
_C.SOLVER.LR_POLICY="cosine" # see utils/lr_policy.py for options and examples
_C.SOLVER.MAX_EPOCH=196
_C.SOLVER.MOMENTUM=0.9
_C.SOLVER.WEIGHT_DECAY=1e-4
_C.SOLVER.WARMUP_EPOCHS=34.0 # gradually warm up the SOLVER.BASE_LR over this number of epochs
_C.SOLVER.WARMUP_START_LR=0.01 # start learning rate of the warm up
_C.SOLVER.OPTIMIZING_METHOD="sgd"

#-----------------
# Model options
#-----------------
_C.MODEL=CfgNode()
_C.MODEL.NUM_CLASSES=101
_C.MODEL.ARCH="slowfast"
_C.MODEL.MODEL_NAME="SlowFast"
_C.MODEL.LOSS_FUNC="cross_entropy"
_C.MODEL.DROPOUT_RATE=0.5 # dropout rate before final projection in the backbone

#----------------
# Testing options
#----------------
_C.TEST=CfgNode()
_C.TEST.ENABLE=True
_C.TEST.DATASET="ucf101"
_C.TEST.BATCH_SIZE=64
_C.TEST.NUM_ENSEMBLE_VIEWS=10 # number of clips to sample from a video uniformly for aggregating the prediction results
_C.TEST.NUM_SPATIAL_CROPS=3 # number of crops to sample from a frame spatially for aggregating the prediction results

#----------------
# Augmentation options
#----------------
_C.AUG=CfgNode()
_C.AUG.ENABLE=False # whether to enable random augmentation
_C.AUG.RE_PROB=0.25 # probability of random erase

#-------------------------------
# Multigrid training options
#-------------------------------
_C.MULTIGRID=CfgNode()
_C.MULTIGRID.DEFAULT_S=0
_C.MULTIGRID.SHORT_CYCLE = False
_C.MULTIGRID.LONG_CYCLE = False

#------------------------------------------
# Common train/test data loader options
#------------------------------------------
_C.DATA_LOADER=CfgNode()
_C.DATA_LOADER.NUM_WORKERS=8 # number of data loader workers per training process
_C.DATA_LOADER.PIN_MEMORY=True # load data to pinned host memory
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE=False # enable multithread decoding

#--------------
# Contrastive models (for MoCo, SimCLR, SwAV, BYOL
#--------------
_C.CONTRASTIVE=CfgNode()
_C.CONTRASTIVE.DELTA_CLIPS_MIN=-math.inf # if sampling multiple clips per video, they need to be at least min frames apart
_C.CONTRASTIVE.DELTA_CLIPS_MAX=math.inf # if sampling multiple clips per vid they can be max frames apart


_C.NUM_GPUS=1 # number of GPUs used
_C.NUM_SHARDS=1 # number of machine to use for the job
_C.RNG_SEED=0
_C.OUTPUT_DIR="."



def get_cfg():
    """Get a copy of the default config"""
    return _C.clone()