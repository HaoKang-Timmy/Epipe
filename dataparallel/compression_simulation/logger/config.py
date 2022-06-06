from yacs.config import CfgNode as CN


def create_config(args):
    _C = CN()
    _C.SYSTEM = CN()
    _C.SYSTEM.NUM_GPUS = args.worker
    _C.SYSTEM.NUM_LOADERS = args.loader
    _C.TRAIN = CN()
    _C.TRAIN.EPOCHS = args.epochs
    _C.TRAIN.BATCHES = args.batches
    _C.TRAIN.OPT = "SGD"
    _C.TRAIN.PARAM = CN()
    _C.TRAIN.PARAM.LR = args.lr
    _C.TRAIN.PARAM.MOMENTUM = args.momentum
    _C.TRAIN.PARAM.UNIFORMQUANT = args.quant
    _C.TRAIN.PARAM.PRUNE = args.prune
    _C.TRAIN.PARAM.SORTQUANT = args.sortquant
    _C.TRAIN.PARAM.SPLIT = args.split
    _C.TRAIN.PARAM.CONVINSERT = args.conv1 and args.conv2
    _C.TRAIN.PARAM.POWERRANK = args.powerrank
    _C.TRAIN.PARAM.POWERRANK1 = args.powerrank1
    _C.TRAIN.PARAM.POWERITER = args.poweriter
    _C.TRAIN.PARAM.SVD = args.svd
    return _C.clone()
