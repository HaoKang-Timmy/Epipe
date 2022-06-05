import torch


def create_optimizer(args, model):

    optimizer = torch.optim.SGD(
        [
            {"params": model.conv1.parameters()},
            {"params": model.conv2.parameters()},
            {"params": model.t_conv1.parameters()},
            {"params": model.t_conv2.parameters()},
        ],
        lr=args.lr,
        momentum=0.9,
    )

    return optimizer
