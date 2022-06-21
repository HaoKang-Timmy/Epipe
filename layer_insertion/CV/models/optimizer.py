import torch


def create_optimizer(args, model):
    if args.type < 4:

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
    else:
        optimizer = torch.optim.SGD(
            [
                {"params": model.conv1.parameters()},
                {"params": model.conv2.parameters()},
                {"params": model.t_conv1.parameters()},
                {"params": model.t_conv2.parameters()},
                {"params": model.conv3.parameters()},
                {"params": model.conv4.parameters()},
                {"params": model.t_conv3.parameters()},
                {"params": model.t_conv4.parameters()},
            ],
            lr=args.lr,
            momentum=0.9,
        )

    return optimizer
