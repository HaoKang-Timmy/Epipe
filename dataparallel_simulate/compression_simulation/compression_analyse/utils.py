import torch


def abl_difference(input, label):
    diff = torch.abs(input) - torch.abs(label)
    diff = torch.abs(diff)
    return diff.mean(), diff.max()
