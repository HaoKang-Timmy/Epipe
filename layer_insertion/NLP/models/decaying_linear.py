import torch
from torch.autograd import Function as F
import torch.nn as nn


def detect_nan(tensor):
    return torch.any(torch.isnan(tensor))


# 3D input only
# Y = XW.T + B
class DecaylinearFunctionFirst(F):
    @staticmethod
    def forward(ctx, input, weight, rank, l1):
        ctx.input = input
        ctx.weight = weight
        ctx.rank, ctx.l1 = rank, l1
        output = input @ weight.t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        weight = ctx.weight
        rank, l1 = ctx.rank, ctx.l1
        grad_input = grad_output @ weight
        grad_weight = grad_output.transpose(-1, -2) @ input
        # grad_weight[rank:,:] += -l1 * grad_weight[rank:,:]  / torch.abs(grad_weight[rank:,:])
        l1_reg = -l1 * grad_weight[rank:, :] / torch.abs(grad_weight[rank:, :])
        l1_reg = torch.nan_to_num(l1_reg, 0.0, 0.0, 0.0)
        grad_weight[rank:, :] += l1_reg
        # print("set1",grad_weight)
        # print(detect_nan(grad_weight))
        # if detect_nan(grad_weight).item() is True:
        #     while(1):
        #         pass
        return grad_input, grad_weight, None, None


class DecaylinearFunctionSecond(F):
    @staticmethod
    def forward(ctx, input, weight, rank, l1):
        ctx.input = input
        ctx.weight = weight
        ctx.rank, ctx.l1 = rank, l1
        output = input @ weight.t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        weight = ctx.weight
        rank, l1 = ctx.rank, ctx.l1
        grad_input = grad_output @ weight
        grad_weight = grad_output.transpose(-1, -2) @ input
        # grad_weight[:,rank:] += -l1 * grad_weight[:,rank:]  / torch.abs(grad_weight[:,rank:])
        l1_reg = -l1 * grad_weight[:, rank:] / torch.abs(grad_weight[:, rank:])
        l1_reg = torch.nan_to_num(l1_reg, 0.0, 0.0, 0.0)
        grad_weight[:, rank:] += l1_reg
        # print("set2",grad_weight)
        # print(detect_nan(grad_weight))
        # if detect_nan(grad_weight).item() is True:
        #     while(1):
        #         pass
        return grad_input, grad_weight, None, None


class DecayLinearFirst(nn.Module):
    def __init__(self, in_features, decay_rate, step, step_stop) -> None:
        super(DecayLinearFirst, self).__init__()
        self.weight = nn.Parameter(torch.eye(in_features))
        self.decay_rate = decay_rate
        self.step = step
        self.step_stop = step_stop
        self.iter = 0
        self.rank = in_features
        self.rank1 = int(in_features * decay_rate)
        print("first layer", self.rank, self.rank1)

    # def reset_parameter(self,rank):
    # self.weight = nn.Parameter(self.weight[:rank,:])
    def forward(self, input):
        if self.training is True:
            self.iter += 1
        if self.iter % self.step == self.step - 1 and self.iter <= self.step_stop:
            self.rank = int(self.rank * self.decay_rate)
            if self.iter < self.step_stop - 30:
                self.rank1 = int(self.rank1 * self.decay_rate)
            print("changing10", self.rank)
            print("changing11", self.rank1)

        # return torch.nn.functional.linear(input, self.weight[: self.rank, :])
        return DecaylinearFunctionFirst.apply(
            input, self.weight[: self.rank, :], self.rank1, 1e-4
        )


class DecayLinearSecond(nn.Module):
    def __init__(self, in_features, decay_rate, step, step_stop) -> None:
        super(DecayLinearSecond, self).__init__()
        self.weight = self.weight = nn.Parameter(torch.eye(in_features))
        self.decay_rate = decay_rate
        self.step = step
        self.step_stop = step_stop
        self.iter = 0
        # def reset_parameter(self,rank):
        #     self.weight = nn.Parameter(self.weight[:,:rank])
        self.rank = in_features
        self.rank1 = int(self.rank * decay_rate)
        print("second layer", self.rank, self.rank1)

    def forward(self, input):
        if self.training is True:
            self.iter += 1
        if self.iter % self.step == self.step - 1 and self.iter <= self.step_stop:
            # rank = int(self.weight.shape[-2]*self.decay_rate)
            self.rank = int(self.rank * self.decay_rate)
            print("changing20", self.rank)
            print("changing21", self.rank1)
            if self.iter < self.step_stop - 30:
                self.rank1 = int(self.rank1 * self.decay_rate)
        return DecaylinearFunctionSecond.apply(
            input, self.weight[:, : self.rank], self.rank1, 1e-4
        )
