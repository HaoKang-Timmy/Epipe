import torch
from torch.autograd import Function as F
import torch.nn as nn

# class DecayLinearFunction(F):
#     @staticmethod
#     def forward(ctx,input,weight):
#         ctx.weight,ctx.input = weight, input
#         output = input @ weight
#         return output
#     @staticmethod
#     def backward(ctx,grad_output):
#         weight, input = ctx.weight,ctx.input
#         grad_input = grad_output @ weight.t()
#         grad_weight = grad_output.transpose(-1,-2) @ input
#         return grad_input, grad_weight
class DecayLinearFirst(nn.Module):
    def __init__(self, in_features, decay_rate, step, step_stop) -> None:
        super(DecayLinearFirst, self).__init__()
        self.weight = nn.Parameter(torch.eye(in_features))
        self.decay_rate = decay_rate
        self.step = step
        self.step_stop = step_stop
        self.iter = 0
        self.rank = 768
        print(decay_rate, step, step_stop)

    # def reset_parameter(self,rank):
    # self.weight = nn.Parameter(self.weight[:rank,:])
    def forward(self, input):
        if self.training is True:
            self.iter += 1
        if self.iter % self.step == self.step - 1 and self.iter <= self.step_stop:
            self.rank = int(self.rank * self.decay_rate)
            print("changing", self.rank)
        return torch.nn.functional.linear(input, self.weight[: self.rank, :])


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
        self.rank = 768

    def forward(self, input):
        if self.training is True:
            self.iter += 1
        if self.iter % self.step == self.step - 1 and self.iter <= self.step_stop:
            # rank = int(self.weight.shape[-2]*self.decay_rate)
            self.rank = int(self.rank * self.decay_rate)
            print("changing", self.rank)
            # self.reset_parameter(rank)
        return torch.nn.functional.linear(input, self.weight[:, : self.rank])
