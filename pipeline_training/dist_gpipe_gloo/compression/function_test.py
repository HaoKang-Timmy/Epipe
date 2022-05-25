from functions import *
import torch


def error(input, label):
    difference = torch.abs(input) - torch.abs(label)
    # print(input.shape)
    # print(label.shape)
    # print(input)
    # print(label)
    return torch.abs(difference).mean()


class PowerSVD1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p_buffer, q_buffer, iter, grad_p_buffer, grad_q_buffer):
        shape = input.shape
        input = input.view(int(input.shape[0]), int(input.shape[1]), -1)
        for i in range(iter):
            if i == iter - 1:
                p_buffer[0] = torch.linalg.qr(p_buffer[0]).Q
            q_buffer[0] = input @ p_buffer[0]
            if i == iter - 1:
                q_buffer[0] = torch.linalg.qr(q_buffer[0]).Q
            p_buffer[0] = input.permute((0, 2, 1)) @ q_buffer[0]
        ctx.p_buffer, ctx.q_buffer = grad_p_buffer, grad_q_buffer
        ctx.iter, ctx.shape = iter, shape
        result = (q_buffer[0] @ p_buffer[0].permute((0, 2, 1))).view(shape)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        iter, shape = ctx.iter, ctx.shape
        grad_output = grad_output.view(
            int(grad_output.shape[0]), int(grad_output.shape[1]), -1
        )
        p_buffer, q_buffer = ctx.p_buffer, ctx.q_buffer
        for i in range(iter):
            if i == iter - 1:
                p_buffer[0] = torch.linalg.qr(p_buffer[0]).Q
            q_buffer[0] = grad_output @ p_buffer[0]
            if i == iter - 1:
                q_buffer[0] = torch.linalg.qr(q_buffer[0]).Q
            p_buffer[0] = grad_output.permute((0, 2, 1)) @ q_buffer[0]

        result = (q_buffer[0] @ p_buffer[0].permute((0, 2, 1))).view(shape)
        return (
            result,
            None,
            None,
            None,
            None,
            None,
        )


class PowerSVDLayer1(nn.Module):
    def __init__(self, rank, shape, iter) -> None:
        super(PowerSVDLayer1, self).__init__()
        self.p_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[2] * shape[3]), rank))
        )
        self.q_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[1]), rank))
        )
        self.grad_p_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[2] * shape[3]), rank))
        )
        self.grad_q_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[1]), rank))
        )
        # print(self.p_buffer.shape,self.q_buffer.shape)
        self.iter = iter

    def forward(self, input):
        return PowerSVD1.apply(
            input,
            [self.p_buffer],
            [self.q_buffer],
            self.iter,
            [self.grad_p_buffer],
            [self.grad_q_buffer],
        )


rank = 3
input = torch.rand([64, 32, 112, 112])
p = [torch.rand([64, 112 * 112, rank])]
q = [torch.rand([64, 32, rank])]
p, q = PowerSVD(input, q, p, 2)
# print(p.shape,q.shape)
output = PowerSVDDecompress(p, q, input.shape)
print(error(output, input))
layer = PowerSVDLayer1(rank, input.shape, 2)
input = torch.rand([64, 32, 112, 112])
output = layer(input)
print(error(output, input))
