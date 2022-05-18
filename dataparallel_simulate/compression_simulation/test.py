import torch
import time


def poweriter(input, p_buffer, q_buffer, iter):
    for i in range(iter):
        if i == iter - 1:
            p_buffer[0] = torch.linalg.qr(p_buffer[0]).Q
        q_buffer[0] = input @ p_buffer[0]
        if i == iter - 1:
            q_buffer[0] = torch.linalg.qr(q_buffer[0]).Q
        p_buffer[0] = input.permute((0, 1, 3, 2)) @ q_buffer[0]
    return q_buffer[0] @ p_buffer[0].permute((0, 1, 3, 2))


def poweriter3d(input, p_buffer, q_buffer, iter):
    shape = input.shape
    input = input.view(int(input.shape[0]), int(input.shape[1]), -1)
    for i in range(iter):
        if i == iter - 1:
            p_buffer[0] = torch.linalg.qr(p_buffer[0]).Q
        q_buffer[0] = input @ p_buffer[0]
        if i == iter - 1:
            q_buffer[0] = torch.linalg.qr(q_buffer[0]).Q
        p_buffer[0] = input.permute((0, 2, 1)) @ q_buffer[0]
    return (q_buffer[0] @ p_buffer[0].permute((0, 2, 1))).view(shape)


input = torch.rand([64, 32, 112, 112])
p_buffer = torch.rand([64, 32, 112, 3])
q_buffer = torch.rand([64, 32, 112, 3])
for i in range(10):
    output = poweriter(input, [p_buffer], [q_buffer], 1)
start = time.time()
output = poweriter(input, [p_buffer], [q_buffer], 2)
end = time.time()
print("powersvd_time:", end - start, "error:", torch.abs(output - input).mean())
input = input.view(64, 32, -1)
start = time.time()
U, S, V = torch.svd_lowrank(input, q=3)
S = torch.diag_embed(S)
V = V.transpose(-1, -2)
output = torch.matmul(U[..., :, :], S[..., :, :])
output = torch.matmul(output[..., :, :], V[..., :, :])
end = time.time()

print("svdlow_time:", end - start, "error:", torch.abs(output - input).mean())
input = torch.rand([64, 32, 112, 112])
p_buffer = torch.rand([64, 12544, 3])
q_buffer = torch.rand([64, 32, 3])
for i in range(10):
    output = poweriter3d(input, [p_buffer], [q_buffer], 1)
start = time.time()
output = poweriter3d(input, [p_buffer], [q_buffer], 2)
end = time.time()
print("powersvd_time3d:", end - start, "error:", torch.abs(output - input).mean())
