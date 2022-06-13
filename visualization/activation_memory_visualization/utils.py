import torch
import torch.autograd as autograd


def linear_seperate(input_tensor, chunk):
    min, max = input_tensor.min(), input_tensor.max()
    step = (max - max) / chunk
    input_tensor = input_tensor.type(torch.long)
    chunk_list = []

    for i in range(chunk):
        lowerbound = min + i * step
        upperbound = min + (i + 1) * step
        temp = torch.where(
            (input_tensor < upperbound) & (input_tensor >= lowerbound),
            input_tensor,
            -10000.0,
        )
        indexs = (temp != -10000.0).nonzero()
        indexs = indexs.view(-1)
        number = indexs.shape[0]
        chunk_list.append(number)


def abl_err(input, label):
    diff = torch.abs(input) - torch.abs(label)
    return torch.abs(diff).mean()


def relative_err(input, label):
    diff = input - label
    diff = torch.abs(diff)
    relative = diff / label
    relative = relative.view(-1)
    not_nan = relative == relative
    # src = relative * not_nan
    src = relative.masked_select(not_nan)
    return torch.abs(src).mean()


class Fakequantize(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits):
        ctx.bits = bits
        min, max = input.min(), input.max()
        step = (max - min) / (pow(2, bits) - 1)
        output = torch.round((input - min) / step) - pow(2, bits - 1)
        output = (output + pow(2, bits - 1)) * step + min
        # print(bits)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits
        min, max = grad_output.min(), grad_output.max()
        step = (max - min) / (pow(2, bits) - 1)
        grad_output = torch.round((grad_output - min) / step) - pow(2, bits - 1)
        grad_output = (grad_output + pow(2, bits - 1)) * step + min
        return grad_output, None


class SortQuantization(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, split_bits):
        # print(bits,ratio,partition)
        shape = input.shape
        input = input.view(-1)
        # print("src_prun",src_temp,"index_prun",index)
        # quantization src1
        # print(src.shape)
        src, index = torch.sort(input, dim=0)
        index = torch.tensor_split(index, 2 ** split_bits)
        src = torch.tensor_split(src, 2 ** split_bits)
        # print(src1[1])
        for i in range(2 ** split_bits):
            min, max = src[i].min(), src[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)

                # print(torch.round((src1[i] - min) / step)- pow(2, bits - 1) )
                temp_src = torch.round((src[i] - min) / step) - pow(2, bits - 1)

                temp_src = (temp_src + pow(2, bits - 1)) * step + min
                # if i == 0:
                #     print(temp_src)
            else:
                # print(src1[i])
                temp_src = src[i]
            # print("origin_src",src1[i])
            # print("quant_dequant_src",temp_src)

            input.scatter_(0, index[i], temp_src)

        # src = src.view(-1)
        # index = index.view(-1)
        # print("final_index",index,"final_src",src)
        ctx.bits = bits
        ctx.split_bits = split_bits
        input = input.view(shape)
        # if input.get_device() == 0:
        #     print("forward",torch.abs(torch.abs(input) - torch.abs(test)).sum()/torch.abs(test).sum())
        return input

    @staticmethod
    def backward(ctx, grad_backward):
        bits, split_bits = ctx.bits, ctx.split_bits
        shape = grad_backward.shape
        grad_backward = grad_backward.view(-1)
        src, index = torch.sort(grad_backward, dim=0)
        index = torch.tensor_split(index, 2 ** split_bits)
        src = torch.tensor_split(src, 2 ** split_bits)
        for i in range(2 ** split_bits):
            min, max = src[i].min(), src[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)
                src_temp = torch.round((src[i] - min) / step) - pow(2, bits - 1)
                src_temp = (src_temp + pow(2, bits - 1)) * step + min
            else:
                src_temp = src[i]
            grad_backward.scatter_(0, index[i], src_temp)

        grad_backward = grad_backward.view(shape)

        return grad_backward, None, None
