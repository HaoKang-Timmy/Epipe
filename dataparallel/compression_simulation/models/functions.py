import torch


def FastQuantizationCPU(input, bits, split_bits, min_step):
    # print("fastq",input)
    shape_tensor = input.shape
    batch = shape_tensor[0]
    input = input.view(-1).type(torch.double)
    separate = torch.tensor(-1e9).type(torch.double)
    shape = int(input.shape[0])
    if bits + split_bits <= 8:
        output = input.type(torch.int8)
    else:
        output = input.type(torch.int16)
    for i in range(2 ** split_bits):

        if i == 2 ** split_bits - 1:
            kthvalue = input.max()
        else:
            input = input.view(batch, -1)
            kthvalue, indice = torch.kthvalue(
                input[0],
                int(shape * (i + 1) / (2 ** split_bits) / batch),
                keepdim=False,
            )

            input = input.view(-1)

            kthvalue = kthvalue.type(torch.double)
        if kthvalue == separate:
            temp = torch.where(
                (input == kthvalue), input, -1000000.0
            )  # TODO maybe could delete
        else:
            temp = torch.where(
                (input <= kthvalue) & (input > separate), input, -1000000.0
            )
        temp = temp.type(torch.float)
        # print(temp)
        indexs = (temp != -1000000.0).nonzero()
        indexs = indexs.view(-1)
        temp = torch.index_select(temp, 0, indexs)
        if bits + split_bits == 8 or bits + split_bits == 16:
            offset = -pow(2, bits + split_bits - 1) + pow(2, bits) * i
            min = temp.min()
            step = (kthvalue - min) / (pow(2, bits) - 1)
            min_step[i, 0], min_step[i, 1] = min, step
            if step != 0.0:
                temp = torch.round((temp - temp.min()) / step)
            else:
                temp = temp - temp
            temp += offset
        else:
            offset = pow(2, bits) * i
            min = temp.min()
            step = (kthvalue - min) / (pow(2, bits) - 1)
            min_step[i, 0], min_step[i, 1] = min, step
            if step != 0.0:
                temp = torch.round((temp - min) / step)
            else:
                temp = temp - temp
            temp += pow(2, bits) * i

        # print(temp)
        temp = temp.type(output.dtype)
        output.scatter_(0, indexs, temp)
        separate = kthvalue

    output = output.view(shape_tensor)
    if bits + split_bits > 8 and bits + split_bits <= 16:
        output = output.view(dtype=torch.int8)
    # print(min_step)
    # print(output)
    return min_step, output


def FastDequantizationCPU(recv: torch.tensor, bits, split_bits, min_step, grad_output):
    shape = recv.shape
    recv = recv.view(-1)
    grad_output = grad_output.view(-1)
    if bits + split_bits > 8 and bits + split_bits <= 16:
        recv = recv.view(dtype=torch.int16)
    recv = recv.type(torch.long)
    for i in range(2 ** split_bits):
        if bits + split_bits == 8 or bits + split_bits == 16:
            upperbound = -pow(2, bits + split_bits - 1) + pow(2, bits) * (i + 1)
            lowerbound = -pow(2, bits + split_bits - 1) + pow(2, bits) * i
            # print(upperbound,lowerbound)
            temp = torch.where(
                (recv < upperbound) & (recv >= lowerbound), recv, -100000
            )
            indexs = (temp != -100000).nonzero()
            indexs = indexs.view(-1)
            temp = torch.index_select(temp, 0, indexs)
            offset = pow(2, bits + split_bits - 1) - pow(2, bits) * i
            temp += offset
            # print(min_step[i, 1],min_step[i, 0])
            temp = temp.type(torch.float)
            temp *= min_step[i, 1]
            temp += min_step[i, 0]
        else:
            upperbound = pow(2, bits) * (i + 1)
            lowerbound = pow(2, bits) * i
            temp = torch.where((recv < upperbound) & (recv >= lowerbound), recv, 0)

            indexs = torch.nonzero(temp)
            indexs = indexs.view(-1)
            temp = torch.index_select(temp, 0, indexs)
            offset = -pow(2, bits) * i
            temp += offset
            temp = temp.type(torch.float)
            temp *= min_step[i, 1]
            temp += min_step[i, 0]

        grad_output.scatter_(0, indexs, temp)

    grad_output = grad_output.view(shape)
    recv = recv.view(shape)
    # print("fastdeq",grad_output)
    return grad_output
