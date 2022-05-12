'''
Author: Beta Cat 466904389@qq.com
Date: 2022-05-12 00:35:04
LastEditors: Beta Cat 466904389@qq.com
LastEditTime: 2022-05-13 01:53:48
FilePath: /research/gpipe_test/dataparallel_simulate/compression_simulation/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE

'''
# """
# Author: Beta Cat 466904389@qq.com
# Date: 2022-05-11 00:35:33
# LastEditors: Beta Cat 466904389@qq.com
# LastEditTime: 2022-05-11 20:24:59
# FilePath: /research/test/powersgd_test.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# """

# from powersgd import PowerSGD, Config, optimizer_step
# import torch
# from utils import *
# import time

# params = torch.rand([64, 32, 112, 112]).to(1).requires_grad_()
# some = torch.rand([64, 32, 112, 112]).to(1)
# # print(params)
# start = time.time()

# powersgd = PowerSGD(
#     params,
#     config=Config(
#         rank=20,  # lower rank => more aggressive compression
#         min_compression_rate=1,  # don't compress gradients with less compression
#         num_iters_per_step=20,  #   # lower number => more aggressive compression
#         start_compressing_after_num_steps=0,
#     ),
# )
# print(time.time() - start)
# output = PowerPCA.apply(params, powersgd)
# output.backward(some)
from utils import PowerSVD,PowerSVDLayer
import torch
import time
layer = PowerSVDLayer(10,[64,32,112,112],3)
start = time.time()
input = torch.rand([64,32,112,112])
output = layer(input)
end = time.time()
print(end - start)