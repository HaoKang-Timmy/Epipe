'''
Author: your name
Date: 2022-02-24 21:25:14
LastEditTime: 2022-03-04 12:11:33
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/code/test.py
'''
import torch.multiprocessing as mp
def add(a,b):
        b[0] = b[0] +a
if __name__ == '__main__':
    a = 5
    b = [6]
    c = 7
    

    p = mp.Process(target=add,args = (a,b))
    p.start()
    p.join()
    print(b)