'''
Author: your name
Date: 2022-03-20 20:08:44
LastEditTime: 2022-03-20 21:05:58
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe/distributedlayers/utils.py
'''
from compression import TopkLayer,RemoteQuantizationLayer
def dist_init_model(model_client:list,model_server:list,settings):
    if len(model_client) != 2:
        print("wrong form, quit")
        return None
    if settings["prun"] is not None:
        # client
        model_client[0].append(TopkLayer(settings["prun"]))
        model_server.append(TopkLayer(settings["prun"]))
    if settings["quantization"] != 0:
        model_client[0].append(RemoteQuantizationLayer(settings["quantization"],settings["leader"],settings["leader"]))

        model_server.insert(TopkLayer(settings["prun"]))
        pass
    pass