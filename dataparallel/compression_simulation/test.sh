
# <<<<<<< HEAD
# python3 dataparallel_test_cv.py --task FOOD101 --prune 0.5 --log ./log/FOOD101_prune0.5.txt
# python3 dataparallel_test_cv.py --task FOOD101 --prune 0.4 --log ./log/FOOD101_prune0.4.txt
# python3 dataparallel_test_cv.py --task FOOD101 --prune 0.3 --log ./log/FOOD101_prune0.3.txt
# python3 dataparallel_test_cv.py --task FOOD101 --prune 0.2 --log ./log/FOOD101_prune0.2.txt
# python3 dataparallel_test_cv.py --task FOOD101 --prune 0.1 --log ./log/FOOD101_prune0.1.txt
# python3 dataparallel_test_cv.py --task FOOD101 --prune 0.05 --log ./log/FOOD101_prune0.05.txt
# python3 dataparallel_test_cv.py --task FOOD101 --quant 16 --log ./log/FOOD101_quant16.txt
# python3 dataparallel_test_cv.py --task FOOD101 --quant 12 --log ./log/FOOD101_quant12.txt
# python3 dataparallel_test_cv.py --task FOOD101 --quant 11 --log ./log/FOOD101_quant11.txt
# python3 dataparallel_test_cv.py --task FOOD101 --quant 10 --log ./log/FOOD101_quant10.txt
# =======

# python3 dataparallel_test_cv.py --log ./test_quant12_2l2l.txt --quant 12
# python3 dataparallel_test_cv.py --log ./test_quant9_1l1l_relu.txt --quant 9
# python3 dataparallel_test_cv.py --log ./test_quant10_1l1l_relu.txt --quant 10
# python3 dataparallel_test_cv.py --log ./test_quant11_1l1l_relu.txt --quant 11
# python3 dataparallel_test_nlp.py --linear --log ./linear_insert_cola.txt --task cola
# python3 dataparallel_test_nlp.py --linear --log ./linear_insert_rte.txt
# python3 dataparallel_test_cv.py --log ./norelu_q10.txt --quant 10 --relu
# python3 dataparallel_test_cv.py --log ./_q10.txt --quant 10
# python3 dataparallel_test_nlp.py --task cola --sortquant --sort 2 --squant 6 --log ./cola_sq62.txt
# python3 dataparallel_test_nlp.py --task cola --quant 8 --log ./cola_q8.txt
# python3 dataparallel_test_cv.py --conv2 --log ./cifar10_cnquant8_conv.txt --secondlayer
# python3 dataparallel_test_cv.py --channelquant 10 --conv2 --log ./cifar10_cnquant10_conv.txt
# python3 dataparallel_test_nlp.py --task cola --sortquant --sort 2 --squant 4 --log ./cola_sq42.txt
# python3 dataparallel_test_nlp.py --task cola --quant 6 --log ./cola_q6.txt
# python3 dataparallel_test_cv.py --powersvd 7 --powersvd 2 --log cifar10_power7_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 5 --powersvd 2 --log cifar10_power5_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 1 --powersvd 2 --log cifar10_power1_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 1 --powersvd 5 --log cifar10_power1_iter5.txt
# python3 dataparallel_test_cv.py --powersvd 1 --powersvd 10 --log cifar10_power1_iter10.txt
# python3 dataparallel_test_cv.py --powersvd 3 --powersvd 2 --log cifar10_power3_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 10 --powersvd 2 --log cifar10_power10_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 15 --powersvd 2 --log cifar15_power10_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 20 --powersvd 2 --log cifar10_power20_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 3 --log cifar10_power3.txt
# python3 dataparallel_test_cv.py --powersvd 20 --log cifar10_power20.txt
# python3 dataparallel_test_cv.py --powersvd 10 --log cifar10_power10.txt
# python3 dataparallel_test_cv.py --powersvd 5 --log cifar10_power5.txt

# python3 dataparallel_test_cv.py --powersvd 1 --log cifar10_power1.txt
# python3 dataparallel_test_cv.py --svd 3 --log cifar10_reshapesvd3.txt
# python3 dataparallel_test_cv.py --svd 5 --log cifar10_reshapesvd5.txt
# python3 dataparallel_test_cv.py --svd 4 --log cifar10_reshapesvd4.txt
# python3 dataparallel_test_cv.py --powersvd 5 --powersvd1 7 --powersvd 3 --log cifar10_powersvd5_7_3.txt
# python3 dataparallel_test_cv.py --powersvd 5 --powersvd1 7 --powersvd 2 --log cifar10_powersvd5_7_2.txt
# python3 dataparallel_test_cv.py --powersvd 7 --powersvd1 7 --powersvd 2 --log cifar10_powersvd7_7_2.txt
# python3 dataparallel_test_cv1.py --powersvd 3 --powersvd1 7 --powersvd 4 --log 1cifar10_powersvd3_7_4.txt
# python3 dataparallel_test_cv1.py --powersvd 3 --powersvd1 7 --powersvd 2 --log 1cifar10_powersvd3_7_2.txt
# python3 dataparallel_test_cv1.py --powersvd 4 --powersvd1 7 --powersvd 2 --log 1cifar10_powersvd4_7_2.txt
# python3 dataparallel_test_cv1.py --powersvd 2 --powersvd1 7 --powersvd 2 --log 1cifar10_powersvd2_7_2.txt
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd1.txt --powersvd 1 --powersvd 1
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd1.txt --powersvd 1 --powersvd 1
# python3 dataparallel_test_nlp_fp16.py --log ./cola_fp32_powersvd1.txt --powersvd 1 --powersvd 1 --task cola
# python3 dataparallel_test_nlp_fp16.py --log ./cola_fp32_powersvd1.txt --powersvd 1 --powersvd 1 --task cola
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd2.txt --powersvd 2 --powersvd 2
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd2.txt --powersvd 2 --powersvd 2
# python3 dataparallel_test_nlp_fp16.py --log ./cola_fp32_powersvd2.txt --powersvd 2 --powersvd 2 --task cola
# python3 dataparallel_test_nlp_fp16.py --log ./cola_fp32_powersvd2.txt --powersvd 2 --powersvd 2 --task cola
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp16.txt --fp16
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp16.txt --fp16
# python3 dataparallel_test_nlp_fp16.py --task cola --log ./cola_fp16.txt --fp16
# python3 dataparallel_test_nlp_fp16.py --task cola --log ./cola_fp16.txt --fp16
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd32.txt --powersvd 32 --powersvd 32
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd32.txt --powersvd 32 --powersvd 32
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd16.txt --powersvd 16 --powersvd 16
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd16.txt --powersvd 16 --powersvd 16
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd16.txt --powersvd 16 --powersvd1 16 
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd16.txt --powersvd 16 --powersvd1 16 
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd32.txt --powersvd 32 --powersvd1 32 
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd32.txt --powersvd 32 --powersvd1 32 
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd32.txt --powersvd 32 --powersvd1 32 

# python3 dataparallel_test_nlp_fp16.py  --log ./rte_fp16.txt --fp16
# python3 dataparallel_test_nlp_fp16.py  --log ./rte_fp16.txt --fp16
# python3 dataparallel_test_nlp_fp16.py --log ./rte_fp32_powersvd48.txt --powersvd 48 --powersvd1 48 
# python3 dataparallel_test_nlp_fp16.py --task rte --log rte_eye.txt --eye
# python3 dataparallel_test_nlp_fp16.py --task cola --log cola_eye.txt --eye
# python3 dataparallel_test_nlp_fp16.py --task rte --log rte_last2_insert50.txt --channelsize 50
# python3 dataparallel_test_nlp_fp16.py --task cola --log cola_last2_insert50.txt --channelsize 50
# # python3 dataparallel_test_nlp_fp16.py --task rte --log rte_eye.txt --eye
# # python3 dataparallel_test_nlp_fp16.py --task cola --log cola_eye.txt --eye
# python3 dataparallel_test_nlp_fp16.py --task rte --log rte_last2_insert50.txt --channelsize 50
# python3 dataparallel_test_nlp_fp16.py --task cola --log cola_last2_insert50.txt --channelsize 50
# python3 dataparallel_test_nlp_fp16.py --task rte --log rte_last2_insert100.txt --channelsize 100
# python3 dataparallel_test_nlp_fp16.py --task cola --log cola_last2_insert100.txt --channelsize 100
python3 dataparallel_test_cv.py --powerrank 2 --powerrank1 7 --log cifar10_iter27.txt

python3 dataparallel_test_nlp.py  --prun 0.5 --log cola_prun0.5.txt
python3 dataparallel_test_nlp.py  --prun 0.3 --log cola_prun0.3.txt
python3 dataparallel_test_nlp.py  --prun 0.2 --log cola_prun0.2.txt
python3 dataparallel_test_nlp.py  --prun 0.1 --log cola_prun0.1.txt
python3 dataparallel_test_nlp.py  --quant 16 --log cola_quant16.txt
python3 dataparallel_test_nlp.py  --quant 12 --log cola_quant12.txt
python3 dataparallel_test_nlp.py  --quant 8 --log cola_quant8.txt
python3 dataparallel_test_nlp.py  --quant 6 --log cola_quant6.txt

python3 dataparallel_test_nlp.py  --prun 0.5 --log cola_prun0.5.txt
python3 dataparallel_test_nlp.py  --prun 0.3 --log cola_prun0.3.txt
python3 dataparallel_test_nlp.py  --prun 0.2 --log cola_prun0.2.txt
python3 dataparallel_test_nlp.py  --prun 0.1 --log cola_prun0.1.txt
python3 dataparallel_test_nlp.py  --quant 16 --log cola_quant16.txt
python3 dataparallel_test_nlp.py  --quant 12 --log cola_quant12.txt
python3 dataparallel_test_nlp.py  --quant 8 --log cola_quant8.txt
python3 dataparallel_test_nlp.py  --quant 6 --log cola_quant6.txt

