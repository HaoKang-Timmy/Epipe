# python3 dataparallel_test_cv.py --pca1 12 --log ./cifar10_pca12.txt
# python3 dataparallel_test_cv.py --pca1 6 --log ./cifar10_pca6.txt
# python3 dataparallel_test_cv.py --pca1 14 --log ./cifar10_pca14.txt
# python3 dataparallel_test_cv.py --pca1 4 --log ./cifar10_pca4.txt
# python3 dataparallel_test_cv.py --sortquant --quant 12 --split 4 --log ./cifar10_q12s4.txt
# python3 dataparallel_test_cv.py --sortquant --quant 9 --split 3 --log ./cifar10_q9s3.txt
# python3 dataparallel_test_cv.py --sortquant --quant 6 --split 2 --log ./cifar10_q6s2.txt
# python3 dataparallel_test_cv.py --sortquant --quant 3 --split 1 --log ./cifar10_q3s1.txt
# python3 dataparallel_test_nlp.py --sortquant --quant 12 --split 4 --log ./rte_q12s4.txt
# python3 dataparallel_test_nlp.py --sortquant --quant 9 --split 3 --log ./rte_q9s3.txt
# python3 dataparallel_test_nlp.py --sortquant --quant 6 --split 2 --log ./rte_q6s2.txt
# python3 dataparallel_test_nlp.py --sortquant --quant 3 --split 1 --log ./rte_q3s1.txt
# python3 dataparallel_test_nlp.py --log rte_pca100.txt --pca 64
# python3 dataparallel_test_nlp.py --log rte_pca100.txt --pca 128
# python3 mix_test.py --pca1 12 --sortquant --quant 6 --split 2 --log ./CIFAR10_pca12_q6s2.txt
# python3 mix_test.py --pca1 12 --sortquant --quant 4 --split 4 --log ./CIFAR10_pca12_q4s4.txt
# python3 mix_test.py --pca1 14 --sortquant --quant 4 --split 4 --log ./CIFAR10_pca14_q4s4.txt
# python3 mix_test.py --pca1 14 --sortquant --quant 6 --split 2 --log ./CIFAR10_pca14_q6s2.txt
# python3 dataparallel_test_cv.py --log ./test_quant10_2l2l_withoutrelu.txt --quant 10
# python3 dataparallel_test_cv.py --log ./test_quant10_1l1l_relu.txt --quant 10 --secondlayer


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
# python3 dataparallel_test_cv.py --powersvd 7 --poweriter 2 --log cifar10_power7_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 5 --poweriter 2 --log cifar10_power5_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 1 --poweriter 2 --log cifar10_power1_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 1 --poweriter 5 --log cifar10_power1_iter5.txt
# python3 dataparallel_test_cv.py --powersvd 1 --poweriter 10 --log cifar10_power1_iter10.txt
# python3 dataparallel_test_cv.py --powersvd 3 --poweriter 2 --log cifar10_power3_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 10 --poweriter 2 --log cifar10_power10_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 15 --poweriter 2 --log cifar15_power10_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 20 --poweriter 2 --log cifar10_power20_iter2.txt
# python3 dataparallel_test_cv.py --powersvd 3 --log cifar10_power3.txt
# python3 dataparallel_test_cv.py --powersvd 20 --log cifar10_power20.txt
# python3 dataparallel_test_cv.py --powersvd 10 --log cifar10_power10.txt
# python3 dataparallel_test_cv.py --powersvd 5 --log cifar10_power5.txt

# python3 dataparallel_test_cv.py --powersvd 1 --log cifar10_power1.txt
# python3 dataparallel_test_cv.py --svd 3 --log cifar10_reshapesvd3.txt
# python3 dataparallel_test_cv.py --svd 5 --log cifar10_reshapesvd5.txt
# python3 dataparallel_test_cv.py --svd 4 --log cifar10_reshapesvd4.txt
python3 dataparallel_test_cv.py --powersvd 5 --powersvd1 7 --poweriter 3 --log cifar10_powersvd5_7_3.txt
python3 dataparallel_test_cv.py --powersvd 5 --powersvd1 7 --poweriter 2 --log cifar10_powersvd5_7_2.txt
python3 dataparallel_test_cv.py --powersvd 7 --powersvd1 7 --poweriter 2 --log cifar10_powersvd7_7_2.txt
python3 dataparallel_test_cv1.py --powersvd 3 --powersvd1 7 --poweriter 4 --log 1cifar10_powersvd3_7_4.txt
python3 dataparallel_test_cv1.py --powersvd 3 --powersvd1 7 --poweriter 2 --log 1cifar10_powersvd3_7_2.txt
python3 dataparallel_test_cv1.py --powersvd 4 --powersvd1 7 --poweriter 2 --log 1cifar10_powersvd4_7_2.txt
python3 dataparallel_test_cv1.py --powersvd 2 --powersvd1 7 --poweriter 2 --log 1cifar10_powersvd2_7_2.txt

