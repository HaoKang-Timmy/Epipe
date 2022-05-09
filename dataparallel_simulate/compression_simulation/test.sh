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
python3 dataparallel_test_nlp.py --task cola --sortquant --sort 2 --squant 4 --log ./cola_sq42.txt
python3 dataparallel_test_nlp.py --task cola --quant 6 --log ./cola_q6.txt
