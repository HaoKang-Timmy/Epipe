python3 dataparallel_test_cv.py --pca1 12 --log ./cifar10_pca12.txt
python3 dataparallel_test_cv.py --pca1 6 --log ./cifar10_pca6.txt
python3 dataparallel_test_cv.py --pca1 14 --log ./cifar10_pca14.txt
python3 dataparallel_test_cv.py --pca1 4 --log ./cifar10_pca4.txt
python3 dataparallel_test_cv.py --sortquant --quant 12 --split 4 --log ./cifar10_q12s4.txt
python3 dataparallel_test_cv.py --sortquant --quant 9 --split 3 --log ./cifar10_q9s3.txt
python3 dataparallel_test_cv.py --sortquant --quant 6 --split 2 --log ./cifar10_q6s2.txt
python3 dataparallel_test_cv.py --sortquant --quant 3 --split 1 --log ./cifar10_q3s1.txt
python3 dataparallel_test_nlp.py --sortquant --quant 12 --split 4 --log ./rte_q12s4.txt
python3 dataparallel_test_nlp.py --sortquant --quant 9 --split 3 --log ./rte_q9s3.txt
python3 dataparallel_test_nlp.py --sortquant --quant 6 --split 2 --log ./rte_q6s2.txt
python3 dataparallel_test_nlp.py --sortquant --quant 3 --split 1 --log ./rte_q3s1.txt