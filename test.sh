# python test_vision_4gpu.py --sortquant --quant 12 --split 4 --log ./test.txt --chunk 4
# python test_vision_4gpu.py --sortquant --quant 9 --split 3 --log ./test.txt --chunk 4
# python test_vision_4gpu.py --sortquant --quant 6 --split 2 --log ./test.txt --chunk 4
# python test_nlp_4gpu.py --sortquant --quant 12 --split 4 --log ./test.txt --chunk 4
# python test_nlp_4gpu.py --sortquant --quant 9 --split 3 --log ./test.txt --chunk 4
# python test_nlp_4gpu.py --sortquant --quant 6 --split 2 --log ./test.txt --chunk 4
# python test_vision_4gpu.py --sortquant --quant 12 --split 4 --log ./test.txt --chunk 4
# python test_vision_4gpu.py --sortquant --quant 9 --split 3 --log ./test.txt --chunk 4
# python test_vision_4gpu.py --sortquant --quant 6 --split 2 --log ./test.txt --chunk 4
# python test_nlp_4gpu.py --sortquant --quant 12 --split 4 --log ./test.txt --chunk 4
# python test_nlp_4gpu.py --sortquant --quant 9 --split 3 --log ./test.txt --chunk 4
# python test_nlp_4gpu.py --sortquant --quant 6 --split 2 --log ./test.txt --chunk 4
python3 test_cv_2gpu.py --fastquant --quant 6 --split 2 --log ./cifar10_fq62.txt
python3 test_nlp_2gpu.py --fastquant --quant 6 --split 2 --log ./rte_fq62.txt