python3 cpu_client_cv.py --sortquant --quant 8 --split 0 --chunk 16 --log ./cifar10_chunk4_40epochs_sq4s4.txt

python3 cpu_client_cv.py --sortquant --quant 4 --split 4 --chunk 16 --log ./cifar10_chunk4_40epochs_sq4s4.txt
python3 cpu_client_cv.py --sortquant --quant 6 --split 2 --chunk 16 --log ./cifar10_chunk4_40epochs_sq6s2.txt

python3 cpu_client_cv.py --log ./cifar10_chunk16.txt --chunk 16



# python3 cpu_client_cv.py --sortquant --quant 2 --split 4 --chunk 16 --log ./cifar10_chunk16_40epochs_sq2s4.txt


python3 cpu_client_cv.py --sortquant --quant 9 --split 3 --chunk 16 --log ./cifar10_chunk16_40epochs_sq9s3.txt

python3 cpu_client_cv.py --sortquant --quant 9 --split 3 --chunk 32 --log ./cifar10_chunk32_40epochs_sq9s3.txt