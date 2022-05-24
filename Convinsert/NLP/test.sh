# python3 insert_linear_finetune.py --log ./tune_rte_r768.txt --rank 768 --epochs 20
# python3 roberta_finetune.py --rank 768 --log ./test_cola_pre_rte_r768.txt --rank 768 --task cola
# python3 roberta_finetune.py --rank 768 --log ./test_cola_pre_rte_r768.txt --rank 768 --task cola
# python3 roberta_finetune.py --rank 768 --log ./test_rte_pre_rte_r768.txt --rank 768 --task rte

# python3 insert_linear_finetune.py --log ./tune_cola_r768.txt --rank 768 --task cola --epochs 20
# python3 roberta_finetune.py --rank 768 --log ./test_rte_pre_cola_r768.txt --rank 768 --task rte --pretrain cola
# python3 roberta_finetune.py --rank 768 --log ./test_rte_pre_cola_r768.txt --rank 768 --task rte --pretrain cola
# python3 roberta_finetune.py --rank 768 --log ./test_cola_pre_cola_r768.txt --rank 768 --task cola --pretrain cola

# python3 insert_linear_finetune.py --log ./tune_wiki_r384.txt --rank 384 --task wiki
# python3 roberta_finetune.py --rank 768 --log ./test_rte_pre_wiki_r384.txt --rank 384 --task rte --pretrain wiki
# python3 roberta_finetune.py --rank 768 --log ./test_rte_pre_wiki_r384.txt --rank 384 --task rte --pretrain wiki
# python3 roberta_finetune.py --rank 768 --log ./test_cola_pre_wiki_r384.txt --rank 384 --task cola --pretrain wiki
# python3 roberta_finetune.py --rank 768 --log ./test_cola_pre_wiki_r384.txt --rank 384 --task cola --pretrain wiki


python3 insert_linear_finetune.py --log ./tune_wiki_r300.txt --rank 300 --task wiki
python3 roberta_finetune.py --log ./test_rte_pre_wiki_r300.txt --rank 300 --task rte --pretrain wiki
python3 roberta_finetune.py --log ./test_rte_pre_wiki_r300.txt --rank 300 --task rte --pretrain wiki
python3 roberta_finetune.py --log ./test_cola_pre_wiki_r300.txt --rank 300 --task cola --pretrain wiki
python3 roberta_finetune.py --log ./test_cola_pre_wiki_r300.txt --rank 300 --task cola --pretrain wiki