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


# python3 insert_linear_finetune.py --log ./tune_wiki_r300.txt --rank 300 --task wiki
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_r300.txt --rank 300 --task rte --pretrain wiki
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_r300.txt --rank 300 --task rte --pretrain wiki
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_r300.txt --rank 300 --task cola --pretrain wiki
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_r300.txt --rank 300 --task cola --pretrain wiki
# python3 insert_linear_finetune.py --log ./tune_wiki_type4_r768.txt --rank 768 --task wiki --type 4
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_type4_r768.txt --rank 768 --task rte --pretrain wiki --type 4
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_type4_r768.txt --rank 768 --task rte --pretrain wiki --type 4
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_type4_r768.txt --rank 768 --task cola --pretrain wiki --type 4
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_type4_r768.txt --rank 768 --task cola --pretrain wiki --type 4

# python3 insert_linear_finetune.py --log ./tune_wiki_type4_r384.txt --rank 384 --task wiki --type 4
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_type4_r384.txt --rank 384 --task rte --pretrain wiki --type 4
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_type4_r384.txt --rank 384 --task rte --pretrain wiki --type 4
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_type4_r384.txt --rank 384 --task cola --pretrain wiki --type 4
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_type4_r384.txt --rank 384 --task cola --pretrain wiki --type 4

# python3 insert_linear_finetune.py --log ./tune_wiki_type4_r300.txt --rank 300 --task wiki --type 4
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_type4_r300.txt --rank 300 --task rte --pretrain wiki --type 4
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_type4_r300.txt --rank 300 --task rte --pretrain wiki --type 4
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_type4_r300.txt --rank 300 --task cola --pretrain wiki --type 4
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_type4_r300.txt --rank 300 --task cola --pretrain wiki --type 4



# python3 insert_linear_finetune.py --log ./tune_wiki_r2_64.txt --rank 64 --task wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_r2_64.txt --rank 64 --task rte --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_r2_64.txt --rank 64 --task rte --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_r2_64.txt --rank 64 --task cola --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_r2_64.txt --rank 64 --task cola --pretrain wiki --compressdim -2

# python3 insert_linear_finetune.py --log ./tune_wiki_r2_48.txt --rank 48 --task wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_r2_48.txt --rank 48 --task rte --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_r2_48.txt --rank 48 --task rte --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_r2_48.txt --rank 48 --task cola --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_r2_48.txt --rank 48 --task cola --pretrain wiki --compressdim -2


# python3 insert_linear_finetune.py --log ./tune_wiki_r2_32.txt --rank 32 --task wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_r2_32.txt --rank 32 --task rte --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_r2_32.txt --rank 32 --task rte --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_r2_32.txt --rank 32 --task cola --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_r2_32.txt --rank 32 --task cola --pretrain wiki --compressdim -2


# python3 insert_linear_finetune.py --log ./tune_wiki_r2_32.txt --rank 16 --task wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_r2_32.txt --rank 16 --task rte --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_rte_pre_wiki_r2_32.txt --rank 16 --task rte --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_r2_32.txt --rank 16 --task cola --pretrain wiki --compressdim -2
# python3 roberta_finetune.py --log ./test_cola_pre_wiki_r2_32.txt --rank 16 --task cola --pretrain wiki --compressdim -2
# python3 roberta_rankdecay.py --task cola --step 100 --stopstep 400 --log cola_dynamicliner_r56_s100_t400.txt
# python3 roberta_rankdecay.py --task cola --step 100 --stopstep 400 --log cola_dynamicliner_r56_s100_t400.txt
# python3 roberta_rankdecay.py --task cola --step 300 --stopstep 1200 --log cola_dynamicliner_r56_s300_t1200.txt
# python3 roberta_rankdecay.py --task cola --step 300 --stopstep 1200 --log cola_dynamicliner_r56_s300_t1200.txt
# python3 roberta_rankdecay.py --task cola --step 200 --stopstep 800 --log cola_dynamicliner_r56_s200_t800.txt
# python3 roberta_rankdecay.py --task cola --step 200 --stopstep 800 --log cola_dynamicliner_r56_s200_t800.txt

# python3 roberta_rankdecay.py --task cola --step 100 --stopstep 400 --log cola_dynamicliner_r34_s100_t400.txt --rate 0.75
# python3 roberta_rankdecay.py --task cola --step 100 --stopstep 400 --log cola_dynamicliner_r34_s100_t400.txt --rate 0.75
# python3 roberta_rankdecay.py --task cola --step 300 --stopstep 1200 --log cola_dynamicliner_r34_s300_t1200.txt --rate 0.75
# python3 roberta_rankdecay.py --task cola --step 300 --stopstep 1200 --log cola_dynamicliner_r34_s300_t1200.txt --rate 0.75
# python3 roberta_rankdecay.py --task cola --step 200 --stopstep 800 --log cola_dynamicliner_r34_s200_t800.txt --rate 0.75
# python3 roberta_rankdecay.py --task cola --step 200 --stopstep 800 --log cola_dynamicliner_r34_s200_t800.txt --rate 0.75


# python3 roberta_rankdecay.py --task cola --step 400 --stopstep 3200 --log cola_dynamicliner_r910_s400_t3200.txt --rate 0.9
# python3 roberta_rankdecay.py --task cola --step 400 --stopstep 3200 --log cola_dynamicliner_r910_s400_t3200.txt --rate 0.9
# python3 roberta_rankdecay.py --task cola --step 400 --stopstep 3200 --log cola_dynamicliner_r9_r95_s400_t3200.txt --rate1 0.9 --rate2 0.95
# python3 roberta_rankdecay.py --task cola --step 400 --stopstep 3200 --log cola_dynamicliner_r9_r95_s400_t3200.txt --rate1 0.9 --rate2 0.95
# python3 roberta_rankdecay.py --task cola --step 400 --stopstep 3200 --log cola_dynamicliner_r9_r95_s400_t3200.txt --rate1 0.95 --rate2 0.95
# python3 roberta_rankdecay.py --task cola --step 400 --stopstep 3200 --log cola_dynamicliner_r9_r95_s400_t3200.txt --rate1 0.95 --rate2 0.95


# python3 roberta_rankdecay.py --task cola --step 700 --stopstep 2800 --log cola_dynamicliner_r9_r95_s700_t2800.txt --rate1 0.9 --rate2 0.9
# python3 roberta_rankdecay.py --task cola --step 700 --stopstep 2800 --log cola_dynamicliner_r9_r95_s700_t2800.txt --rate1 0.9 --rate2 0.9


# python3 roberta_rankdecay.py --task cola --step 700 --stopstep 2800 --log cola_dynamicliner_r9_r95_s700_t2800.txt --rate1 0.9 --rate2 0.9
# python3 roberta_rankdecay.py --task cola --step 700 --stopstep 2800 --log cola_dynamicliner_r9_r95_s700_t2800.txt --rate1 0.9 --rate2 0.9
# python3 roberta_rankdecay.py --task cola --step 400 --stopstep 3200 --log cola_dynamicliner_r9_s400_t3200.txt --rate1 0.9 --rate2 1.0
# python3 roberta_rankdecay.py --task cola --step 400 --stopstep 3200 --log cola_dynamicliner_r9_s400_t3200.txt --rate1 0.9 --rate2 1.0
# python3 roberta_rankdecay.py --task cola --step 600 --stopstep 2400 --log cola_dynamicliner_r56_s600_t2400.txt
python3 roberta_rankdecay.py --task cola --step 300 --stopstep 2700 --log cola_dynamicliner_test.txt --rate1 1.0
python3 roberta_rankdecay.py --task cola --step 300 --stopstep 2700 --log cola_dynamicliner_test.txt --rate1 1.0
python3 roberta_rankdecay.py --task cola --step 300 --stopstep 2700 --log cola_dynamicliner_r910_s300_t2700.txt --rate1 0.9
python3 roberta_rankdecay.py --task cola --step 300 --stopstep 2700 --log cola_dynamicliner_r910_s300_t2700.txt --rate1 0.9
python3 roberta_rankdecay.py --task cola --step 100 --stopstep 900 --log cola_dynamicliner_r910_s100_t900.txt --rate1 0.9
python3 roberta_rankdecay.py --task cola --step 100 --stopstep 900 --log cola_dynamicliner_r910_s100_t900.txt --rate1 0.9

python3 roberta_rankdecay.py --task cola --step 200 --stopstep 1800 --log cola_dynamicliner_r910_s200_t1800.txt --rate1 0.9
python3 roberta_rankdecay.py --task cola --step 200 --stopstep 1800 --log cola_dynamicliner_r910_s200_t1800.txt --rate1 0.9
python3 roberta_rankdecay.py --task cola --step 500 --stopstep 2500 --log cola_dynamicliner_r56_s500_t2500.txt 
python3 roberta_rankdecay.py --task cola --step 500 --stopstep 2500 --log cola_dynamicliner_r56_s500_t2500.txt 