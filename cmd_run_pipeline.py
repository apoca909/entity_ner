import os
'''
pip install torch transformers six seqeval tensorboardX
'''

def run_train_job():
    command = f'python run_task_sl.py --local_rank -1 ' \
    '--learning_rate 2e-4  --config_name ./raw_disf/bert_tagging.json ' \
    '--per_gpu_train_batch_size 16 ' \
    '--vocab_file ./raw_disf/vocab.txt '\
    '--init_checkpoint hfl/chinese-bert-wwm-ext'
    '--num_train_epochs 20 --do_train --do_eval '\
    '--train_file ./data/train.raw.txt '\
    '--dev_file ./data/dev.raw.txt ' \
    '--nline -1 '
    print(command)
    os.system(command)

def run_eval_job():
    command = f'python run_task_sl.py  --do_eval  --config_name ./raw_disf/electra_tagging_small.json '\
    '--vocab_file ./raw_disf/vocab.txt '\
    '--init_checkpoint ./output/ckpts/tagging_20240327_114014/checkpoint-epoch0-final/model.safetensors  '\
    '--dev_file ./data/dev.txt '\
    '--nline -1'
    os.system(command)

if __name__ == '__main__':
    run_train_job()
