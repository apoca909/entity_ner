import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'

'''
pip install torch transformers six seqeval tensorboardX
'''
def run_train_job():
    command = f'python run_task_sl.py --local_rank -1 ' \
    '--learning_rate 1e-2  --config_name ./raw_disf/bert_tagging.json ' \
    '--per_gpu_train_batch_size 8 ' \
    '--vocab_file ./raw_disf/vocab.txt '\
    '--num_train_epochs 20 --do_train --do_eval '\
    '--train_file ./data/offer/train.raw.txt '\
    '--dev_file ./data/offer/val.raw.txt ' \
    '--label_file ./raw_disf/offer_ner.json ' \
    '--logging_steps 100 ' \
    '--nline -1 '
    print(command)
    os.system(command)

def run_train_job2():
    command = f'python run_task_sl.py --local_rank -1 ' \
    '--learning_rate 1e-3  --config_name ./raw_disf/bert_tagging.json ' \
    '--per_gpu_train_batch_size 8 ' \
    '--vocab_file ./raw_disf/vocab.txt '\
    '--num_train_epochs 20 --do_train --do_eval '\
    '--train_file ./data/msra/train.txt '\
    '--dev_file ./data/msra/val.txt ' \
    '--logging_steps 100 ' \
    '--nline 100 ' \
    '--label_file ./raw_disf/msra_ner.json '
    print(command)
    os.system(command)
def run_eval_job():
    command = f'python run_task_sl.py  --do_eval  --config_name ./raw_disf/bert_tagging.json '\
    '--vocab_file ./raw_disf/vocab.txt '\
    '--init_checkpoint path_to_ckpt  '\
    '--dev_file ./data/offer/val.raw.txt '\
    '--label_file ./raw_disf/offer_ner.json ' \
    '--nline -1'
    
    os.system(command)

if __name__ == '__main__':
    run_train_job2()
