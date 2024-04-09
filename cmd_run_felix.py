import os
python = '/opt/homebrew/Caskroom/miniconda/base/envs/wenet/bin/python'
'''
pip install torch transformers six seqeval tensorboardX ltp
'''

def run_train_tsfmrnn():
    command = f'HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=-1 {python} run_task_disf.py --local_rank -1 ' \
    '--learning_rate 2e-4 --model_type tsfmrnn_tagging --config_name ./raw_disf/electra_tagging_small.json ' \
    '--per_gpu_train_batch_size 16 ' \
    '--vocab_file ./raw_disf/vocab.txt '\
    '--num_train_epochs 20 --do_train --do_eval '\
    '--train_file ./data/train.txt.seg '\
    '--dev_file ./data/dev.txt.seg ' \
    '--nline -1 '\
    '--weight 0.5 0.5'
    #'--init_checkpoint ./output/ckpts/tagging_20240325_204922/checkpoint-epoch9-final/model.safetensors'
    print(command)
    os.system(command)
def run_train_rounds():
    bs = 64
    epoch = 20
    model_cfgs = ['electra_tagging_small.json','electra_tagging_pro.json']
    weights = ['1.0 0.0', '0.75 0.25', '0.5 0.5', '0.25 0.75', '0.0 1.0']
    for model_cfg in model_cfgs:
        for weight in weights:
            command = f'{python} run_task_disf.py --local_rank -1 ' \
            f'--learning_rate 2e-4 --model_type tsfmrnn_tagging --config_name ./raw_disf/{model_cfg} ' \
            f'--per_gpu_train_batch_size {bs} '\
            '--vocab_file ./raw_disf/vocab.txt '\
            f'--num_train_epochs {epoch} --do_train --do_eval '\
            '--train_file ./data/train.txt.seg '\
            '--dev_file ./data/dev.txt.seg ' \
            '--nline -1 '\
            f'--weight {weight} '
            #'--init_checkpoint ./output/ckpts/tagging_20240325_204922/checkpoint-epoch9-final/model.safetensors'
            print(command)
            os.system(command)

def run_train_rounds_base():
    bs = 16
    epoch = 5
    datas = ['train.txt.seg', 'train2.txt.seg']
    ckpt = 'hfl/chinese-electra-180g-base-discriminator'
    #ckpt = './data/models--hfl--chinese-electra-180g-base-discriminator'  #使用electra预训练模型
    weights = ['1.0 0.0', '0.75 0.25', '0.5 0.5', '0.25 0.75', '0.0 1.0']
    for train_data in datas:
        for weight in weights:
            command = f'HF_ENDPOINT=https://hf-mirror.com {python} run_task_disf.py --local_rank -1 ' \
            f'--learning_rate 2e-4 --model_type tsfmrnn_tagging --config_name ./raw_disf/electra_tagging.json ' \
            f'--per_gpu_train_batch_size {bs} '\
            '--vocab_file ./raw_disf/vocab.txt '\
            f'--num_train_epochs {epoch} --do_train --do_eval '\
            f'--train_file /home/jovyan/work/punc_on/punc_v2/data/{train_data} '\
            '--dev_file /home/jovyan/work/punc_on/punc_v2/data/dev.txt.seg ' \
            '--nline -1 '\
            f'--weight {weight} ' \
            f'--init_checkpoint {ckpt} '\
            f'--output_dir /home/jovyan/work/punc_on/punc_v2/output/ckpts/ '
            print(command)
            os.system(command)


#增加数据
#模型配置调整
def eval_tsfmrnn():
    command = f'{python} run_task_disf.py  --do_eval --model_type tsfmrnn_tagging --config_name ./raw_disf/electra_tagging_small.json '\
    '--vocab_file ./raw_disf/vocab.txt '\
    '--init_checkpoint ./output/ckpts/tagging_20240327_114014/checkpoint-epoch0-final/model.safetensors  '\
    '--dev_file ./data/dev.txt '\
    '--nline -1'
    os.system(command)

if __name__ == '__main__':
    run_train_rounds_base()
    #eval_tsfmrnn()
