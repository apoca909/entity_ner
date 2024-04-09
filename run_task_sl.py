import os
import random
import sys
import time

import numpy as np
import torch

import flags
import utils
import logging
from logging import handlers
from tensorboardX import SummaryWriter


from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers.models.bert import BertConfig
from transformers.models.electra import ElectraConfig
from transformers import ElectraTokenizer, get_linear_schedule_with_warmup, AdamW
from seqeval.metrics import classification_report
import tokenization
from data_loader import FelixIterableDataset, collate_fn_tagging
from modeling_sl import  ElectraForTokenClassification
from datetime import datetime


MODEL_CLASSES = {
    'tsfmrnn_tagging' : (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
    }
today = datetime.now().strftime(f'%Y%m%d_%H%M%S')
logger = logging.getLogger('__log__')

def get_point_score(golden_tags, pred_tags, masks, tokens):

    ttag = 0
    ftag = 0
    for i, golden_tag in enumerate(golden_tags):
        pred_tag = pred_tags[i]
        mask_i = masks[i]
        for j, v in enumerate(golden_tag):
            if mask_i[j] == 0: continue
            if golden_tag[j] == pred_tag[j]:
                ttag += 1
            else:
                ftag += 1
                # gold_points = [tokens[i][k] for k, p in enumerate(golden_tag) if mask_i[k] == 1 and p != 0]
                # pred_points = [tokens[i][k] for k, p in enumerate(pred_tag) if mask_i[k] == 1 and p != 0]
                # import pdb
                # pdb.set_trace()

    acc = ttag / (ttag + ftag)

    return f'{acc:.4}'

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def save_checkpoint(args, model, output_dir):
    logger.info("Before Saving model checkpoint to %s", output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    #model_to_save.save_pretrained(output_dir)
    path = os.path.join(output_dir, 'pytorch_model.bin')
    state_dict = model.state_dict()
    torch.save(state_dict, path)
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    args_writer = open(os.path.join(output_dir, 'training_args.txt'), 'w', encoding='utf-8')
    print(f'{args}', file=args_writer)
    logger.info("Saving model checkpoint to %s", output_dir)

def save_pred_result( save_filename, line_tokens, masks, preds, labels=None, golds=None, args=None):
    assert len(line_tokens) == len(masks)
    assert len(line_tokens) == len(preds)
    if golds is not None:
        assert len(line_tokens) == len(golds)

    save_file=open(save_filename, 'w', encoding='utf-8')
    #errs_file = open(save_filename+'.token.txt', 'w', encoding='utf-8')
    err_parts = {}
    for i, v in enumerate(line_tokens):
        _tokens = v
        _golds = golds[i]
        _preds = preds[i]

        if args.save_mode == 'only_pred':
            raw = utils.only_pred(_tokens, _preds)
        elif args.save_mode == 'gold_pred':
            raw = utils.gold_pred(_tokens, _golds, _preds)
        else:
            raise
        save_file.write(f"{raw}\n")
        #错误分析
        errs = get_err_tag(raw)
        for e in errs:
            err_parts[e] = err_parts.get(e, 0) + 1
    save_file.close()

    for k, v in err_parts.items():
        #k = k[-4:] + '_' + k[0:-4] 
        #print(f'{k}\t{v}\t{len(k)-4}', file=errs_file)
        pass

    #errs_file.flush()

def get_err_tag(rawline):
    tups = rawline.split(' ')
    errs = []
    for t in tups:
        if t.endswith('/O/E'):
            errs.append(t)
        elif t.endswith('/O/F'):
            #errs.append(t)
            pass
        elif t.endswith('/F/O'):
            #errs.append(t)
            pass
        elif t.endswith('/E/O'):
            #errs.append(t)
            pass
    return errs

def train(args, model, taskdir):
    message = f'*****running training*****\n' \
              f'ngpu = {args.n_gpu}\n' \
              f'init = {args.init_checkpoint}\n' \
              f'num epoch {args.num_train_epochs}\n' \
              f'batch size per gpu {args.per_gpu_train_batch_size}\n' \
              f'gradient accumulation steps {args.gradient_accumulation_steps}\n' \
              f'lr {args.learning_rate}\n' \
              f'train_file {args.train_file}\n' \
              f'input_format {args.input_format}\n' \
              f'Note {args.note}\n'
    logger.info(message)
    writer = SummaryWriter(os.path.join(taskdir, 'summary'))

    train_dataset = FelixIterableDataset(args, args.train_file)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataloader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=args.train_batch_size, 
                                  shuffle=False, collate_fn=collate_fn_tagging, drop_last=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    total_bs = args.train_batch_size * args.gradient_accumulation_steps * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    message = f'*****running training*****\n' \
              f'num examples {len(train_dataset)}\n' \
              f'total train batch size(w. parallel, distributed & accumulation) {total_bs}\n' \
              f'total optimization steps {t_total}\n'
    logger.info(message)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, amsgrad=True)
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_fuse_f1 = 0
    best_ep = 0
    model.zero_grad()
    train_iter = trange(args.num_train_epochs, desc='Epoch', disable=args.local_rank not in [-1, 0])
    for epoch_idx in train_iter:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.task == 'tagging':
                inputs = {'input_ids': batch[0],
                          'attention_mask':batch[2],
                          'punc_labels': batch[3],
                          'cws_labels': batch[4],}

            outputs = model(**inputs)
            if len(outputs) == 5:
                #n_gpu == 1
                loss = outputs[0]
                punc_logits = outputs[1]
                punc_logits_ids = outputs[2]
                log_str = f'loss {loss:.5}, '

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                #loss = torch.abs((loss - b)) + b
            else:
                pass

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 :
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    logging_loss = tr_loss
                    logger.info(f'{log_str} lr {scheduler.get_lr()[0]:.5}')
                    writer.add_scalar('step/tr_loss', loss, global_step)


        if args.local_rank in [-1, 0]:
            output_dir = os.path.join(taskdir, 'checkpoint-epoch{}-final'.format(epoch_idx))
            os.makedirs(output_dir, exist_ok=True)
            if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                punc_f1, val_loss, cws_f1 = evaluate(args, model, )
                writer.add_scalar('epoch/val_loss', val_loss, epoch_idx)
                writer.add_scalar('epoch/punc_f1', punc_f1, epoch_idx)
                writer.add_scalar('epoch/cws_f1', cws_f1, epoch_idx)
                if punc_f1 > best_fuse_f1:
                    logger.info(f'f1 update epoch {epoch_idx} fuse_f1: {best_fuse_f1} -> {punc_f1}')
                    best_fuse_f1 = punc_f1
                    best_ep = epoch_idx
                    save_checkpoint(args, model, output_dir)
                else:
                    logger.info(f'f1 not updated {epoch_idx} fuse_f1: {punc_f1}; best: {best_fuse_f1} epoch: {best_ep}')
            else:
                save_checkpoint(args, model, output_dir)


    #save_checkpoint(args, model, output_dir)  #save the last epoch.

    return global_step, tr_loss / global_step


def evaluate(args, model, ):
    #eval single tagging or insertion
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    label_ids = utils.read_label_map(args.label_map_file)
    punc_id_labels = {v:k for k, v in label_ids['punc'].items()}
    cws_id_labels = {v:k for k, v in label_ids['cws'].items()}
    eval_dataset = FelixIterableDataset(args, args.dev_file)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size, collate_fn=collate_fn_tagging)
    tokenizer = tokenization.FullTokenizer(args.vocab_file)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    tok_ids = []
    punc_label_golds = []
    punc_label_preds = []

    cws_label_golds = []
    cws_label_preds = []
    masks = []
    t0 = time.time()
    model.eval()
    batches = 0
    val_loss = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batches += 1
        batch = tuple(t.to(args.device) for t in batch)
        if args.task == 'tagging':
            inputs = {  'input_ids': batch[0],
                        'attention_mask':batch[2],
                        'punc_labels': batch[3],
                        'cws_labels': batch[4],
                     }

        if batches == 1 and args.do_train is False:
            script_model = torch.jit.trace(model, batch[0])
            #torch.jit.script(model)
            logger.info(f'code {script_model.code}')

            logger.info(f'save script model to {args.init_checkpoint}.script')
            torch.jit.save(script_model, f"{args.init_checkpoint}.script")

            _logits = script_model.forward(batch[0],)[2]
            logger.info(f'test script out {_logits}')

            script_model2 = torch.jit.load(f"{args.init_checkpoint}.script")
            _logits2 = script_model2.forward(batch[0], )[2]
            logger.info(f'test loaded script out {_logits2}')


        with torch.no_grad():
            outputs = model(**inputs)

        if len(outputs) == 5:
            loss = outputs[0]
            tag_logits = outputs[1]
            punc_logits_ids = outputs[2]
            cws_logits_ids = outputs[4]

            log_str = f'loss {loss:.5}, '
            val_loss += loss
            nb_eval_steps += 1

        if args.task == 'tagging':
            ba_punc_pred = punc_logits_ids.detach().cpu().numpy().tolist()
            ba_tok_ids = batch[0].detach().cpu().numpy().tolist()
            ba_mask = batch[2].float().detach().cpu().numpy().tolist()
            ba_punc_gold = batch[3].detach().cpu().numpy().tolist()
            ba_cws_pred  = cws_logits_ids.detach().cpu().numpy().tolist()
            ba_cws_gold  = batch[4].detach().cpu().numpy().tolist()

            tok_ids.extend(ba_tok_ids)
            masks.extend(ba_mask)
            punc_label_golds.extend(ba_punc_gold)
            punc_label_preds.extend(ba_punc_pred)

            cws_label_golds.extend(ba_cws_gold)
            cws_label_preds.extend(ba_cws_pred)

    t1 = time.time()
    bss = batches / (t1 - t0)
    logger.info(f'finished predict, timecost {t1 - t0:.4}, batches/sec {bss:.4} \nckpt:{args.init_checkpoint}')
    tokens = [tokenizer.convert_ids_to_tokens(indices) for indices in tok_ids] 
    #sequence labeling 

    gold_punc_tags = [[punc_id_labels[v]  for j, v in enumerate(arr) if masks[i][j] == 1] for i, arr in enumerate(punc_label_golds)]
    pred_punc_tags = [[punc_id_labels[v]  for j, v in enumerate(arr) if masks[i][j] == 1 ] for i, arr in enumerate(punc_label_preds) ]

    for i, v1 in enumerate(gold_punc_tags):
        for j, v2 in enumerate(v1):
            if gold_punc_tags[i][j] == 'O':
                pass
            else:
                gold_punc_tags[i][j] = 'S-' + gold_punc_tags[i][j]

    for i, v1 in enumerate(pred_punc_tags):
        for j, v2 in enumerate(v1):
            if pred_punc_tags[i][j] == 'O':
                pass
            else:
                pred_punc_tags[i][j] = 'S-' + pred_punc_tags[i][j]

    tokens = [[v  for j, v in enumerate(arr) if masks[i][j] == 1 ] for i, arr in enumerate(tokens) ]

    sl_report_punc = classification_report(gold_punc_tags, pred_punc_tags, digits=4, output_dict=True)
    sl_report_punc_s = classification_report(gold_punc_tags, pred_punc_tags, digits=4, output_dict=False)

    logger.info(f'\n{sl_report_punc_s}\n')

    output_eval_file = os.path.join(eval_output_dir, os.path.basename(args.dev_file[0]) + '.pred')
    t2 = time.time()
    logger.info(f'finished compute_metrics, timecost {t2 - t0:.4} save result to {output_eval_file}')

    #去掉S-标记
    for i, v1 in enumerate(gold_punc_tags):
        for j, v2 in enumerate(v1):
            if gold_punc_tags[i][j][0] == 'S':
                gold_punc_tags[i][j] =  gold_punc_tags[i][j][2:]

    for i, v1 in enumerate(pred_punc_tags):
        for j, v2 in enumerate(v1):
            if pred_punc_tags[i][j][0] == 'S':
                pred_punc_tags[i][j] = pred_punc_tags[i][j][2:]
    save_pred_result(output_eval_file, tokens, masks, pred_punc_tags, labels=None, golds=gold_punc_tags, args=args)

    gold_cws_tags = [[cws_id_labels[v]  for j, v in enumerate(arr) if masks[i][j] == 1] for i, arr in enumerate(cws_label_golds)]
    pred_cws_tags = [[cws_id_labels[v]  for j, v in enumerate(arr) if masks[i][j] == 1 ] for i, arr in enumerate(cws_label_preds) ]

    sl_report_cws = classification_report(gold_cws_tags, pred_cws_tags, digits=4, output_dict=True)
    sl_report_cws_s = classification_report(gold_cws_tags, pred_cws_tags, digits=4, output_dict=False)
    logger.info(f'\ncws:\n')
    logger.info(f'\n{sl_report_cws_s}\n')
    
    return sl_report_punc['weighted avg']['f1-score'], val_loss / nb_eval_steps, sl_report_cws['weighted avg']['f1-score']


def main():
    args = flags.config_opts()
    global today

    if args.do_train == False:
        today = 'eval'

    taskdir = os.path.join(args.output_dir, args.task + '_' + today)
    if not os.path.exists(taskdir):
        os.makedirs(taskdir)
    format_str = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s')
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    th = handlers.TimedRotatingFileHandler(filename=os.path.join(taskdir, '_log_.log'),
                                           when='D',
                                           backupCount=3,
                                           encoding='utf-8')
    sh.setFormatter(format_str)
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)
    logger.info(f'taskdir {taskdir}')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite or rename.")

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    logger.info(f'Process rank:{args.local_rank}, device:{device}, n_gpu:{args.n_gpu}, dist training:{bool(args.local_rank != -1)}, 16-bits:{args.fp16}')
    set_seed(args)
    logger.info("Training/evaluation parameters %s", args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    label_map = utils.read_label_map(args.label_map_file)
    tokenizer = tokenization.FullTokenizer(args.vocab_file)

    args.model_type = args.model_type.lower()
    assert (args.task =='tagging' and args.model_type.find('tagging') >-1 ), 'task and model mismatch'

    config_class, model_class, _ = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name)
    if args.task == 'tagging':
        config.punc_num_labels = len(label_map['punc'])
        config.cws_num_labels = len(label_map['cws'])
        config.vocab_size = len(tokenizer.vocab)
        config.weight =  [float(args.weight[0]), float(args.weight[1])] 
    
    if args.init_checkpoint is not None:
        #checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
        #model.load_state_dict(checkpoint, strict=False)
        model = model_class.from_pretrained(args.init_checkpoint, config=config)
        logger.info(f'ckpt loaded {args.init_checkpoint} done.')
    else:
        logger.info('init ckpt is None, cold start.')
        model = model_class(config=config)
        if args.do_train is False and args.do_eval is True:
            logger.warn('init ckpt is None, please check!')
            raise
    num_params = sum(p.numel() for p in model.parameters())
    logger.info('the number of model params: {}'.format(num_params))
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.do_train:
        model.to(args.device)
        train(args, model, taskdir)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0] and \
                (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)


    if args.do_eval and args.local_rank in [-1, 0]:
        #model = model_class.from_pretrained(args.init_checkpoint, config=config)

        model.to(args.device)
        evaluate(args, model)


    #repeat
    logger.info("Training/evaluation parameters %s", args)


if __name__ == "__main__":
    #CUDA_VISIBLE_DEVICES=0 python run_task_disf.py 
    #CUDA_VISIBLE_DEVICES=1 python run_task_disf.py --task insertion --model_type bert_insertion --config_name ./raw_disf/bert_insertion.json --init_checkpoint /evafs/angzhao/ElectraRes/chinese-bert-wwm-ext/pytorch_model.bin
    main()

