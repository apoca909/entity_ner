import os
import random
from typing import Counter

import torch

from torch.utils.data import IterableDataset
import flags
import preprocess
import utils
import logging
logger = logging.getLogger('__log__')

def log_everyn(message, i, level='info', n=20000):
    if i % n == 0:
        logger.info(f"{i} {message}")

def collate_fn_tagging(data):
    def merge(datas):
        bs = len(datas)
        lens = [len(d['input_ids']) for d in datas]
        maxlen =  max(lens)
        
        input_ids = torch.zeros(bs, maxlen).long()
        segment_ids = torch.zeros(bs, maxlen).long()
        input_mask  = torch.zeros(bs, maxlen).long()
        
        punc_edit_ids = torch.zeros(bs, maxlen).long() #edit tag
        cws_edit_ids  = torch.zeros(bs, maxlen).long()
        
        for i, data in enumerate(datas):
            _input_ids = data['input_ids']
            _segment_ids = data['segment_ids']
            _punc_edit_ids = data['punc_edit_ids']
            _cws_edit_ids  = data['cws_edit_ids']
            _mask_ids = data['input_mask']
           
            end = lens[i]
            input_ids[i, : end] = torch.tensor(_input_ids, dtype=torch.long)
            segment_ids[i, : end] = torch.tensor(_segment_ids, dtype=torch.long)
            input_mask[i, :end] = torch.tensor(_mask_ids, dtype=torch.long)
            punc_edit_ids[i, : end] = torch.tensor(_punc_edit_ids, dtype=torch.long)
            
            cws_edit_ids[i, :end]   = torch.tensor(_cws_edit_ids, dtype=torch.long)

        return input_ids, segment_ids, input_mask, punc_edit_ids, cws_edit_ids
    
    exa = merge(data)
    return exa


def collate_fn_cls(data):
    def merge(datas):
        bs = len(datas)
        lens = [len(d['input_ids']) for d in datas] + [15]
        maxlen =  max(lens)
        #lmids_len = 20
        input_ids_p = torch.zeros(bs, maxlen).long()
        disf_cls_p = torch.zeros(bs).long()

        for i, data in enumerate(datas):
            input_ids = data['input_ids']
            disf_cls = data['cls_tag']
            end = lens[i]
            input_ids_p[i, : end] = torch.tensor(input_ids, dtype=torch.long)
            disf_cls_p[i] = disf_cls

        kv = {'input_ids': input_ids_p,  'disf_cls':disf_cls_p}

        return input_ids_p, disf_cls_p
    exa = merge(data)
    return exa

class FelixIterableDataset(IterableDataset):
    def __init__(self, args, input_file):
        self.args = args
        self.input_file = input_file
        self.end = self._get_file_info()
        self.nline = args.nline

    def __iter__(self):
        for exa in self.build_data(self.args, self.input_file, self.nline):
            yield exa

    def __len__(self):
        if self.nline > -1:
            return self.nline
        else:
            return self.end

    def _get_file_info(self):
        cnt = 0
        for f in self.input_file:
            for i, _ in enumerate(open(f, 'r', encoding='utf-8')):
                cnt += 1
        return cnt


    def build_data(self, args, input_file, nline):
        random.seed(args.seed)
        logger.info(f'task preprocess {input_file}')
        builder = preprocess.initialize_builder(args.label_map_file, args.max_seq_length, args.vocab_file, args.do_lower_case)
        num_tagging_skip = 0
        tokens = []
        num_converted_tagging = 0
        for i, (source, target_punc, target_cws) in enumerate(utils.yield_sources_and_targets(input_file, args.input_format, builder.get_labels())):
            if nline > 0 and i > nline:
                break
            tagging_example = builder.build_example(source, target_punc, target_cws, input_format=args.input_format)
            log_everyn(f'{i} examples processed', i)
            if tagging_example is None :
                logger.info(f'exp None skip {source}')
                continue
            
            if tagging_example:
                #tagging_examples.append(tagging_example.to_pt_example())
                input_ids = tagging_example.features['input_ids']
                log_everyn(f'{source}->{target_punc} {target_cws}; input_ids:{input_ids}', i)
                num_converted_tagging += 1
            else:
                num_tagging_skip += 1

            if args.task == 'tagging':
                exa = tagging_example.to_pt_example()

            yield exa
        
        logger.info(f'Done. {num_converted_tagging} tagging  {num_tagging_skip} skipped')
        
        print(Counter(tokens))


if __name__ == '__main__':
    #python data_loader.py --dev_file ./raw_disf/rephrase.txt --input_format para
    import logging
    from logging import handlers
    args = flags.config_opts()
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    taskdir = os.path.join(args.output_dir, args.task + '_eval' )
    th = handlers.TimedRotatingFileHandler(filename=os.path.join(taskdir, '_log_.log'),
                                           when='D',
                                           backupCount=3,
                                           encoding='utf-8')
    format_str = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s')
    sh.setFormatter(format_str)
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)
    logger.info(f'taskdir {taskdir}')

    
    [f for f in FelixIterableDataset(args, args.train_file)]
