import os
import random
from typing import Counter

import torch

from torch.utils.data import IterableDataset
import tokenization
import utils
import logging
logger = logging.getLogger('__log__')

def log_everyn(message, i, level='info', n=20000):
    if i % n == 0:
        logger.info(f"{i} {message}")

def collate_fn_tagging(data):
    def merge(datas):
        bs = len(datas)
        lens = [len(d['tokens']) for d in datas]
        maxlen =  max(lens)
        
        input_ids = torch.zeros(bs, maxlen).long()
        segment_ids = torch.zeros(bs, maxlen).long()
        input_mask  = torch.zeros(bs, maxlen).long()
        ner_tag_ids = torch.zeros(bs, maxlen).long() #edit tag

        input_tokens = []
        input_tags = []
        
        for i, data in enumerate(datas):
            _input_ids = data['tokens_ids']
            #_segment_ids = data['segment_ids']
            _mask_ids = [1.0] *  len(_input_ids)
            _ner_tag_ids = data['ner_tags_ids']
            input_tokens.append(data['tokens'])
            input_tags.append(data['ner_tags'])
           
            end = lens[i]
            input_ids[i, : end] = torch.tensor(_input_ids, dtype=torch.long)
            segment_ids[i, : end] = torch.tensor(_segment_ids, dtype=torch.long)
            input_mask[i, :end] = torch.tensor(_mask_ids, dtype=torch.long)
            ner_tag_ids[i, : end] = torch.tensor(_ner_tag_ids, dtype=torch.long)
        
        return input_ids, segment_ids, input_mask, ner_tag_ids, input_tokens, input_tags
    
    exa = merge(data)
    return exa



class NerIterableDataset(IterableDataset):
    def __init__(self, args, input_file):
        self.args = args
        self.input_file = input_file
        self.end = self._get_file_info()
        self.nline = args.nline
        self.labels = utils.read_json(args.label_map_file)
        self.max_seq_length = args.max_seq_length
        self.tokenizer = tokenization.FullTokenizer(args.vocab_file)

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
        
        for i, source_target in enumerate(utils.yield_sources_and_targets(input_file, 
                                                                          args.input_format, 
                                                                          self.labels, 
                                                                          self.tokenizer, 
                                                                          self.max_seq_length)):
            if nline > 0 and i > nline:
                break
            log_everyn(f'{i} examples processed {source_target}', i)
            yield source_target
        


if __name__ == '__main__':
    pass
