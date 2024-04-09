   # coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for Felix."""

import json
from typing import Counter

import random
import glob
import logging
import re
try:
    from cws import ltp_cws
except Exception as e:
    print(e)
logger = logging.getLogger('__log__')


def text_file_iterator(fnames):
    """Returns an iterator over lines of the files covered by fname_pattern."""
    do_shuffle = False
    lines = []
    for fname in fnames:
        with open(fname, "r", encoding='utf-8') as f:
            for _, line in enumerate(f):
                line = line.strip()
                if line:
                    if len(line) < 50 and len(lines) > 0 and len(lines[-1]) < 50:
                        lines[-1] = lines[-1] + line
                    else:
                        lines.append(line)
        if fname.find('train') >= 0:
            do_shuffle = True
    logger.info(f"all lines:{len(lines)}")
    
    if do_shuffle:
        logging.info(f'do shuffle = True')
        random.shuffle(lines)
    
    for line in lines:
        yield line

def parse_raw_tagging_punc(line, labels):
    #处理人民日报语料，转成标点标签
    tokens = []
    punc_tags = []
    cws_tags  = []
    line = line.replace('!', '！').replace('?', "？").replace(',', '，')
    words = ['|'] + line.split() + ['|']   #ltp_cws(line)
    #line = ['[CLS]'] + list(line.strip().lower()) + ['[SEP]']
    label_punc_ids = labels['punc']
    #label_cws_ids  = label_ids['cws']
    
    for _, word in enumerate(words):
        _word = list(word)
        _word_no_punc = []
        for _, token in enumerate(_word):
            if token in label_punc_ids.keys():
                #当前位置为标点, 连续标点使用靠后的
                if len(punc_tags) > 0:
                    punc_tags[-1] = token
                else:
                    pass #首位标点忽略
            else:
                tokens.append(token)
                _word_no_punc.append(token)
                punc_tags.append("O") #default tag O
        if len(_word_no_punc) == 0:
            pass
        elif len(_word_no_punc) == 1:
            cws_tags.append('S-wd')
        elif len(_word_no_punc) == 2:
            cws_tags.append('B-wd')
            cws_tags.append('E-wd')
        else:
            _cws_tag = ['B-wd'] + ['I-wd'] * (len(_word_no_punc) - 2) + ['E-wd']
            cws_tags.extend(_cws_tag)
            
    tokens[0] = "[CLS]"
    tokens[-1] = "[SEP]"
    assert len(tokens) == len(punc_tags) == len(cws_tags), f'{len(tokens)} {len(punc_tags)} {len(cws_tags)}'
    
    
    return [(' '.join(tokens), ' '.join(punc_tags), ' '.join(cws_tags) )]

def proc_quot(text):
    text0 =  text.replace('(( ))', '').replace('_', ' ')
    text1 = re.sub(u"\\<.*?>||/", '', text0) #去掉 <> /
    tokens = []
    tags = []
    text2 = proc_bracket(text1) #中括号处理
    text3 = proc_brace(' '.join(text2)) #大括号处理
    
    for item in text3:
        if type(item) is str:
            tokens.append(item)
            tags.append('O')
        elif type(item) is list:
            item = [ i for i in flatten(item)]
            if len(item) == 0:
                continue
            if item[0] in 'C':
                tags.append('C')
            elif item[0] in 'D':
                tags.append('D')
            elif item[0] in 'E':
                tags.append('E')
            elif item[0] in 'F':
                tags.append('F')
            elif item[0] in 'A':
                tags.append('A')
            else:
                continue

            tokens.append(' '.join(item[1:]))
    
    return '_'.join(tokens), '_'.join(tags)

def proc_brace(text):
    text = pad_pair(text, '{', '}')
    
    res = proc_nested(text, '[{]', '[}]')
    return res

def proc_bracket(text="[ [ [ [ that,1 + that,1 ] + that,2 ] + that,3 ] + [ that's,4 + that's4 ] ] the way it works there, {F um. } /"):
    #proc []
    text = pad_pair(text, '[', ']')
    res = proc_nested(text, '[[]', '[]]')
    tokens = []
    #flatten the nested and pick the last item
    for x in res:
        if type(x) is list:
            x = flatten(x)
            last = get_last(x)
            tokens.extend(last)
        else:
            tokens.append(x)
    
    return tokens

def proc_nested(text, left='[[]', right='[]]', sep=' '):
    pat = '({}|{}|{})'.format(left, right, sep)
    tokens = re.split(pat, text)
    stack = [[]]
    for x in tokens:
        if not x or re.match(sep, x): continue
        if re.match(left, x):
            stack[-1].append([])
            stack.append(stack[-1][-1])
        elif re.match(right, x):
            stack.pop()
        else:
            stack[-1].append(x)
    return stack.pop()

def pad_pair(text, left='[', right=']'):
    cnter = Counter(text)
    diff = cnter.get(left, 0) - cnter.get(right, 0)
    
    #pad the bracket
    if diff == 0:
        pass
    elif diff > 0:
        text = text + right * diff
    elif diff < 0:
        text = left* (-diff) + text
    return text

def flatten(x):
    for i in x:
        if type(i) is list:
            yield from flatten(i)
        else:
            yield i
    
def get_last(x):
    vals = []
    for i in x:
        if i == '+':
            vals = []
        else:
            vals.append(i)
    return vals

def yield_sources_and_targets(
        input_file_pattern,
        input_format,
        labels):
    
    data_spec = {'raw':(text_file_iterator, parse_raw_tagging_punc)}

    if input_format not in data_spec:
        raise ValueError("Unsupported input_format: {}".format(input_format))

    file_iterator_fn, parse_fn = data_spec[input_format]
    for item in file_iterator_fn(input_file_pattern):
        parsed_items = parse_fn(item, labels)
        for parsed_item in parsed_items:
            if parsed_item is not None :
                yield parsed_item


def get_filenames(patterns):
    all_files = []
    for pattern in patterns.split(","):
        # points to a specific file.
        files = glob.glob(pattern)
        if not files:
            raise RuntimeError("Could not find files matching: %s" % pattern)
        all_files.extend(files)

    return all_files


def read_label_map(path):
    ''' read the label ids'''
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith(".json"):
            labels = json.load(f)
        else:
            raise ValueError(f"invalid label map format: {path}")
    return labels

def gold_pred(toks, golds, preds):
    assert len(toks) == len(golds) == len(preds)
    token_gold_pred = [('', 'O', 'O')] + [i for i in zip(toks, golds, preds)] + [('', 'O', 'O')]
    size = len(token_gold_pred)
    raw_str = ''
    
    for k in range(1, size):
        cur = token_gold_pred[k]
        prev = token_gold_pred[k - 1]
        if prev[1:3] == cur[1:3]:
            raw_str += cur[0]
        elif prev[1:3] != cur[1:3]:
            if prev[1:3] == ('O', 'O'):
                raw_str += '' + cur[0]      #' ' + cur[0]
            elif prev[1:3] != ('O', 'O'):
                    raw_str += '/' + prev[1] + '/' + prev[2] + '' + cur[0]  #'/' + prev[1] + '/' + prev[2] + ' ' + cur[0]
    
    return raw_str.strip()

def only_pred(toks, tags):
    toks = [''] + toks + ['']
    tags = ['O'] + tags + ['O']
    # merge
    raw_str = ''
    tag_len = len(tags)
    for j in range(1, len(toks)):
        prev = j - 1
        if j < tag_len and tags[j] != tags[prev]:
            if tags[prev] == 'E':
                raw_str += '/' + tags[prev] + ' ' + toks[j]
            elif tags[prev] == 'F':
                raw_str += '/' + tags[prev] + ' ' + toks[j]
            else:
                raw_str += ' ' + toks[j]
        else:
            raw_str += toks[j]
    return raw_str.strip().replace('  ', ' ')

#
def keep_chinese_chars(text):
    def is_chinese_char(cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False
    zh_chars = []
    for c in text:
        cp = ord(c)
        if is_chinese_char(cp):
            zh_chars.append(c)
    return ''.join(zh_chars)

if __name__ == '__main__':
    text = 'As a matter of fact, [ one thing, + {A I have a young son /'
    text = "so, that's, [child_talking] /"
    
    print(parse_raw_tagging_punc('此后，兰越峰便开始在走廊上班至今', {"N": 0, "，": 1, "。": 2, "？": 3, "！": 4}))
     
