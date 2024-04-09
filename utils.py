
import json
import random
import logging

logger = logging.getLogger('__log__')


def text_file_iterator(fnames):
    """Returns an iterator over lines of the files covered by fname_pattern."""
    do_shuffle = False
    lines = []
    for fname in fnames:
        with open(fname, "r", encoding='utf-8') as f:
            for _, line in enumerate(f):
                lines.append(line.strip())
        if fname.find('train') >= 0:
            do_shuffle = True
    logger.info(f"all lines:{len(lines)}")
    
    if do_shuffle:
        logging.info(f'do shuffle = True')
        random.shuffle(lines)
    
    for line in lines:
        yield line

def parse_raw_tagging_ner(line, labels, tokenizer, max_seq_length):
    #处理语料  line:我的名字叫 张三/name 。 
    # labels: {'S-O':0, 'S-NAME':1, }
    tokens = []
    ner_tags = []
    
    for part in line.strip('\\n').lower().split():
        pos = part.find('/')
        if part.find('/') > 0:
            word = part[0:pos]
            tag = part[pos+1:].lower()
            word_size = len(word)
            std_tag = []
            
            if tag == 'o':
                std_tag = ['o'] * word_size
            else:
                if word_size == 1:
                    std_tag = [f'S-{tag}']
                elif word_size == 2:
                    std_tag = [f'B-{tag}', f'E-{tag}']
                elif word_size > 2:
                    std_tag = [f'B-{tag}'] + [f'I-{tag}'] * (word_size - 2) + [f'E-{tag}']
                else:
                    raise

            tokens.extend(word)
            ner_tags.extend(std_tag)
        else:
            std_tag = ['O'] * len(part)
            tokens.extend(part)
            ner_tags.extend(std_tag)
    #truncate
    tokens = tokens[0:max_seq_length-2]
    ner_tags = ner_tags[0:max_seq_length-2]

    tokens = ['[CLS]'] + tokens + ['[SEP]']
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    ner_tags = ['O'] + ner_tags + ['O']
    ner_tags_ids = [labels.get(t, 0) for t in ner_tags]  #未并入标签字典的标记 转为0处理

    assert len(tokens) == len(ner_tags) == len(ner_tags_ids)
    
    return {"tokens":tokens, "tokens_ids":tokens_ids, "ner_tags":ner_tags, "ner_tags_ids":ner_tags_ids}

def yield_sources_and_targets(
        input_file_pattern,
        input_format,
        labels,
        tokenizer,
        max_seq_length, 
        nline):
    
    data_spec = {'raw':(text_file_iterator, parse_raw_tagging_ner)}

    if input_format not in data_spec:
        raise ValueError("Unsupported input_format: {}".format(input_format))
    
    file_iterator_fn, parse_fn = data_spec[input_format]
    for idx, line in enumerate(file_iterator_fn(input_file_pattern)):
        if 0 < nline < idx:
            break
        parsed_t = parse_fn(line, labels, tokenizer, max_seq_length)
        yield parsed_t


def read_json(path):
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
                raw_str += ' ' + cur[0]      #' ' + cur[0]
            elif prev[1:3] != ('O', 'O'):
                    raw_str += '/' + prev[1] + '/' + prev[2] + ' ' + cur[0]  #'/' + prev[1] + '/' + prev[2] + ' ' + cur[0]
    
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

def proc_json_tagging_to_raw_tagging(path):
    fw = open(path[0:-4] + '.raw.txt', 'w', encoding='utf-8')
    labels = set()
    for line in open(path, 'r', encoding='utf-8'):
        js = json.loads(line)
        text = js['text']
        pad = [{'entity_index': {'begin': -1, 'end': 0}, 'entity_type': 'o', 'entity': 'none'}]
        entity_list = pad + js['entity_list']
        text_new = []
        
        last_end = 0
        for i, entity in enumerate(entity_list):
            if i == 0:continue
            entity_text = entity['entity']
            entity_type = entity['entity_type'].lower()
            entity_index_begin = entity['entity_index']['begin']
            entity_index_end = entity['entity_index']['end']
            last_end = entity_list[i-1]['entity_index']['end']
            assert entity_index_end > last_end
            pre_text = text[last_end:entity_index_begin].strip()
            assert entity_text == text[entity_index_begin:entity_index_end]
            entity_raw = text[entity_index_begin:entity_index_end] + '/' + entity_type
            labels.add(entity_type)
            if len(pre_text) > 0:
                text_new.append(pre_text+"/o")
            text_new.append(entity_raw)

        text_new.append(text[last_end:])
        
        print(' '.join(text_new).strip(), file=fw)
    fw.flush()
    fw.close()
    tag_produce(" ".join(list(labels)))

def proc_cws(path):
    fw = open(path[0:-4] + '.cws.txt', 'w', encoding='utf-8')
    for line in open(path, 'r', encoding='utf-8'):
        if line.strip() == "":continue
        words = line.strip().split()
        new_words = [w+'/wd' for w in words]
        print(' '.join(new_words), file=fw)
    fw.flush()
    fw.close()
def proc_msra(path):
    fw = open(path[0:-4] + '.ner.txt', 'w', encoding='utf-8' )
    labels = set()
    tokens = [""]
    tags   = [""]
    for _line in open(path, 'r', encoding='utf-8'):
        if _line.strip() == '':
            new_lines = [z[0] + '/'+ z[1].lower() for z in zip(tokens, tags) if len(z[0]) > 0]
            print(' '.join(new_lines), file=fw)
            tokens = [""]
            tags = [""]
            continue
        t = _line.strip().split('\t')
        assert len(t) == 4, _line
        tok = t[0]
        tag = t[3]
        if tag.find('-') > 0:
            tag = tag.split('-')[1]
            labels.add(tag)
        if tag == tags[-1]:
            tokens[-1] = tokens[-1] + tok
            tags[-1] = tag
        else:
            tokens.append(tok)
            tags.append(tag)
    fw.flush()
    fw.close()
    tag_produce(" ".join(list(labels)))
def tag_produce(tag = "COMPANY_NAME PERSON_NAME TIME ORG_NAME LOCATION PRODUCT_NAME"):
    tag_map = {"o":0}
    for _tag in tag.lower().split():
        tag_map["S-" + _tag] = len(tag_map)
        tag_map["B-" + _tag] = len(tag_map)
        tag_map["I-" + _tag] = len(tag_map)
        tag_map["E-" + _tag] = len(tag_map)
    print(json.dumps(tag_map))

if __name__ == '__main__':
    pass
    #proc_json_tagging_to_raw_tagging('data/offer/train.txt')
    # proc_json_tagging_to_raw_tagging('data/test.txt')
    # proc_json_tagging_to_raw_tagging('data/val.txt')
    
    # proc_cws('./data/cws/train.txt')
    # proc_cws('./data/cws/dev.txt')
    # tag_produce("wd")
    #proc_msra('/Users/on/Downloads/oo/pos.dev.msra.txt')