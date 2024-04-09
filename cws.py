from ltp import LTP
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
ltp = LTP(pretrained_model_name_or_path="LTP/base", 
          map_location=device)


def ltp_cws(text):
    words = ltp.pipeline(text, tasks=['cws'])['cws']
    return words

def proc_file(path):
    fw = open(path+'.seg', 'w', encoding='utf-8')
    for line in open(path, 'r', encoding='utf-8'):
        words = ltp_cws(line.strip())
        print(' '.join(words), file=fw)
    fw.flush()
    fw.close()

if __name__ == '__main__':
    proc_file('./data/train.txt')
    proc_file('./data/dev.txt')
    proc_file('./data/test.txt')