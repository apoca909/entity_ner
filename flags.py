import argparse

idx = 0
def config_opts():
    parser = argparse.ArgumentParser(description='felix flags')

    parser.add_argument('--task', default='tagging', type=str, choices=['tagging',], 
                        help='task train tagging or insertion, if predict, both')

    parser.add_argument('--usecrf', default=True, type=bool, help=
                        'use crf ')
    parser.add_argument('--nline', default=-1, type=int, help=
                        'train samples number.')
    parser.add_argument('--config_name', default=None, type=str, help=
                        'Path to the config file for the tagging model.')
    parser.add_argument('--init_checkpoint',
                        default=None, 
                        type=str,
                        help='init from a pretrain model train or evaluate')
    #used for predict pipline

    parser.add_argument('--do_train', action='store_true', help='do train')
    parser.add_argument('--do_eval', action='store_true',help='do train')
    ######
    parser.add_argument('--input_format', default='raw', choices=['raw'])
    #
    parser.add_argument('--train_file', default=[''], nargs='+', type=str, help='train file ')
    parser.add_argument('--dev_file', default=[''], nargs='+', type=str, help='dev file')
    parser.add_argument('--save_mode', default='gold_pred', type=str, choices=['only_pred', 'gold_pred'], help='dev file')

    #
    parser.add_argument('--output_dir', default='./output/ckpts/', type=str, required=False,
                        help='the output dir of the model preds and checkpoints')
    parser.add_argument('--overwrite_outputdir', default=True, type=bool,
                        help='overwrite the content of the output dir')

    parser.add_argument("--no_cuda", default=False, type=bool, help="Avoid using CUDA when available")

    parser.add_argument('--max_seq_length', default=512, type=int, help='')

    parser.add_argument('--warmup_steps', default=0, type=int, help='Warmup steps for Adam weight decay optimizer.')

    parser.add_argument(
        '--label_file', default='./raw_disf/label_map.json', type=str, help=
        'Path to the label map file. ')
    parser.add_argument('--vocab_file', default='./raw_disf/vocab.txt', type=str,
                        help='Path to the BERT vocabulary file.')
    parser.add_argument(
        '--predict_batch_size', default=32, type=int,
        help='Batch size for the prediction of insertion and tagging models.')
    


    parser.add_argument(
        '--num_output_variants', default=0, type=int, help=
        'Number of output variants to be considered. By default, the value is set '
        'to 0 and thus, no variants are considered. Warning! This feature only '
        'makes sense if num_output_variants >= 2.')

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--max_steps', type=int, default=-1, help="max steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Num of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--logging_steps', type=int, default=1000, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--num_train_epochs', default=20, type=int, help='Total number of training epochs to perform.')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='The initial learning rate for Adam. 2e-5')
    parser.add_argument('--evaluate_during_training', default=True, type=bool, help='evaluate during training')

    parser.add_argument('--overwrite', default=True, type=bool, help='overwrite the ckpt')
    parser.add_argument('--fp16', default=False, type=bool, help='mixd precision')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument('--seed', default=42, type=int, )
    parser.add_argument('--note', default='desc note', type=str)

    args = parser.parse_args()

    return args

