import os
import sys
from elmoformanylangs import Embedder
import argparse
import logging
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, required=True)
parser.add_argument('--elmo_model_path', type=str)
parser.add_argument('--vector_path', type=str)
parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--test_file', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

logFormatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger()

fileHandler = logging.FileHandler(os.path.join(args.output_path, 'log.txt'))
fileHandler.setFormatter(logFormatter)
log.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)
log.setLevel(logging.DEBUG)

log.info('=====Pre-processing=====')

log.info('{}'.format(args))

if args.type == 'elmo':
    e = Embedder(args.elmo_model_path, batch_size=2)


def work(input_path, output_text_file, output_label_file):
    log.info('Loading data')

    label_list = []
    text_list = []

    with open(input_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            data[1] = data[1].strip().split()
            label = [0 for i in range(8)]
            for i in range(0, 8):
                label[i] = int(data[1][1 + i].split(':')[1])
            label_list.append(label)
            text_list.append(data[2].strip().split())

    log.info('size: {}'.format(len(text_list)))

    seq_len = [len(x) for x in text_list]
    log.info('max seq len: {}'.format(max(seq_len)))
    log.info('ava seq len: {:.3f}'.format(sum(seq_len) / len(seq_len)))

    if args.type == 'elmo':
        log.info('Loading elmo model')
        log.info('    Loaded')
        log.info('Processing')
        text_embed_list = e.sents2elmo(text_list)
        log.info('    Done')
    elif args.type == 'word2vec':
        log.info('Loading word2vec model')

        # https://github.com/Embedding/Chinese-Word-Vectors/blob/master/evaluation/ana_eval_dense.py
        def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
            lines_num, dim = 0, 0
            vectors = {}
            with open(path, encoding='utf-8', errors='ignore') as f:
                first_line = True
                for l in f:
                    if first_line:
                        first_line = False
                        dim = int(l.rstrip().split()[1])
                        continue
                    lines_num += 1
                    tokens = l.rstrip().split(' ')
                    vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
                    if topn != 0 and lines_num >= topn:
                        break
            return vectors, dim

        vct, dim = read_vectors(args.vector_path, 0)
        # https://github.com/Embedding/Chinese-Word-Vectors/issues/23
        avg = np.zeros(dim)
        '''
        for v in vct.values():
            avg += v
        avg /= len(vct)
        '''
        log.info('    Loaded')
        log.info('Processing, dim: {}'.format(dim))
        text_embed_list = []
        for sen in text_list:
            sen_embed = []
            for w in sen:
                if w in vct:
                    w_embed = vct[w]
                else:
                    w_embed = avg
                sen_embed.append(w_embed)
            text_embed_list.append(sen_embed)
        log.info('    Done')
    else:
        log.fatal('Invalid type. Should be "elmo" or "word2vec"')

    log.info('sample: \n{}'.format(text_embed_list[0][0]))
    np.save(output_text_file, text_embed_list)
    np.save(output_label_file, label_list)


work(args.train_file,
     os.path.join(args.output_path, 'train_text.npy'),
     os.path.join(args.output_path, 'train_label.npy'))

work(args.test_file,
     os.path.join(args.output_path, 'test_text.npy'),
     os.path.join(args.output_path, 'test_label.npy'))
