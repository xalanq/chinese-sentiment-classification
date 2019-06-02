import os
import sys
import argparse
import logging
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()
with open(args.config_path, 'r') as f:
    args = json.load(f)

os.makedirs(args['output_path'], exist_ok=True)

logFormatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger()

fileHandler = logging.FileHandler(os.path.join(args['output_path'], 'log.txt'))
fileHandler.setFormatter(logFormatter)
log.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)
log.setLevel(logging.DEBUG)

log.info('{}'.format(args))

if not torch.cuda.is_available() and args['gpu']:
    log.warning('Cannot use gpu. use cpu instead.')
    args['gpu'] = False
is_gpu = args['gpu']

with open(os.path.join(args['output_path'], 'config.json'), 'w') as f:
    json.dump(args, f)

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
if is_gpu:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

writer = SummaryWriter(log_dir=os.path.join(args['output_path'], 'runs'))


class Data:
    def __init__(self, text, label, training):
        self.text = text
        self.label = label
        self.training = training

    def split(self, size):
        return (Data(self.text[:-size], self.label[:-size], training=self.training),
                Data(self.text[-size:], self.label[-size:], training=False))

    def __len__(self):
        return len(self.text)


def load_data(input_path, max_length, training):
    text_file = os.path.join(input_path, 'train_text.npy' if training else 'test_text.npy')
    label_file = os.path.join(input_path, 'train_label.npy' if training else 'test_label.npy')
    log.info('load data from {}, {}, training: {}'.format(text_file, label_file, training))
    text = np.load(text_file, allow_pickle=True)
    label = np.load(label_file, allow_pickle=True)
    label = torch.tensor([np.array(label[i]) / np.sum(label[i]) for i in range(len(label))],
                         dtype=torch.float)
    if max_length == -1:
        text = [torch.tensor(t, dtype=torch.float) for t in text]
    else:
        text_temp = []
        for t in text:
            b = torch.tensor(t, dtype=torch.float)
            zero = torch.zeros(max_length, b.size(1))
            length = min(max_length, b.size(0))
            zero[:length, :] = b[:length, :]
            text_temp.append(zero)
        text = text_temp
    log.info('loaded. total len: {}'.format(len(text)))
    return Data(text, label, training)


class BatchGen:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        size = len(self.data)
        ids = torch.randperm(size)
        if not self.data.training:
            ids = torch.arange(size)

        for i in range(len(self)):
            batch_idx = ids[self.batch_size * i: self.batch_size * (i + 1)]
            label = torch.index_select(self.data.label, 0, batch_idx)
            if is_gpu:
                label = label.cuda()
                text = [self.data.text[j].cuda() for j in batch_idx]
            else:
                text = [self.data.text[j] for j in batch_idx]
            yield (text, label)


class Stat:
    def __init__(self, training):
        self.loss = []
        self.gold_labels = []
        self.norm_gold_labels = []
        self.pred_labels = []
        self.norm_pred_labels = []
        self.training = training
        self.save = {
            'acc': [],
            'f1': [],
            'corr': [],
        }

    def add(self, pred, gold, loss):
        gold_labels = torch.argmax(gold, dim=1).cpu().numpy()
        norm_gold_labels = gold.cpu().numpy()
        pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
        norm_pred_labels = pred.cpu().numpy()
        self.loss.append(loss)
        self.gold_labels.extend(gold_labels)
        self.norm_gold_labels.extend(norm_gold_labels)
        self.pred_labels.extend(pred_labels)
        self.norm_pred_labels.extend(norm_pred_labels)

    def eval(self):
        acc = accuracy_score(self.gold_labels, self.pred_labels) * 100
        f1 = f1_score(self.gold_labels, self.pred_labels, average='macro') * 100
        norm_gold = np.asarray(self.norm_gold_labels).transpose((1, 0))
        norm_pred = np.asarray(self.norm_pred_labels).transpose((1, 0))
        corr = sum([pearsonr(norm_gold[i], norm_pred[i])[0] for i in range(len(norm_gold))]) / len(norm_gold)
        return acc, f1, corr

    def log(self, global_step, epoch, batch):
        acc, f1, corr = self.eval()
        if self.training:
            loss = sum(self.loss) / len(self.loss)
            log.info('step: {}, epoch: {}, batch: {}, loss: {}, acc: {}, f1: {}, r: {}'.format(
                global_step, epoch, batch, loss, acc, f1, corr))
            writer.add_scalar('train_Loss', loss, global_step)
            writer.add_scalar('train_Accuracy', acc, global_step)
            writer.add_scalar('train_F1_macro', f1, global_step)
            writer.add_scalar('train_CORR', corr, global_step)
        else:
            log.info('step: {}, epoch: {}, acc: {}, f1: {}, r: {}'.format(
                global_step, epoch, acc, f1, corr))
            writer.add_scalar('dev_Accuracy', acc, global_step)
            writer.add_scalar('dev_F1_macro', f1, global_step)
            writer.add_scalar('dev_CORR', corr, global_step)
            self.save['acc'].append(acc)
            self.save['f1'].append(f1)
            self.save['corr'].append(corr)
        self.loss = []
        self.gold_labels = []
        self.norm_gold_labels = []
        self.pred_labels = []
        self.norm_pred_labels = []


class MLP(nn.Module):
    """
    b: batch_size, n: seq_len, d: embedding_size
    """
    def __init__(self, config):
        super().__init__()
        opt = config['mlp']
        self.max_length = opt['max_length']
        dropout = opt['dropout']
        u = opt['hidden_size']
        self.mlp = nn.Sequential(
            nn.Linear(self.max_length * config['embedding_size'], u),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(u, config['num_labels']),
        )
        self.loss_type = opt['loss']
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            log.fatal('Invalid loss type. Should be "l1" or "cross_entropy"')

    def forward(self, embedding, gold_labels=None):
        """
        :param embedding: [b, n, d]
        :param gold_labels: [b, num_labels]
        :return: If training, return (loss, predicted labels). Else return predicted labels
        """
        data = torch.stack(embedding)
        output = self.mlp(data.view(data.size(0), -1))
        labels = F.softmax(output, dim=1)
        if not self.training:
            return labels.detach()
        if self.loss_type == 'cross_entropy':
            loss = self.loss(output, torch.argmax(gold_labels, dim=1))
        else:
            loss = self.loss(labels, gold_labels)
        return loss, labels.detach()


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        opt = config['cnn']
        self.cnn_1 = nn.Sequential(
            nn.Conv1d(config['embedding_size'], opt['conv_1']['size'], opt['conv_1']['kernel_size'],
                      padding=opt['conv_1']['kernel_size'] // 2),
            # nn.BatchNorm1d(opt['conv_1']['size']),
            nn.ReLU(),
            nn.Dropout(opt['conv_1']['dropout']),
            nn.MaxPool1d(opt['max_pool_1']['kernel_size'], opt['max_pool_1']['stride']),
        )
        """
        self.cnn_2 = nn.Sequential(
            nn.Conv1d(opt['conv_1']['size'], opt['conv_2']['size'], opt['conv_2']['kernel_size'],
                      padding=opt['conv_2']['kernel_size'] // 2),
            nn.ReLU(),
            nn.Dropout(opt['conv_2']['dropout']),
            nn.MaxPool1d(opt['max_pool_2']['kernel_size'], opt['max_pool_2']['stride']),
        )
        """
        mlp_u = opt['fc']['hidden_size']
        self.mlp = nn.Sequential(
            nn.Linear(opt['conv_1']['size'] * opt['max_length'] // 2, mlp_u),
            nn.ReLU(),
            nn.Dropout(opt['fc']['dropout']),
            nn.Linear(mlp_u, config['num_labels']),
        )
        self.loss_type = opt['loss']
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            log.fatal('Invalid loss type. Should be "l1" or "cross_entropy"')

    def forward(self, embedding, gold_labels=None):
        """
        :param embedding: [b, n, d]
        :param gold_labels: [b, num_labels]
        :return: If training, return (loss, predicted labels). Else return predicted labels
        """
        data = torch.stack(embedding).transpose(1, 2)  # [b, d, n]
        out_1 = self.cnn_1(data)
        # out_2 = self.cnn_2(out_1)
        # output = self.mlp(out_2.view(out_2.size(0), -1))
        output = self.mlp(out_1.view(out_1.size(0), -1))
        labels = F.softmax(output, dim=1)
        if not self.training:
            return labels.detach()
        if self.loss_type == 'cross_entropy':
            loss = self.loss(output, torch.argmax(gold_labels, dim=1))
        else:
            loss = self.loss(labels, gold_labels)
        return loss, labels.detach()


class RNN(nn.Module):
    """
    b: batch_size, n: seq_len, u: rnn_hidden_size, da: param_da, r: param_r, d: embedding_size
    """

    def __init__(self, config):
        super().__init__()
        opt = config['rnn']
        u = opt['rnn_hidden_size']
        da = opt['param_da']
        r = opt['param_r']
        d = config['embedding_size']
        num_layers = opt['num_layers']
        bidirectional = opt['bidirectional']
        if opt['type'] == 'lstm':
            self.rnn = nn.LSTM(input_size=d, hidden_size=u, num_layers=num_layers,
                               bidirectional=bidirectional, batch_first=True)
        elif opt['type'] == 'gru':
            self.rnn = nn.GRU(input_size=d, hidden_size=u, num_layers=num_layers,
                              bidirectional=bidirectional, batch_first=True)
        else:
            log.fatal('Invalid rnn type. Should be "lstm" or "gru"')
        if bidirectional:
            u = u * 2
        mlp_u = opt['mlp_hidden_size']
        self.mlp = nn.Sequential(
            nn.Linear(r * u, mlp_u),
            nn.ReLU(),
            nn.Dropout(opt['dropout']),
            nn.Linear(mlp_u, config['num_labels']),
        )
        self.Ws1 = nn.Parameter(torch.randn(da, u))
        self.Ws2 = nn.Parameter(torch.randn(r, da))
        self.p_c = opt['p_coefficient']
        self.loss_type = opt['loss']
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            log.fatal('Invalid loss type. Should be "l1" or "cross_entropy"')

    def forward(self, embedding, gold_labels=None):
        """
        :param embedding: [b, n, d]
        :param gold_labels: [b, num_labels]
        :return: If training, return (loss, predicted labels). Else return predicted labels
        """
        padded = nn.utils.rnn.pad_sequence(embedding, batch_first=True)  # [b, n, d]
        H = self.rnn(padded)[0]  # [b, n, u]
        A = F.softmax(torch.matmul(self.Ws2, torch.tanh(torch.matmul(self.Ws1, H.transpose(1, 2)))), dim=2)  # [b, r, n]
        M = torch.matmul(A, H)  # [b, r, u]
        output = self.mlp(M.view(M.size(0), -1))
        labels = F.softmax(output, dim=1)
        if not self.training:
            return labels.detach()
        I = torch.eye(A.size(1))
        if is_gpu:
            I = I.cuda()
        tmp = torch.matmul(A, A.transpose(1, 2)) - I
        P = (tmp * tmp).sum() / A.size(0)
        loss = self.p_c * P
        if self.loss_type == 'cross_entropy':
            loss = self.loss(output, torch.argmax(gold_labels, dim=1))
        else:
            loss = self.loss(labels, gold_labels)
        return loss, labels.detach()


def main():
    log.info('Loading Train Data')
    batch_size = args['batch_size']
    train_data = load_data(args['input_path'],
                           -1 if args['type'] == 'rnn' else args[args['type']]['max_length'],
                           True)
    train_data, dev_data = train_data.split(len(train_data) // 10)
    log.info('Train: length: {}, total batch: {}, batch size: {}'.format(
        len(train_data), (len(train_data) + batch_size - 1) // batch_size, batch_size))
    log.info('Dev: length: {}, total batch: {}, batch size: {}'.format(
        len(dev_data), (len(dev_data) + batch_size - 1) // batch_size, batch_size))

    log.info('Loading model {}'.format(args['type']))
    model = None
    if args['type'] == 'rnn':
        model = RNN(args)
    elif args['type'] == 'cnn':
        model = CNN(args)
    elif args['type'] == 'mlp':
        model = MLP(args)
    else:
        log.fatal('Invalid type. Should be "rnn", "cnn" or "mlp"')

    if is_gpu:
        model.cuda()

    optimizer = None
    if args['optimizer'] == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args['lr'],
                                  lr_decay=args['lr_decay'], weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'],
                              weight_decay=args['weight_decay'])
    else:
        log.fatal('Invalid optimizer type. Should be "adagrad" or "sgd"')

    train_stat = Stat(True)
    eval_stat = Stat(False)
    best_epoch = -1
    best_state_dict = None

    for epoch in range(args['num_epochs']):
        log.info('*** epoch: {} ***'.format(epoch + 1))

        log.info('*** training ***')
        model.train()
        gen = BatchGen(train_data, batch_size)
        cnt = 0
        for batch, data in enumerate(gen):
            optimizer.zero_grad()
            loss, pred_labels = model(data[0], data[1])
            loss.backward()
            optimizer.step()
            cnt += 1
            if cnt == args['display_per_batch']:
                cnt = 0
                train_stat.add(pred_labels, data[1], loss.item())
                train_stat.log(epoch * len(gen) + batch + 1, epoch, batch)

        log.info('*** evaluating ***')
        model.eval()
        gen = BatchGen(dev_data, batch_size)
        for batch, data in enumerate(gen):
            with torch.no_grad():
                pred_labels = model(data[0])
                eval_stat.add(pred_labels, data[1], None)
        eval_stat.log(epoch + 1, epoch, None)
        if best_epoch == -1 or eval_stat.save['acc'][-1] > eval_stat.save['acc'][best_epoch]:
            best_epoch = epoch
            best_state_dict = model.state_dict()

    log.info('\n*** Best acc model ***\nepoch: {}\nacc: {}\nf1: {}\ncorr: {}'.format(
        best_epoch + 1, eval_stat.save['acc'][best_epoch], eval_stat.save['f1'][best_epoch],
        eval_stat.save['corr'][best_epoch]))
    writer.close()

    log.info('Loading Test Data')
    test_data = load_data(args['input_path'],
                          -1 if args['type'] == 'rnn' else args[args['type']]['max_length'],
                          False)
    log.info('Test: length: {}, total batch: {}, batch size: {}'.format(
        len(test_data), (len(test_data) + batch_size - 1) // batch_size, batch_size))

    model.load_state_dict(best_state_dict)
    model.eval()
    gen = BatchGen(dev_data, batch_size)
    for batch, data in enumerate(gen):
        with torch.no_grad():
            pred_labels = model(data[0])
            eval_stat.add(pred_labels, data[1], None)
    acc, f1, corr = eval_stat.eval()
    log.info('\n*** Test Result ***\nacc: {}\nf1: {}\ncorr: {}'.format(acc, f1, corr))


if __name__ == '__main__':
    main()
