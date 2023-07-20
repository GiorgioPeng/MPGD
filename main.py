#!/usr/bin/env python3

import argparse
import random
import sys
import tempfile
import time

import gc
import numpy as np
import torch
from utils.EvalHelper import EvalHelper
from utils.RandomDataReader import RandomDataReader
from utils.DataReader4npz import DataReader4npz  
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


def train_and_eval(datadir, datname, hyperpm):
    set_rng_seed(hyperpm.seed) 
    print(hyperpm.seed)
    if datname.endswith('.npz'):
        agent = EvalHelper(DataReader4npz(datname, datadir), hyperpm)
    else:
        agent = EvalHelper(RandomDataReader(datname, datadir), hyperpm)
    tm = time.time()
    best_val_acc, wait_cnt = 0.0, 0
    model_sav = tempfile.TemporaryFile()
    neib_sav = torch.zeros_like(agent.neib_sampler.nb_all, device='cpu')
    for t in range(hyperpm.nepoch):
        print('%3d/%d' % (t, hyperpm.nepoch), end=' ')
        start_time = time.time()
        agent.run_epoch(end=' ') 
        _, cur_val_acc = agent.print_trn_acc() 
        if cur_val_acc > best_val_acc:
            wait_cnt = 0
            best_val_acc = cur_val_acc
            model_sav.close()
            model_sav = tempfile.TemporaryFile()
            torch.save(agent.model.state_dict(), model_sav)
            neib_sav.copy_(agent.neib_sampler.nb_all)
            label = agent.get_labels()
            numberOfClass = agent.get_nclass()
         
        else:
            wait_cnt += 1
            if wait_cnt > hyperpm.early: 
                break
    print("time: %.4f sec." % (time.time() - tm))
    model_sav.seek(0)
    agent.model.load_state_dict(torch.load(model_sav))
    agent.neib_sampler.nb_all.copy_(neib_sav)
    return best_val_acc, agent.print_tst_acc()


def main(args_str=None):
    assert float(torch.__version__[:3]) + 1e-3 >= 0.4
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--datname', type=str, default='cora')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--nepoch', type=int, default=300,
                        help='Max number of epochs to train.') # 1784750
    parser.add_argument('--early', type=int, default=10,
                        help='Extra iterations before early-stopping.')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='Initial learning rate.')
    parser.add_argument('--reg', type=float, default=0.0036,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.35,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=1,
                        help='Number of conv layers.')
    parser.add_argument('--ncaps', type=int, default=12,
                        help='Maximum number of capsules per layer.')
    parser.add_argument('--nhidden', type=int, default=16,
                        help='Number of hidden units per capsule.')
    parser.add_argument('--routit', type=int, default=6,
                        help='Number of iterations when routing.')
    parser.add_argument('--iterat', type=int, default=8,
                        help='Number of iterations when aggregating.')
    parser.add_argument('--nbsz', type=int, default=20,
                        help='Size of the sampled neighborhood.')
    parser.add_argument('--l1', type=float, default=0.9, help='l1 coeiffient')
    parser.add_argument('--lap', type=float, default=0.5, help='lap coeiffient')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--att', type=float, default=1.0, help='weight of attention loss')
    # parser.add_argument('--visulization', type=bool, default=False, help='whether visulize the result')
    parser.add_argument('--agg', type=str, default='NR', help='the aggregation method')

    
    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())
    with RedirectStdStreams(stdout=sys.stderr):
        val_acc, tst_acc = train_and_eval(args.datadir, args.datname, args)
        print('val=%.2f%% tst=%.2f%%' % (val_acc * 100, tst_acc * 100))
    return val_acc, tst_acc


if __name__ == '__main__':
    print('(%.4f, %.4f)' % main())
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
