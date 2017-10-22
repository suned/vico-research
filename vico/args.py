import argparse
import logging
from datetime import datetime


_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


def get():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bilstm-dim',
        type=int,
        help='dimension of the lstm hidden layers',
        default=50
    )
    parser.add_argument(
        '--log-level',
        type=str,
        help='log level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='DEBUG'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='path to data',
        default='./data'
    )
    parser.add_argument(
        '--folds',
        type=int,
        help='number of cross validation folds',
        default=2
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        help='word embedding dimension',
        default=50
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='name of .csv file in which to store validation results',
        default=_now + '.csv'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='name of output dir',
        default='./reports'
    )
    args = parser.parse_args()
    args.log_level = getattr(logging, args.log_level)
    return args


def hyper_parameters():
    args = get()
    return {
        'bilstm_dim': args.bilstm_dim,
        'embedding_dim': args.embedding_dim,
    }
