import argparse
import logging

from vico.config import Config


class LogLevelAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        log_level = getattr(logging, values)
        setattr(namespace, self.dest, log_level)


def get() -> Config:
    default = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        type=str,
        help='log level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default=logging.getLevelName(default.log_level),
        action=LogLevelAction
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='path to data',
        default=default.data_dir
    )
    parser.add_argument(
        '--folds',
        type=int,
        help='number of cross validation folds',
        default=default.folds
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        help='word embedding dimension',
        default=default.embedding_dim
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='name of .csv file in which to store validation results',
        default=default.output_file
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='name of output dir',
        default=default.output_dir
    )
    parser.add_argument(
        "--use-attributes",
        type=bool,
        help='include html attributes',
        default=default.use_attributes
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help='number of epochs',
        default=default.epochs
    )
    parser.add_argument(
        "--patience",
        type=int,
        help='patience parameter for early stopping',
        default=default.patience
    )
    parser.add_argument(
        '--embedding-path',
        type=str,
        help='path to pre-trained embedding to load',
        default=default.embedding_path
    )
    args = parser.parse_args()
    return Config(**vars(args))
