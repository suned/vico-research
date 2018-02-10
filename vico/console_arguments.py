import argparse
import logging

from serum import Component

from vico import Config


class LogLevelAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        log_level = getattr(logging, values)
        setattr(namespace, self.dest, log_level)


def _get(_) -> Config:
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
        '--database-path',
        type=str,
            default=default.database_path
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
    parser.add_argument(
        '--targets',
        type=str,
        nargs='+',
        default=default.targets
    )
    parser.add_argument(
        '--filters',
        type=int,
        default=default.filters
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=default.n_samples
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=default.window_size,
        choices=[5, 11, 21]
    )
    parser.add_argument(
        '--skip',
        type=str,
        nargs='+',
        default=default.skip
    )
    args = parser.parse_args()
    return Config(**vars(args))


class ConsoleArguments(Component):
    get = _get

