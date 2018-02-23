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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
        help='Path to SQLite database with pre-processed data',
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
        help='name of directory in which to store validation results',
        default=default.output_dir
    )
    parser.add_argument(
        "--patience",
        type=int,
        help='patience parameter for early stopping. Number of epochs without'
             ' improvement to tolerate before stopping.',
        default=default.patience
    )
    parser.add_argument(
        '--embedding-path',
        type=str,
        help='path to pickled pre-trained embedding dictionary to load. '
             'This dictionary maps indices to word vectors',
        default=default.embedding_path
    )
    parser.add_argument(
        '--indices-path',
        type=str,
        help='path to pickled indices dictionary. This dictionary maps words to'
             ' indices.',
        default=default.indices_path
    )
    parser.add_argument(
        '--targets',
        type=str,
        nargs='+',
        help='Name of target tasks to run in a multi-task learning experiment',
        default=default.targets
    )
    parser.add_argument(
        '--filters',
        type=int,
        help='Number of filters to include pr. filter size in the '
             'convolutional layer',
        default=default.filters
    )
    parser.add_argument(
        '--n-samples',
        help='Number of HTML pages to run the experiment on. '
             'Useful for testing/debugging purposes',
        type=int,
        default=default.n_samples
    )
    parser.add_argument(
        '--window-size',
        type=int,
        help='Size of the window to use in IO label prediction setup',
        default=default.window_size,
        choices=[5, 11, 21]
    )
    parser.add_argument(
        '--skip',
        type=str,
        help='Languages to skip in leave-one-language-out cross validation. '
             'Useful for stopping and resuming experiments',
        nargs='+',
        default=default.skip
    )
    args = parser.parse_args()
    return Config(**vars(args))


class ConsoleArguments(Component):
    get = _get

