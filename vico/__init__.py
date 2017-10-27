import logging
from typing import TypeVar, Iterable

from vico import args
from vico.config import Config
from numpy import ndarray, array


def configure_root_logger(c: Config):
    log = logging.getLogger('vico')
    log.setLevel(c.log_level)

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)15s] - %(message)s',
        '%H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    log.addHandler(console_handler)


T = TypeVar('T')


def immutable_array(a: Iterable[T]) -> ndarray:
    a_ = array(a)  # type: ndarray[T]
    a_.flags.writeable = False
    return a_
