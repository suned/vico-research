from functools import wraps, partial
from typing import TypeVar, Generic

A = TypeVar('A')


def identity(v: A) -> A:
    return v
