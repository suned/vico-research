from functools import reduce
from typing import TypeVar, Generic, Sequence

M = TypeVar('M')
T = TypeVar('T')


def append(first: 'Monoid[M]', second: 'Monoid[M]'):
    return first + second


def concat(m: 'Monoid[M]', ms: 'Sequence[Monoid[M]]') -> 'Monoid[M]':
    return m @ ms


class Monoid(Generic[M]):
    @staticmethod
    def empty() -> 'Monoid[M]':
        raise NotImplementedError()

    def append(self: T, m: T) -> T:
        raise NotImplementedError()

    def concat(self, ms: 'Sequence[Monoid[M]]') -> 'Monoid[M]':
        return reduce(append, reversed(ms), self.empty())

    def __add__(self, other: 'Monoid[M]') -> 'Monoid[M]':
        return self.append(other)

    def __matmul__(self, other: 'Sequence[Monoid[M]]') -> 'Monoid[M]':
        return self.concat(other)
