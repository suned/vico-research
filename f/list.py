from typing import TypeVar, Callable, Iterator, Tuple, cast, Generator, Set

from f.monoid import Monoid
from .functor import Functor, Generic
from .util import Unary

A = TypeVar('A')
B = TypeVar('B')


class List(Monoid[A], Functor[A], Generic[A]):
    @staticmethod
    def empty() -> 'List[A]':
        t = set()  # type: Set[A]
        return List(v for v in t)

    def __or__(self, f: Unary[A, B]) -> 'List[B]':
        return self.map(f)

    def append(self, m: 'List[A]') -> 'List[A]':
        return List(v for v in self.values + m.values)

    def __contains__(self, x: A) -> bool:
        return x in self.values

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[A]:
        return iter(self.values)

    def map(self, f: Unary[A, B]) -> 'List[B]':
        mapped = map(f, self.values)
        return List(mapped)

    def __init__(self, values: Iterator[A]) -> None:
        self._values = tuple(values)

    @property
    def values(self) -> Tuple[A, ...]:
        return self._values

    def __repr__(self) -> str:
        return '[' + ', '.join(repr(v) for v in self.values) + "]"
