from typing import TypeVar, Callable, Any, Generic, Tuple, Type
from f.applicative import Applicative
from f.functor import Functor
from f.monad import Monad
from .util import identity

V = TypeVar('V')
C = TypeVar('C')
N = TypeVar('N')


class Reader(Monad,
             Generic[C, V]):
    def __init__(self, f: Callable[[C], V]) -> None:
        self._f = f

    @staticmethod
    def pure(value: V) -> 'Reader[C, V]':
        return Reader(lambda _: value)

    @staticmethod
    def ask(_: Type[C]) -> 'Reader[C, C]':
        # Without the type parameter, mypy can't do inference
        # in client code
        return Reader(identity)

    def __call__(self, w: C) -> V:
        return self._f(w)

    def bind(self, f: 'Callable[[V], Reader[C, N]]') -> 'Reader[C, N]':
        return Reader(lambda w: f(self(w))(w))

    def map(self, _):
        pass

    def skip(self, n: 'Reader[C, N]') -> 'Reader[C, N]':
        return self.bind(lambda _: n)

    def __and__(self, n: 'Reader[C, N]') -> 'Reader[C, N]':
        return self.skip(n)

    def apply(self, f: 'Functor') -> 'Applicative':
        raise NotImplementedError()

    # todo: find a way to type this properly in the Monad super class
    def __rshift__(self, f: 'Callable[[V], Reader[C, N]]') -> 'Reader[C, N]':
        return self.bind(f)


def test() -> Reader[str, int]:
    return Reader.ask(str) >> (lambda env: Reader.pure(1))


def t() -> Reader[str, int]:
    return Reader.ask(str) & Reader.pure(1)

