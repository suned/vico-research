from typing import TypeVar, Generic, Callable, Any, cast, Type
from abc import ABC, abstractmethod
from f.applicative import Applicative
from f.functor import Functor
from f.monad import Monad

V = TypeVar('V')
N = TypeVar('N')


class Maybe(Monad, Generic[V], Functor[V], ABC):
    def __init__(self, value: V) -> None:
        super().__init__()
        self._value = value

    @classmethod
    def pure(cls, value: V) -> 'Maybe[V]':
        return Just(value)

    @property
    def value(self) -> V:
        return self._value

    @abstractmethod
    def bind(self, f: 'Callable[[V], Maybe[N]]') -> 'Maybe[N]':
        raise NotImplementedError()

    def __rshift__(self, f: 'Callable[[V], Maybe[N]]') -> 'Maybe[N]':
        return self.bind(f)

    @abstractmethod
    def apply(self, _) -> 'Maybe[N]':
        raise NotImplementedError()

    def skip(self, n: 'Maybe[N]') -> 'Maybe[N]':
        return n

    def __or__(self, f: Callable[[V], N]) -> 'Maybe[N]':
        return self.map(f)

    def __and__(self, n: 'Maybe[N]') -> 'Maybe[N]':
        return self.skip(n)

    @abstractmethod
    def map(self, f: Callable[[V], N]) -> 'Maybe[N]':
        raise NotImplementedError()

    def __eq__(self, other) -> bool:
        return self.__dict__ == other.__dict__


class Just(Maybe[V], Generic[V]):
    def apply(self, o: Maybe[N]) -> Maybe[N]:
        if not callable(self.value):
            raise ValueError('Applicative Maybe needs a callable')
        f = self.value
        return o | self.value

    def bind(self, f: Callable[[V], Maybe[N]]) -> Maybe[N]:
        return f(self.value)

    def map(self, f: Callable[[V], N]) -> Maybe[N]:
            return Just(f(self.value))

    def __repr__(self) -> str:
        return 'Just({})'.format(self.value)


class Nothing(Maybe):
    def apply(self, _) -> Maybe:
        return Nothing()

    def __init__(self):
        super().__init__(None)

    def bind(self, _) -> Maybe:
        return self

    def map(self, _) -> Maybe:
        return Nothing()

    def __repr__(self):
        return "Nothing"
