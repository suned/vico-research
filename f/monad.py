from typing import TypeVar, Generic, Callable, Any, Type
from abc import ABC, abstractclassmethod, abstractmethod

from f.functor import Functor
from f.applicative import Applicative


class Monad(Applicative, ABC):

    def apply(self, f):
        raise NotImplementedError()

    # todo: improve typing here
    @abstractclassmethod
    def pure(cls, value):
        raise NotImplementedError()

    @abstractmethod
    def bind(self, f):
        raise NotImplementedError()

    @abstractmethod
    def skip(self, n):
        return self.bind(lambda _: n)
   
    @abstractmethod
    def __and__(self, n):
        return self.skip(n)

    @abstractmethod
    def __rshift__(self, f):
        return NotImplementedError()
      