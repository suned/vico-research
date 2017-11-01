from abc import ABC, abstractclassmethod, abstractmethod
from f.functor import Functor


class Applicative(Functor, ABC):
    @abstractclassmethod
    def pure(cls, value):
        raise NotImplementedError()

    @abstractmethod
    def apply(self, f):
        raise NotImplementedError()

    def __mul__(self, f):
        return self.apply(f)
