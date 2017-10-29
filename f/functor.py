from typing import TypeVar, Generic, Callable
from abc import ABC, abstractmethod

A = TypeVar('A')
B = TypeVar('B')


class Functor(Generic[A], ABC):
    @abstractmethod
    def map(self, f: Callable[[A], B]) -> 'Functor[B]':
        raise NotImplementedError()

    @abstractmethod
    def __or__(self, f: Callable[[A], B]) -> 'Functor[B]':
        raise NotImplementedError()