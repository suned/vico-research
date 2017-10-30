from functools import wraps, partial
from typing import TypeVar, Generic, Callable, NamedTuple, Any, Type

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def identity(v: A) -> A:
    return v

Unary = Callable[[A], B]


def compose(f: Unary[B, C], g: Unary[A, B]) -> Unary[A, C]:
    return lambda z: f(g(z))

Immutable = NamedTuple
Predicate = Callable[[A], bool]


def has_type(t: Type) -> Unary[Any, bool]:
    def _(v):
        return isinstance(v, t)
    return _