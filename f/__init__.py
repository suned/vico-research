from .functor import Functor
from .monad import Monad
from .list import List
from .monoid import Monoid
from .maybe import Just, Nothing
from .reader import Reader
from .util import (
	identity, 
	compose, 
	Immutable, 
	Unary, 
	Predicate, 
	has_type
)

from functools import partial
