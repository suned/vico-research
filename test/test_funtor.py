from unittest import TestCase
from f import *


class MaybeTester(TestCase):
    def test_equality(self):
        self.assertEqual(Just(1), Just(1))
        self.assertEqual(Nothing(), Nothing())

    def test_inequality(self):
        self.assertNotEqual(Just(1), Nothing())
        self.assertNotEqual(Just(1), Just(''))

    def test_identity_law(self):
        self.assertEqual(Just(1) | identity, Just(1))
        self.assertEqual(Nothing() | identity, Nothing())

    def test_composition_law(self):
        def f(x):
            return x + 1

        def g(y):
            return y * 2
        h = compose(f, g)
        self.assertEqual(
            Just(1) | h,
            Just(1) | g | f
        )
        self.assertEqual(
            Nothing() | h,
            Nothing() | g | f
        )
