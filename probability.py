from typing import Callable, Optional, Union
import functools
import unittest, math
from decimal import Decimal
from fractions import Fraction
import warnings
import operator as o

try:
    from IPython.display import DisplayObject, TextDisplayObject
    class Markdown(TextDisplayObject):
        def __init__(self,TextDisplayObject):
            import markdown as md
            self.html = md.markdown(TextDisplayObject)
        def _repr_html_(self):
            return self.html
except:
    display = print
    Markdown = lambda x: x
    Latex = lambda x: x

def in_notebook():
    """
    Returns:
        `True` if the module is running in interactive Jupyter notebook and
        `False` if in terminal
    """
    import sys
    return 'ipykernel' in sys.modules
in_notebook()

class P:
    """
        Arithmetics utility functions for discrete probability objects `P`

        Allowed operators:
            apply simple addition: a + b (result of up to +1)
            apply simple subtraction: a - b (result of up to -1)
            apply multiplicative rule of probability: a & b = operator.and_(a, b) = callable via overloaded bitwise and operator
            apply additive rule of probability: a | b = operator.or_(a, b) = callable via overloaded bitwise or operator

        Disallowed operators:
            apply truth checks (and, or): disallowed to avoid confusion
    """
    def __init__(self, value: float):
        if value < -1 or value > 1:
            raise ValueError("Probability value must be between -1 and 1")
        self.value = value
        self.val = self.value
        self.v = self.value
        self.as_float = self.value

    def __str__(self):
        return f"P({self.v})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other: Union[float, "P"], independent=True) -> "P":
        if isinstance(other, float):
            other = P(value=other)
        if independent:
            result = P(self.v + other.v)
            display(Markdown(
                f"{self!r} + {other!r} (independent) = {self.v} + {other.v} = {result.v}"
            ))
            return result
        else:
            result = P(self.v + other.v - (self.v * other.v))
            display(Markdown(
                f"{self} + {other} (dependent) = {self.v} + {other.v} - ({self.v} * {other.v}) = {result.v}"
            ))
            return result
    def p_add(self, other: Union[float, "P"], independent=True) -> "P":
        return self.__add__(other, independent=independent)

    def __mul__(self, other: Union[float, "P"]) -> "P":
        """
        Apply multiplicative rule of probability: 
        P(A and B) = P(A) * P(B)
        """
        if isinstance(other, float):
            other = P(value=other)
        result = P(self.v * other.v)
        display(Markdown(f"P({self.v} and {other.v}) := {self!r} * {other} = {self.v} * {other.v} = {result.v}"))
        return result

    def __or__(self, other: Union[float, "P"]) -> "P":
        """
        Apply additive rule of probability: 
        P(A or B) = P(A) + P(B) - P(A and B)
        """
        if isinstance(other, float):
          other = P(value=other)
        p_A_and_B = self.__mul__(other)
        display(Markdown(f"P({self.v} or {other.v}) := {self!r} + {other} - ({self} and {other} = {self.v:.2f} + {other.v:.2f} - {p_A_and_B}"))
        return self + other - p_A_and_B
    def p_or(self, other: Union[float, "P"]) -> "P":
        return self.__or__(other)

    def __and__(self, other: Union[float, "P"]) -> "P":
        return self.__mul__(other)
    def p_and(self, other: Union[float, "P"]) -> "P":
            return self.__and__(other)

    def __sub__(self, other: Union[float, "P"], independent=True) -> "P":
        if isinstance(other, P):
            other_sub = P(value = -other.v)
        else:
            other_sub = P(value = -other)
        return self.__add__(other=other_sub, independent=independent)
    def p_sub(self, other: "P", independent=True) -> "P":
        return self.__sub__(other, independent=independent)
    
    def __bool__(self):
        warnings.warn("P has no truth value (use | for or and & for and instead)")
        return False

def test_probability_class():
    # Test additive rule of probability
    p1 = P(0.5)
    p2 = P(0.3)
    assert math.isclose((p1 | p2).v, 0.65)
    assert math.isclose(o.or_(p1, p2).v, 0.65)

    # Test multiplicative rule of probability
    p3 = P(0.4)
    p4 = P(0.2)
    assert math.isclose((p3 & p4).v, 0.08)
    assert math.isclose((o.and_(p3, p4).v), 0.08)

    # Test error checking for probabilities greater than 1
    try:
        p5 = P(1.1)
    except ValueError as e:
        assert str(e) == "Probability value must be between -1 and 1"

    # Test error checking for probabilities less than 0
    try:
        p6 = P(-0.1)
    except ValueError as e:
        assert str(e) == "Probability value must be between -1 and 1"

    # Test non-independence
    p7 = P(0.5)
    p8 = P(0.3)
    assert (p7 | p8).v == 0.65
    assert (p7 & p8).v == 0.15

test_probability_class()

class P_fn:
    """
    Probability class to handle probability functions, conditions, and operations.
    """
    def __init__(self, value: Union[Callable, float], condition: Optional[Callable] = None):
        """
        Initialize a probability function P_fn with an optional condition.

        Args:
        - value (Union[Callable, float]): The main probability function or a constant probability value.
        - condition (Optional[Callable]): An optional condition function.
        """
        self.value = value
        self.condition = condition

    def __call__(self, *args):
        """
        Evaluate the probability function or return the constant value.

        Args:
        - *args: Arguments for the probability function.

        Returns:
        - float: The result of the probability function or the constant value.
        """
        if callable(self.value):
            return self.value(*args) if self.condition is None or self.condition(*args) else 0
        else:
            return self.value if self.condition is None or self.condition(*args) else 0

    def given(self, condition: Optional[Callable], replace=True) -> 'P_fn':
        """
        Create a new probability function with the given condition.

        Args:
        - condition (Callable): The condition function.

        Returns:
        - P_fn: New probability function with the given condition.
        """
        if not replace and self.condition is not None:
          return P_fn(value=self.value, condition=(self.condition and condition))
        else:
          return P_fn(value=self.value, condition=condition)

    def conditional(self, other: 'P_fn') -> 'P_fn':
        """
        Combine two probability functions with the given condition using the conditional probability formula.

        Args:
        - other (P_fn): Another probability function.

        Returns:
        - P_fn: Combined probability function with the given condition.
        """
        return P_fn(value=(lambda x: self(x) / other(x) if other(x) != 0 else 0), condition=self.condition)

    def independent(self, other: 'P_fn') -> 'P_fn':
        """
        Combine two probability functions using the independent events probability formula or the mutually exclusive events formula.

        Args:
        - other (P_fn): Another probability function.
        - mutually_exclusive (bool): A flag indicating whether the events are mutually exclusive. Default is False.

        Returns:
        - P_fn: Combined probability function.
        """
        return P_fn(lambda x: self(x) + other(x))
      
    def mutually_exclusive(self, other: 'P_fn') -> 'P_fn':
        return independent(self, other)

    def dependent(self, other: 'P_fn') -> 'P_fn':
        """
        Combine two probability functions using the independent events probability formula or the mutually exclusive events formula.

        Args:
        - other (P_fn): Another probability function.
        - mutually_exclusive (bool): A flag indicating whether the events are mutually exclusive. Default is False.

        Returns:
        - P_fn: Combined probability function.
        """
        return P_fn(lambda x: self(x[0]) * other(x[1]) if isinstance(x, tuple) else self(x) * other(x))

    def __repr__(self):
        """
    	Return a string representation of the probability function.

        Returns:
        - str: String representation of the probability function.
        """
        if callable(self.value):
            return f"P_fn({self.value.__name__} | {self.condition.__name__ if self.condition else '-'})"
        else:
            return f"P_fn({self.value})"

def test_P_fnambda():
    # Test case 1: Create a probability function for a fair six-sided die
    p_die = P_fn(lambda x: 1/6 if 1 <= x <= 6 else 0)
    assert abs(p_die(3) - 0.16666666666666666) < 1e-9

    # Test case 2: Create a probability function with a condition for the die roll to be even
    is_even = lambda x: x % 2 == 0
    p_die_even = p_die.given(is_even)
    assert p_die_even(3) == 0

    # Test case 3: Create a probability function for a fair coin flip
    p_coin = P_fn(lambda x: 1/2 if x in ['H', 'T'] else 0)

    # Test case 4: Combine the two probability functions using the independent events formula
    p_die_coin = p_die.independent(p_coin)
    assert abs(p_die_coin((3, 'H')) - 0.08333333333333333) < 1e-9

    # Test case 5: Create a conditional probability function
    p_A = P_fn(lambda x: 1/2 if x in ['A', 'B'] else 0)
    p_B = P_fn(lambda x: 1/2 if x in ['B', 'C'] else 0)
    p_A_given_B = p_A.conditional(p_B)
    print(f"{p_A_given_B=}")
    assert abs(p_A_given_B('B') - 1.0) < 1e-9
    assert abs(p_A_given_B('A') - 0.0) < 1e-9

    # Test case 6: Test the __repr__ method
    assert repr(p_die) == "P_fn(<lambda> | -)"
    assert repr(p_die_even) == "P_fn(<lambda> | <lambda>)"
