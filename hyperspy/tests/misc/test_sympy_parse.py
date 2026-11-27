# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""Independent corpus, grammar, and resource checks for the expression gate."""

import ast
import importlib.util
import math
import sys
import unicodedata
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np
import pytest
import sympy as s
from scipy.special import wofz
from sympy.parsing import sympy_parser

from hyperspy.misc import _sympy_parse as parser

x, y, a, b = s.symbols("x y a b")

# This is deliberately independent of the production namespace. Constants
# belong with G1; the remaining entries are the five classes in the contract.
CLASSIFICATION = {
    "G1": "pi E I Rational".split(),
    "G2": "pow Pow sqrt".split(),
    "G3": "exp sinh cosh tanh sin cos tan erf erfc".split(),
    "G4": "gamma beta zeta".split(),
    "G5": (
        "log ln asin acos atan asinh acosh atanh wofz atan2 re im sign abs Abs "
        "floor ceiling Heaviside Min Max Piecewise Mod Eq Ne Lt Le Gt Ge "
        "And Or Xor Not where"
    ).split(),
}


def _function_tripwire(function, bound):
    """Catch unsafe concrete arguments without blocking symbolic construction."""
    original = function.eval.__func__

    def guarded(cls, *args):
        for arg in args:
            if arg.is_number and s.Abs(arg) > bound:
                raise AssertionError(
                    f"TRIPWIRE {function.__name__}: argument exceeds {bound}"
                )
        return original(cls, *args)

    return classmethod(guarded)


def _power_tripwire(number_class):
    """Detect both an excessive exponent and excessive pre-call bit growth."""
    original = number_class._eval_power

    def guarded(base, exponent):
        if exponent.is_number:
            size = s.Abs(exponent)
            bits = max(abs(base.p), base.q).bit_length()
            if size > 1024 or bits * size > 4096:
                raise AssertionError(
                    f"TRIPWIRE {number_class.__name__}: power exceeds bound"
                )
        return original(base, exponent)

    return guarded


@contextmanager
def tripwires():
    """Fail fast if a resource rejection reaches an evaluating constructor."""
    # SymPy caches constructors globally. A previously computed result must not
    # hide a missed guard or a test mutation, in either direction.
    s.core.cache.clear_cache()
    try:
        with ExitStack() as stack:
            for function, bound in (
                (s.gamma, 64),
                (s.beta, 64),
                (s.zeta, 64),
                (s.exp, 2048),
                (s.sinh, 2048),
                (s.cosh, 2048),
                (s.tanh, 2048),
            ):
                stack.enter_context(
                    patch.object(function, "eval", _function_tripwire(function, bound))
                )
            for number_class in (s.Integer, s.Rational):
                stack.enter_context(
                    patch.object(
                        number_class, "_eval_power", _power_tripwire(number_class)
                    )
                )
            yield
    finally:
        s.core.cache.clear_cache()


def _legacy_parse(text):
    """The former split-and-sympify path, used only for this literal corpus."""
    parts = iter(map(str.strip, text.split(";")))
    expression = s.sympify(next(parts))
    for substitution in parts:
        name, value = map(str.strip, substitution.split("="))
        expression = expression.subs(name, s.sympify(value))
    return expression


# Each row declares its free symbols independently; no expected input is
# derived from the result returned by the compiler.
CORPUS = [
    ("Bleasdale", "where((a + b*x) > 0, pow(a + b*x, -1/c), 0)", "a b c x"),
    (
        "PowerLaw",
        "where(left_cutoff<x, A*(-origin + x)**-r, 0)",
        "A left_cutoff origin r x",
    ),
    (
        "Voigt",
        "area * re(V); V=wofz(z)/(sqrt(2.0*pi)*sigma); z=(x-centre+1j*gamma_)/(sigma*sqrt(2.0))",
        "area centre gamma_ sigma x",
    ),
    (
        "SkewNormal",
        "2*A*normpdf*normcdf; normpdf=exp(-t**2/2)/sqrt(2*pi); normcdf=(1+erf(_shape*t/sqrt(2)))/2; t=(x-x0)/scale",
        "A _shape scale x x0",
    ),
    (
        "Doniach",
        "A*cos(0.5*pi*alpha+  ((1.0-alpha)*atan((x-centre+offset)/sigma))) / (sigma**2+(x-centre+offset)**2)**(0.5*(1.0-alpha)); offset=2.354820*sigma/(2*tan(pi/(2-alpha)))",
        "A alpha centre sigma x",
    ),
    ("HeavisideStep", "A*Heaviside(x-n,0.5)", "A n x"),
    (
        "Gaussian2D",
        "A*(1/(sigma_x*sigma_y*2*pi))*exp(-((x-centre_x)**2/(2*sigma_x**2)+(y-centre_y)**2/(2*sigma_y**2)))",
        "A centre_x centre_y sigma_x sigma_y x y",
    ),
    (
        "GaussianHF",
        "height*exp(-(x-centre)**2*4*log(2)/fwhm**2)",
        "centre fwhm height x",
    ),
    ("Lorentzian", "A/pi*(gamma_/((x-centre)**2+gamma_**2))", "A centre gamma_ x"),
    ("Logistic", "a/(1+b*exp(-c*(x-origin)))", "a b c origin x"),
    ("RC", "V0+Vmax*(1-exp(-x/tau))", "V0 Vmax tau x"),
    ("Exponential", "A*exp(-x/tau)", "A tau x"),
    (
        "Gaussian",
        "A*(1/(sigma*sqrt(2*pi)))*exp(-(x-centre)**2/(2*sigma**2))",
        "A centre sigma x",
    ),
    ("Arctan", "A*atan(k*(x-x0))", "A k x x0"),
    ("Erf", "A*erf((x-origin)/sqrt(2)/sigma)/2", "A origin sigma x"),
    ("Polynomial", "a2*x**2+a1*x**1+a0*x**0", "a0 a1 a2 x"),
    ("caret", "x^2", "x"),
    ("symbolic-power", "x ** power", "power x"),
    ("reciprocal", "1/(1+ x)", "x"),
    ("axis", "a/(x+1)+b", "a b x"),
    (
        "substitutions",
        "A*B/(A+B) ; A = sin(x)+one; B = cos(y)-two; y = tan(x)",
        "one two x",
    ),
    ("absolute", "abs(x)-1", "x"),
    ("square", "x*x", "x"),
    ("sqrt", "sqrt(x)", "x"),
    ("offset", "15+x", "x"),
    ("negative-float", "-0.5*x", "x"),
    ("divide", "x/2", "x"),
    ("Piecewise", "Piecewise((x, x > 0), (0, True))", "x"),
    ("Min-Max", "Min(x,1)+Max(x,2)", "x"),
    ("caret-precedence", "x^2+1", "x"),
    ("cancel-before-substitute", "A/A; A=0", ""),
    ("unicode", "\u03c3*x", "x \u03c3"),
    ("multiline", "\n  a * x\n  + b\t\r\f\v", "a b x"),
    ("unused-substitution", "x+1; not_used=2", "x"),
    ("nested-substitution", "a*x; a=b+1; b=c*2; c=3", "x"),
]
CORPUS_VALUES = {
    "x": np.array([0.6, 1.2, 2.4]),
    "y": np.array([0.4, 0.9, 1.5]),
    "a": 2,
    "b": 3,
    "c": 1.3,
    "A": 2,
    "left_cutoff": 0.1,
    "origin": 0,
    "r": 1.5,
    "area": 2,
    "centre": 0.2,
    "gamma_": 0.4,
    "sigma": 1.4,
    "_shape": 0.6,
    "scale": 1.2,
    "x0": 0.1,
    "alpha": 0.3,
    "n": 1.2,
    "centre_x": 0.2,
    "centre_y": 0.3,
    "sigma_x": 1.4,
    "sigma_y": 1.7,
    "height": 2,
    "fwhm": 1.5,
    "V0": 0.2,
    "Vmax": 2,
    "tau": 1.3,
    "k": 1.1,
    "a0": 0.3,
    "a1": 0.4,
    "a2": 0.5,
    "power": 2,
    "one": 0.8,
    "two": 0.3,
    "\u03c3": 1.2,
}


@pytest.mark.parametrize("name,text,names", CORPUS, ids=[row[0] for row in CORPUS])
def test_corpus_differential(name, text, names):
    symbols = tuple(s.Symbol(name) for name in names.split())
    reference = _legacy_parse(text)
    assert reference.free_symbols == set(symbols), name
    result = parser.parse_expression(text)
    assert result.free_symbols == set(symbols), name
    values = [CORPUS_VALUES[name] for name in names.split()]
    expected = s.lambdify(symbols, reference, modules=["numpy", "scipy"])(*values)
    actual = s.lambdify(symbols, result, modules=["numpy", "scipy"])(*values)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "text", ["0.12345678901234567890123456789012345678", "1e1000", "1e-1000"]
)
def test_differential_exact_float_tokens(text):
    expected = s.Float(text)
    assert _legacy_parse(text) == expected
    assert parser.parse_expression(text) == expected
    # Scaling by the code-owned literal keeps the numerical comparison finite.
    actual = s.lambdify((), parser.parse_expression(text) / expected, "numpy")()
    np.testing.assert_allclose(actual, 1, rtol=1e-12)


def test_allowlist_classification_complete():
    names = [name for group in CLASSIFICATION.values() for name in group]
    assert len(names) == len(set(names))
    assert set(parser.NS) == set(names)


@pytest.mark.parametrize(
    "function,args,bound",
    [
        (s.gamma, (64,), 64),
        (s.beta, (64, 64), 64),
        (s.zeta, (64,), 64),
        (s.exp, (2048,), 2048),
        (s.sinh, (2048,), 2048),
        (s.cosh, (2048,), 2048),
        (s.tanh, (2048,), 2048),
    ],
    ids=["gamma", "beta", "zeta", "exp", "sinh", "cosh", "tanh"],
)
def test_tripwires_function_delegate_and_reject(function, args, bound):
    expected = function(*args)
    symbolic_args = (x, x) if function is s.beta else (x,)
    symbolic = function(*symbolic_args)
    with tripwires():
        assert function(*args) == expected
        assert function(*symbolic_args) == symbolic
        with pytest.raises(AssertionError, match="TRIPWIRE"):
            function(bound + 1, *args[1:])


@pytest.mark.parametrize(
    "base", [s.Integer(2), s.Rational(3, 2)], ids=["Integer", "Rational"]
)
def test_tripwires_power_delegate_and_reject(base):
    expected = s.Rational(int(base.p) ** 1024, int(base.q) ** 1024)
    with tripwires():
        assert base**1024 == expected
        assert base**x == s.Pow(base, x, evaluate=False)
        with pytest.raises(AssertionError, match="TRIPWIRE"):
            base**1025
        with pytest.raises(AssertionError, match="TRIPWIRE"):
            base**-1025


def test_tripwires_power_bits_and_negative_base():
    assert (1 << 1023).bit_length() == 1024
    assert (1 << 1024).bit_length() == 1025
    with tripwires():
        assert s.Integer(1 << 1023) ** 4 == s.Integer(1 << 4092)
        assert s.Integer(-(1 << 1023)) ** 4 == s.Integer(1 << 4092)
        for base in (
            s.Integer(1 << 1024),
            s.Integer(-(1 << 1024)),
            s.Rational(1, 1 << 1024),
        ):
            with pytest.raises(AssertionError, match="TRIPWIRE"):
                base**4


def _balanced(terms):
    """Keep generated size fixtures below the independent depth limit."""
    if len(terms) == 1:
        return terms[0]
    middle = len(terms) // 2
    return "(" + _balanced(terms[:middle]) + "+" + _balanced(terms[middle:]) + ")"


def _fixture_shape(text, *, nodes, depth=None, exempt=None):
    """Assert derived input truth before testing a particular resource bound."""
    assert len(text.encode("utf-8")) <= 8192
    tree = ast.parse(text, mode="eval")
    count = sum(isinstance(node, ast.expr) for node in ast.walk(tree))
    assert count == nodes
    calls = sum(isinstance(node, ast.Call) for node in ast.walk(tree))
    constants = {name for group in CLASSIFICATION.values() for name in group}
    symbols = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } - constants
    maximum = 0
    stack = [(tree, 0)]
    while stack:
        node, level = stack.pop()
        level += isinstance(node, ast.expr)
        maximum = max(maximum, level)
        stack.extend((child, level) for child in ast.iter_child_nodes(node))
    if depth is not None:
        assert maximum == depth
    assert count <= 2048 or exempt == "R-NODES"
    assert maximum <= 160 or exempt == "R-DEPTH"
    assert calls <= 128 or exempt == "R-CALLS"
    assert len(symbols) <= 128 or exempt == "R-SYMBOLS"
    return text


def _node_boundary(accept):
    """A balanced raw AST with exactly 2048 or 2049 expression nodes."""
    if accept:
        return _fixture_shape("-" + _balanced(["x"] * 1024), nodes=2048, depth=12)
    return _fixture_shape(
        _balanced(["x"] * 1025), nodes=2049, depth=12, exempt="R-NODES"
    )


def _distributed_boundary(count):
    """Coefficient distribution grows 1637/1641 raw nodes to 2046/2051."""
    assert count in (409, 410)
    text = "2*" + _balanced([f"s{i % 100}**{i // 100 + 2}" for i in range(count)])
    expected = s.Add(
        *(2 * s.Symbol(f"s{i % 100}") ** (i // 100 + 2) for i in range(count))
    )
    assert sum(1 for _ in s.preorder_traversal(expected)) == (
        2046 if count == 409 else 2051
    )
    _fixture_shape(text, nodes=1637 if count == 409 else 1641, depth=12)
    return text, expected


def _growth_boundary(accept):
    """The preflight sees 2048/2049 nodes; Add later flattens one away."""
    text, rhs = _distributed_boundary(409)
    if not accept:
        text = "(" + text + ")+1"
        rhs = rhs + 1
    _fixture_shape(text, nodes=1637 if accept else 1639, depth=12 if accept else 13)
    current = x + y
    assert sum(1 for _ in s.preorder_traversal(current)) == 3
    assert current.count(x) == 1
    rhs_nodes = sum(1 for _ in s.preorder_traversal(rhs))
    assert rhs_nodes == (2046 if accept else 2047)
    after = 3 + (rhs_nodes - 1)
    assert after == (2048 if accept else 2049)
    expected = rhs + y
    assert sum(1 for _ in s.preorder_traversal(expected)) == (2047 if accept else 2048)
    return "x+y; x=" + text, expected


def _unicode_symbol_boundary(count):
    """Keep 128/129 distinct written names despite only 26 normalized names."""
    assert count in (128, 129)
    starts = (97, 0xFF41, 0x1D41A, 0x1D5EE, 0x1D68A)
    names = [chr(start + i) for start in starts for i in range(26)][:count]
    assert len(set(names)) == count
    assert all(name.isidentifier() for name in names)
    assert len({unicodedata.normalize("NFKC", name) for name in names}) == 26
    source = "+".join(names)
    assert len(source.encode("utf-8")) == (535 if count == 128 else 540)
    _fixture_shape(source, nodes=2 * count - 1, depth=count)
    expected = s.Add(*(s.Symbol(name) for name in names))
    assert len(expected.free_symbols) == count
    return source, expected


ACCEPT_CASES = [
    ("R-BYTES-accept", "x" + " " * 8191, x),
    ("R-SUBS-accept", "x" + "".join(f";s{i}=1" for i in range(32)), x),
    ("R-NODES-accept-raw", _node_boundary(True), -1024 * x),
    ("R-NODES-accept-post", *_distributed_boundary(409)),
    ("R-DEPTH-accept", _fixture_shape("-" * 159 + "x", nodes=160, depth=160), -x),
    (
        "R-CALLS-accept",
        _fixture_shape("+".join(["sin(x)"] * 128), nodes=511, depth=129),
        128 * s.sin(x),
    ),
    (
        "R-SYMBOLS-accept",
        _fixture_shape("+".join(f"s{i}" for i in range(128)), nodes=255, depth=128),
        s.Add(*(s.Symbol(f"s{i}") for i in range(128))),
    ),
    ("R-IDENT-LEN-accept", "a" * 64, s.Symbol("a" * 64)),
    ("R-SYMBOLS-accept-unicode", *_unicode_symbol_boundary(128)),
    ("R-INT-BITS-accept", str((1 << 256) - 1), s.Integer((1 << 256) - 1)),
    (
        "R-INT-BITS-accept-Rational",
        f"Rational({(1 << 256) - 1}, 1)",
        s.Integer((1 << 256) - 1),
    ),
    ("R-FLOAT-DIGITS-accept", "0." + "1" * 64, s.Float("0." + "1" * 64)),
    ("R-FLOAT-EXP-accept-positive", "1e1000", s.Float("1e1000")),
    ("R-FLOAT-EXP-accept-negative", "1e-1000", s.Float("1e-1000")),
    ("R-CONST-POW-accept-operator", "2**1024", s.Integer(1 << 1024)),
    ("R-CONST-POW-accept-pow", "pow(2, 1024)", s.Integer(1 << 1024)),
    ("R-CONST-POW-accept-Pow", "Pow(2, 1024)", s.Integer(1 << 1024)),
    ("R-CONST-POW-accept-folded", "2**(1000+24)", s.Integer(1 << 1024)),
    ("R-CONST-POW-accept-pow-folded", "pow(2, 32*32)", s.Integer(1 << 1024)),
    ("R-CONST-POW-accept-Pow-folded", "Pow(2, 2**10)", s.Integer(1 << 1024)),
    ("R-CONST-POW-accept-symbolic", "x**1024", x**1024),
    ("R-CONST-POW-accept-negative", "x**-1024", x**-1024),
    ("R-CONST-POW-accept-cancellation", "2**(a-b+512+512); a=b", s.Integer(1 << 1024)),
    ("R-TOKEN-accept-floor", "floor(x / 2)", s.floor(x / 2)),
    ("R-TOKEN-accept-Mod", "Mod(x, 3)", s.Mod(x, 3)),
    (
        "R-TOKEN-accept-floor-power",
        "floor(x / 2**1024)",
        s.floor(x / s.Integer(1 << 1024)),
    ),
    (
        "R-IDENT-KEYWORD-accept-True",
        "Piecewise((x,x>0),(0,True))",
        s.Piecewise((x, x > 0), (0, True)),
    ),
    ("R-IDENT-KEYWORD-accept-single-True", "Piecewise((x,True))", x),
    (
        "R-IDENT-KEYWORD-accept-False",
        "Piecewise((x,x>0),(0,False))",
        s.Piecewise((x, x > 0)),
    ),
    ("R-EVAL-DOMAIN-accept-Min", "Min(1,2)", s.Integer(1)),
    ("R-EVAL-DOMAIN-accept-Max", "Max(x,1)", s.Max(x, 1)),
    (
        "R-EVAL-DOMAIN-accept-Heaviside",
        "Heaviside(x-1,0.5)",
        s.Heaviside(x - 1, s.Float("0.5")),
    ),
    ("R-EVAL-DOMAIN-accept-zeta", "zeta(1+10**-5)", s.zeta(s.Rational(100001, 100000))),
    ("R-CONST-ARG-accept-dead-branch", "Piecewise((x,True),(exp(10**9),True))", x),
    ("R-POW-NEST-accept-operator", "x**x**x**x**x", x ** (x ** (x ** (x**x)))),
    (
        "R-POW-NEST-accept-pow",
        "pow(x,pow(x,pow(x,pow(x,x))))",
        x ** (x ** (x ** (x**x))),
    ),
    ("R-POW-NEST-accept-Pow", "Pow(x,x**x**x**x)", x ** (x ** (x ** (x**x)))),
    (
        "R-RATIONAL-BITS-accept-product",
        "2**1024*2**1024*2**1024*2**1023",
        s.Integer(1 << 4095),
    ),
    ("R-RATIONAL-BITS-accept-power", "(2**1023)**4", s.Integer(1 << 4092)),
    (
        "R-RATIONAL-BITS-accept-gamma",
        "gamma(64)**14",
        s.Integer(math.factorial(63) ** 14),
    ),
    ("R-RATIONAL-BITS-accept-negative", "(-2**1023)**4", s.Integer(1 << 4092)),
    ("R-CONST-ARG-accept-gamma", "gamma(64)", s.Integer(math.factorial(63))),
    (
        "R-CONST-ARG-accept-gamma-substitution",
        "x+gamma(a); a=64",
        x + math.factorial(63),
    ),
    (
        "R-CONST-ARG-accept-gamma-rational",
        "gamma(Rational(127,2))",
        s.gamma(s.Rational(127, 2)),
    ),
    ("R-CONST-ARG-accept-beta", "beta(64,64)", s.beta(64, 64)),
    ("R-CONST-ARG-accept-zeta", "zeta(64)", s.zeta(64)),
    ("R-CONST-ARG-accept-exp", "exp(2048)", s.exp(2048)),
    ("R-CONST-ARG-accept-sinh", "sinh(a); a=2048", s.sinh(2048)),
    ("R-CONST-ARG-accept-cosh", "cosh(-2048)", s.cosh(2048)),
    ("R-CONST-ARG-accept-tanh", "tanh(2048)", s.tanh(2048)),
    ("R-CONST-ARG-accept-exp-sum", "exp(1000)+exp(1000)", 2 * s.exp(1000)),
    (
        "R-CONST-ARG-accept-exp-cancellation",
        "floor(exp(2048)*exp(-2048)+1/3)",
        s.Integer(1),
    ),
    ("R-CONST-ARG-accept-sin-imaginary", "sin(2048*I)", s.I * s.sinh(2048)),
    ("R-CONST-ARG-accept-cos-imaginary", "cos(2048*I)", s.cosh(2048)),
    ("R-CONST-ARG-accept-tan-imaginary", "tan(2048*I)", s.I * s.tanh(2048)),
    ("R-CONST-ARG-accept-erf-imaginary", "erf(50*I)", s.erf(50 * s.I)),
    ("R-CONST-ARG-accept-erfc-imaginary", "erfc(50*I)", s.erfc(50 * s.I)),
    ("R-GROWTH-accept", *_growth_boundary(True)),
    ("R-TYPE-accept-numpy-string", np.str_("x"), x),
    ("R-RATIONAL-BITS-accept-ten-power", "10**400", s.Integer(10**400)),
    ("R-NONFINITE-accept", "1/2", s.Rational(1, 2)),
    ("R-IDENT-UNICODE-accept", "\u03c3*x", s.Symbol("\u03c3") * x),
    ("R-IDENT-UNICODE-accept-function-alias", "\uff53\uff49\uff4e(x)", s.sin(x)),
    ("R-IDENT-DUNDER-accept", "_shape+gamma_", s.Symbol("_shape") + s.Symbol("gamma_")),
    ("R-SUBS-LHS-accept", "a+b; a=x; b=2", x + 2),
    ("R-SEMI-accept", "x; a=2", x),
    ("R-EMPTY-accept", "0", s.Integer(0)),
    ("R-CTRL-accept", "\t\n\r\f\v x ", x),
    ("R-FUNC-VALUE-accept", "gamma_", s.Symbol("gamma_")),
    ("R-CHAINED-CMP-accept", "x<1", s.Lt(x, 1)),
    ("R-ARITY-accept-one", "Min(x)", x),
    (
        "R-ARITY-accept-sixteen",
        "Min(" + ",".join(f"s{i}" for i in range(16)) + ")",
        s.Min(*(s.Symbol(f"s{i}") for i in range(16))),
    ),
    (
        "R-PIECEWISE-accept",
        "Piecewise((1,x>0),(0,True))",
        s.Piecewise((1, x > 0), (0, True)),
    ),
    ("R-KEYWORD-accept", "sin(x)", s.sin(x)),
    ("R-CALL-TARGET-accept", "cos(x)", s.cos(x)),
    ("R-AST-NODE-accept", "~(x>0)", s.Le(x, 0)),
    ("R-PARSE-accept", "(x+1)", x + 1),
]


# The final flag marks P0-P6 errors. Those rows replace both raw entry points
# with failing mocks, proving no SymPy parser or fallback was reached.
REJECT_CASES = [
    ("R-BYTES-reject", "x" + " " * 8192, "R-BYTES", True),
    ("R-SUBS-reject", "x" + "".join(f";s{i}=1" for i in range(33)), "R-SUBS", True),
    ("R-NODES-reject-raw", _node_boundary(False), "R-NODES", True),
    ("R-NODES-reject-post", _distributed_boundary(410)[0], "R-NODES", False),
    (
        "R-DEPTH-reject",
        _fixture_shape("-" * 160 + "x", nodes=161, depth=161, exempt="R-DEPTH"),
        "R-DEPTH",
        True,
    ),
    (
        "R-CALLS-reject",
        _fixture_shape(
            "+".join(["sin(x)"] * 129), nodes=515, depth=130, exempt="R-CALLS"
        ),
        "R-CALLS",
        True,
    ),
    (
        "R-SYMBOLS-reject",
        _fixture_shape(
            "+".join(f"s{i}" for i in range(129)),
            nodes=257,
            depth=129,
            exempt="R-SYMBOLS",
        ),
        "R-SYMBOLS",
        True,
    ),
    ("R-IDENT-LEN-reject", "a" * 65, "R-IDENT-LEN", True),
    (
        "R-SYMBOLS-reject-unicode",
        _unicode_symbol_boundary(129)[0],
        "R-SYMBOLS",
        True,
    ),
    ("R-INT-BITS-reject", str(1 << 256), "R-INT-BITS", True),
    ("R-INT-BITS-reject-Rational", f"Rational(1,{1 << 256})", "R-INT-BITS", True),
    ("R-FLOAT-DIGITS-reject", "0." + "1" * 65, "R-FLOAT-DIGITS", True),
    ("R-FLOAT-EXP-reject-positive", "1e1001", "R-FLOAT-EXP", True),
    ("R-FLOAT-EXP-reject-negative", "1e-1001", "R-FLOAT-EXP", True),
    ("R-CONST-POW-reject-operator", "2**1025", "R-CONST-POW", False),
    ("R-CONST-POW-reject-pow", "pow(2,1025)", "R-CONST-POW", False),
    ("R-CONST-POW-reject-Pow", "Pow(2,1025)", "R-CONST-POW", False),
    ("R-CONST-POW-reject-folded", "2**(1000+25)", "R-CONST-POW", False),
    ("R-CONST-POW-reject-pow-folded", "pow(2,5*205)", "R-CONST-POW", False),
    ("R-CONST-POW-reject-Pow-folded", "Pow(2,2**11)", "R-CONST-POW", False),
    ("R-CONST-POW-reject-inner-power", "2**(2**20)", "R-CONST-POW", False),
    ("R-CONST-POW-reject-large-Pow", "Pow(2,1000000000)", "R-CONST-POW", False),
    ("R-CONST-POW-reject-tower", "3**3**3**3", "R-CONST-POW", False),
    ("R-CONST-POW-reject-symbolic", "x**1025", "R-CONST-POW", False),
    ("R-CONST-POW-reject-negative", "x**-1025", "R-CONST-POW", False),
    ("R-CONST-POW-reject-substitution", "x**a; a=2000", "R-CONST-POW", False),
    ("R-CONST-POW-reject-substitution-tower", "x**x**x**x; x=3", "R-CONST-POW", False),
    ("R-CONST-POW-reject-cancellation", "2**(a-b+512+513); a=b", "R-CONST-POW", False),
    ("R-POW-NEST-reject-operator", "x**x**x**x**x**x", "R-POW-NEST", True),
    (
        "R-POW-NEST-reject-pow",
        "pow(x,pow(x,pow(x,pow(x,pow(x,x)))))",
        "R-POW-NEST",
        True,
    ),
    ("R-POW-NEST-reject-Pow", "Pow(x,x**x**x**x**x)", "R-POW-NEST", True),
    (
        "R-RATIONAL-BITS-reject-product",
        "2**1024*2**1024*2**1024*2**1024",
        "R-RATIONAL-BITS",
        False,
    ),
    ("R-RATIONAL-BITS-reject-power", "(2**1024)**4", "R-RATIONAL-BITS", False),
    ("R-RATIONAL-BITS-reject-precall-exact", "(2**16)**241", "R-RATIONAL-BITS", False),
    ("R-RATIONAL-BITS-reject-gamma", "gamma(64)**15", "R-RATIONAL-BITS", False),
    ("R-RATIONAL-BITS-reject-negative", "(-2**1024)**4", "R-RATIONAL-BITS", False),
    (
        "R-RATIONAL-BITS-reject-negative-unique",
        "(-2**1024)**1024",
        "R-RATIONAL-BITS",
        False,
    ),
    (
        "R-RATIONAL-BITS-reject-nested-gamma",
        "((gamma(64)**1024)**1024)**1024",
        "R-RATIONAL-BITS",
        False,
    ),
    ("R-RATIONAL-BITS-reject-floats", "1e1000*1e1000*1e1000", "R-RATIONAL-BITS", False),
    (
        "R-RATIONAL-BITS-reject-cancellation-unique",
        "(2**1024*2**1024*2**1024*2**1024)-(2**1024*2**1024*2**1024*2**1024)",
        "R-RATIONAL-BITS",
        False,
    ),
    ("R-CONST-ARG-reject-gamma", "gamma(65)", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-gamma-substitution", "x+gamma(a); a=65", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-beta", "beta(a,b); a=65; b=1", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-beta-second", "beta(1,65)", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-zeta", "zeta(a); a=65", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-zeta-second", "zeta(2,65)", "R-CONST-ARG", False),
    (
        "R-CONST-ARG-reject-gamma-large",
        "x+gamma(a); a=1000000000",
        "R-CONST-ARG",
        False,
    ),
    (
        "R-CONST-ARG-reject-gamma-folded",
        "gamma(a+b); a=1000000000; b=1",
        "R-CONST-ARG",
        False,
    ),
    (
        "R-CONST-ARG-reject-gamma-multiply",
        "gamma(a); a=b*c; b=1000000000; c=1",
        "R-CONST-ARG",
        False,
    ),
    ("R-CONST-ARG-reject-gamma-nonreal", "gamma(I)", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-exp-limit", "exp(2049)", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-exp-floor", "floor(exp(10**9))", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-exp-cos", "cos(exp(10**9))", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-exp-float", "exp(exp(1.0e9))", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-exp-nested", "exp(exp(8))", "R-CONST-ARG", False),
    (
        "R-CONST-ARG-reject-sinh-folded",
        "sinh(a*b); a=1000; b=1000",
        "R-CONST-ARG",
        False,
    ),
    ("R-CONST-ARG-reject-sin-floor", "floor(sin(I*10**9))", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-cos-floor", "floor(cos(1j*10**9))", "R-CONST-ARG", False),
    ("R-CONST-ARG-reject-erf-floor", "floor(erf(I*10**9))", "R-CONST-ARG", False),
    (
        "R-CONST-ARG-reject-erf-folded",
        "erf(a*b); a=10**5; b=I*10**4",
        "R-CONST-ARG",
        False,
    ),
    ("R-GROWTH-reject", _growth_boundary(False)[0], "R-GROWTH", False),
]
for _name in ("exp", "sinh", "cosh", "tanh"):
    REJECT_CASES.extend(
        [
            (
                f"R-CONST-ARG-reject-{_name}-direct-boundary",
                f"{_name}(2049)",
                "R-CONST-ARG",
                False,
            ),
            (
                f"R-CONST-ARG-reject-{_name}-large",
                f"{_name}(10**9)",
                "R-CONST-ARG",
                False,
            ),
            (
                f"R-CONST-ARG-reject-{_name}-substitution",
                f"{_name}(a); a=10**9",
                "R-CONST-ARG",
                False,
            ),
        ]
    )
for _name, _limit in (
    ("sin", 2048),
    ("cos", 2048),
    ("tan", 2048),
    ("erf", 50),
    ("erfc", 50),
):
    REJECT_CASES.extend(
        [
            (
                f"R-CONST-ARG-reject-{_name}-imaginary-limit",
                f"{_name}({_limit + 1}*I)",
                "R-CONST-ARG",
                False,
            ),
            (
                f"R-CONST-ARG-reject-{_name}-imaginary-large",
                f"{_name}(I*10**9)",
                "R-CONST-ARG",
                False,
            ),
            (
                f"R-CONST-ARG-reject-{_name}-imaginary-substitution",
                f"{_name}(a); a=I*10**9",
                "R-CONST-ARG",
                False,
            ),
        ]
    )
for _name, _text in [
    ("Min", "Min(I,1)"),
    ("Max", "Max(1,I*10**9)"),
    ("Min-symbol", "Min(x,I)"),
    ("Heaviside", "Heaviside(I)"),
    ("Heaviside-two", "Heaviside(2*I)"),
    ("Piecewise", "Piecewise((1,I>0),(0,True))"),
    ("comparison", "x<I"),
    ("Gt", "Gt(I,1)"),
    ("Lt", "Lt(I,1)"),
    ("Mod", "Mod(1,0)"),
    ("zeta-60", "zeta(1+10**-60)"),
    ("zeta-300", "zeta(1+10**-300)"),
    ("floor-precision", "floor(exp(2048))"),
    ("ceiling-precision", "ceiling(exp(2048))"),
    ("floor-real-precision", "floor(re(exp(2048)))"),
]:
    REJECT_CASES.append(
        (f"R-EVAL-DOMAIN-reject-{_name}", _text, "R-EVAL-DOMAIN", False)
    )


EXACT_MAGNITUDE_ACCEPTS = [
    ("R-CONST-POW-exact-accept-direct", "x**(1024-sqrt(2)/10**20)"),
    ("R-CONST-POW-exact-accept-substitution", "x**a; a=1024-sqrt(2)/10**20"),
    (
        "R-RATIONAL-BITS-exact-accept-precall",
        "(16-sqrt(2)/10**20)**1024",
    ),
    ("R-CONST-ARG-exact-accept-exp", "exp(2048-sqrt(2)/10**20)"),
    ("R-CONST-ARG-exact-accept-exp-substitution", "exp(a); a=2048-sqrt(2)/10**20"),
    ("R-CONST-ARG-exact-accept-sin", "sin((2048-sqrt(2)/10**20)*I)"),
    (
        "R-CONST-ARG-exact-accept-sin-substitution",
        "sin(a); a=(2048-sqrt(2)/10**20)*I",
    ),
    ("R-CONST-ARG-exact-accept-erf", "erf((50-sqrt(2)/10**20)*I)"),
    (
        "R-CONST-ARG-exact-accept-erf-substitution",
        "erf(a); a=(50-sqrt(2)/10**20)*I",
    ),
    ("R-CONST-ARG-exact-accept-gamma", "gamma(64-sqrt(2)/10**20)"),
    (
        "R-CONST-ARG-exact-accept-gamma-substitution",
        "gamma(a); a=64-sqrt(2)/10**20",
    ),
    (
        "R-RATIONAL-BITS-exact-accept-postcall",
        "2.0**1024*2**1024*2**1024*2**1024*(1-sqrt(2)/10**20)",
    ),
]

EXACT_MAGNITUDE_REJECTS = [
    ("R-CONST-POW-exact-reject-direct", "x**(1024+sqrt(2)/10**20)", "R-CONST-POW"),
    (
        "R-CONST-POW-exact-reject-substitution",
        "x**a; a=1024+sqrt(2)/10**20",
        "R-CONST-POW",
    ),
    (
        "R-RATIONAL-BITS-exact-reject-precall",
        "(16+sqrt(2)/10**20)**1024",
        "R-RATIONAL-BITS",
    ),
    ("R-CONST-ARG-exact-reject-exp", "exp(2048+sqrt(2)/10**20)", "R-CONST-ARG"),
    (
        "R-CONST-ARG-exact-reject-exp-substitution",
        "exp(a); a=2048+sqrt(2)/10**20",
        "R-CONST-ARG",
    ),
    (
        "R-CONST-ARG-exact-reject-sin",
        "sin((2048+sqrt(2)/10**20)*I)",
        "R-CONST-ARG",
    ),
    (
        "R-CONST-ARG-exact-reject-sin-substitution",
        "sin(a); a=(2048+sqrt(2)/10**20)*I",
        "R-CONST-ARG",
    ),
    ("R-CONST-ARG-exact-reject-erf", "erf((50+sqrt(2)/10**20)*I)", "R-CONST-ARG"),
    (
        "R-CONST-ARG-exact-reject-erf-substitution",
        "erf(a); a=(50+sqrt(2)/10**20)*I",
        "R-CONST-ARG",
    ),
    ("R-CONST-ARG-exact-reject-gamma", "gamma(64+sqrt(2)/10**20)", "R-CONST-ARG"),
    (
        "R-CONST-ARG-exact-reject-gamma-substitution",
        "gamma(a); a=64+sqrt(2)/10**20",
        "R-CONST-ARG",
    ),
    (
        "R-RATIONAL-BITS-exact-reject-postcall",
        "2.0**1024*2**1024*2**1024*2**1024*(1+sqrt(2)/10**20)",
        "R-RATIONAL-BITS",
    ),
]

BOOLEAN_OPERAND_REJECTS = [
    ("R-EVAL-DOMAIN-reject-boolean-Pow-True", "x**True"),
    ("R-EVAL-DOMAIN-reject-boolean-pow-False", "pow(x,False)"),
    ("R-EVAL-DOMAIN-reject-boolean-exp-True", "exp(True)"),
    ("R-EVAL-DOMAIN-reject-boolean-exp-False", "exp(False)"),
    ("R-EVAL-DOMAIN-reject-boolean-sin-True", "sin(True)"),
    ("R-EVAL-DOMAIN-reject-boolean-sin-False", "sin(False)"),
    ("R-EVAL-DOMAIN-reject-boolean-erf-True", "erf(True)"),
    ("R-EVAL-DOMAIN-reject-boolean-erf-False", "erf(False)"),
    ("R-EVAL-DOMAIN-reject-boolean-gamma-True", "gamma(True)"),
    ("R-EVAL-DOMAIN-reject-boolean-gamma-False", "gamma(False)"),
    ("R-EVAL-DOMAIN-reject-boolean-asin-True", "asin(True)"),
    ("R-EVAL-DOMAIN-reject-boolean-asin-False", "asin(False)"),
    ("R-EVAL-DOMAIN-reject-boolean-log", "log(True)"),
    ("R-EVAL-DOMAIN-reject-boolean-Min", "Min(True,1)"),
    ("R-EVAL-DOMAIN-reject-boolean-Add", "x+True"),
    ("R-EVAL-DOMAIN-reject-relational-sin", "sin(x>0)"),
    ("R-EVAL-DOMAIN-reject-concrete-relational-exp", "exp(1>0)"),
    ("R-EVAL-DOMAIN-reject-And-sqrt", "sqrt(And(x>0,x<1))"),
    ("R-EVAL-DOMAIN-reject-Or-log", "log(Or(x>0,x<1))"),
    ("R-EVAL-DOMAIN-reject-Not-cos", "cos(Not(x>0))"),
    (
        "R-EVAL-DOMAIN-reject-boolean-Piecewise-value",
        "sin(Piecewise((True,x>0),(False,True)))",
    ),
    ("R-EVAL-DOMAIN-reject-boolean-where-branch", "where(x>0,True,0)"),
    (
        "R-EVAL-DOMAIN-reject-substitution-Piecewise-condition",
        "Piecewise((1,q>0),(0,True));q=Piecewise((1,x>0),(0,True))",
    ),
]


def _assert_rejected(text, rule, preparse):
    """Check the specified stage, never merely that some ValueError occurred."""
    with tripwires(), ExitStack() as stack:
        if preparse:
            stack.enter_context(
                patch.object(
                    sympy_parser,
                    "parse_expr",
                    side_effect=AssertionError("parse_expr reached before validation"),
                )
            )
            stack.enter_context(
                patch.object(
                    s,
                    "sympify",
                    side_effect=AssertionError("raw sympify fallback reached"),
                )
            )
        with pytest.raises(ValueError) as caught:
            parser.parse_expression(text)
        if isinstance(caught.value.__cause__, AssertionError):
            raise caught.value.__cause__
        assert isinstance(caught.value, parser.UnsafeExpressionError)
        assert caught.value.rule == rule
        assert str(caught.value).startswith(f"[{rule}] ")


@pytest.mark.parametrize(
    "case,text,expected", ACCEPT_CASES, ids=[row[0] for row in ACCEPT_CASES]
)
def test_boundary_accept(case, text, expected):
    with tripwires():
        assert parser.parse_expression(text) == expected


@pytest.mark.parametrize(
    "case,text,rule,preparse", REJECT_CASES, ids=[row[0] for row in REJECT_CASES]
)
def test_boundary_reject(case, text, rule, preparse):
    _assert_rejected(text, rule, preparse)


@pytest.mark.parametrize(
    "case,text",
    EXACT_MAGNITUDE_ACCEPTS,
    ids=[row[0] for row in EXACT_MAGNITUDE_ACCEPTS],
)
def test_exact_magnitude_accept(case, text):
    with tripwires():
        parser.parse_expression(text)


@pytest.mark.parametrize(
    "case,text,rule",
    EXACT_MAGNITUDE_REJECTS,
    ids=[row[0] for row in EXACT_MAGNITUDE_REJECTS],
)
def test_exact_magnitude_reject(case, text, rule):
    _assert_rejected(text, rule, False)


def test_exact_magnitude_fixture_truth():
    delta = s.sqrt(2) / 10**20
    assert delta.is_positive is True
    for limit in (50, 64, 1024, 2048):
        assert (s.Integer(limit) + delta - limit).is_positive is True
        assert (s.Integer(limit) - delta - limit).is_negative is True
    above_scale = s.Abs(s.log(16 + delta, 2)) * 1024
    below_scale = s.Abs(s.log(16 - delta, 2)) * 1024
    assert (above_scale - 4096).is_positive is True
    assert (below_scale - 4096).is_negative is True
    result_limit = s.Integer(1 << 4096)
    above_result = s.Float(2) ** 4096 * (1 + delta)
    below_result = s.Float(2) ** 4096 * (1 - delta)
    assert (above_result - result_limit).is_positive is True
    assert (below_result - result_limit).is_negative is True


@pytest.mark.parametrize(
    "case,text",
    BOOLEAN_OPERAND_REJECTS,
    ids=[row[0] for row in BOOLEAN_OPERAND_REJECTS],
)
def test_boolean_numeric_operand_reject(case, text):
    _assert_rejected(text, "R-EVAL-DOMAIN", False)


@pytest.mark.parametrize(
    "text",
    [
        "erfc(1e300)",
        "erfc(10**300)",
        "erfc(q); q=10**300",
        "x*erfc(10**309)",
        "erfc(-10**300)",
    ],
    ids=["float", "integer", "substitution", "nested", "negative"],
)
def test_erfc_overflow_reject(text):
    s.core.cache.clear_cache()
    try:
        with pytest.raises(parser.UnsafeExpressionError) as caught:
            parser.parse_expression(text)
        assert isinstance(caught.value, ValueError)
        assert caught.value.rule == "R-EVAL-DOMAIN"
    finally:
        s.core.cache.clear_cache()


@pytest.mark.parametrize(
    "stage", ["magnitude", "comparison", "evaluation", "substitution"]
)
def test_evaluation_overflow_conversion(stage):
    # Isolate each conversion so outer handlers cannot hide a missing one.
    with pytest.raises(parser.UnsafeExpressionError) as caught, ExitStack() as stack:
        if stage == "magnitude":
            # Injecting the failure keeps this stage independent of the numerical
            # backend: with gmpy2 the real overflow that used to raise here does
            # not occur for erfc of a huge constant.
            stack.enter_context(
                patch.object(parser.sympy, "Abs", side_effect=OverflowError("probe"))
            )
            parser._magnitude(s.erfc(s.Symbol("x"), evaluate=False))
        elif stage == "comparison":
            stack.enter_context(
                patch.object(parser, "_magnitude", side_effect=OverflowError("probe"))
            )
            parser._exceeds_magnitude(s.Integer(1), 2)
        elif stage == "evaluation":
            stack.enter_context(
                patch.object(
                    parser, "_check_result", side_effect=OverflowError("probe")
                )
            )
            parser._checked_eval(s.Integer(1))
        else:
            stack.enter_context(
                patch.object(s.Symbol, "xreplace", side_effect=OverflowError("probe"))
            )
            parser.parse_expression("x; x=1")
    assert isinstance(caught.value, ValueError)
    assert caught.value.rule == "R-EVAL-DOMAIN"


@pytest.mark.parametrize(
    "text,expected_rule",
    [
        ("Mod(Abs(I),acos(pi))", "R-EVAL-DOMAIN"),
        ("And(q,x>0); q=1+I", "R-EVAL-DOMAIN"),
        ("zeta(1e-300,0.0)", "R-EVAL-DOMAIN"),
        ("Mod(atanh(10**9),2)", "R-EVAL-DOMAIN"),
        ("1e" + "9" * 4301, "R-FLOAT-EXP"),
        ("1.5e" + "0" * 5000, "R-FLOAT-EXP"),
    ],
    ids=[
        "RecursionError-evaluation",
        "RecursionError-substitution",
        "NoConvergence",
        "NotImplementedError",
        "float-exponent-value-digits",
        "float-exponent-leading-zero-digits",
    ],
)
def test_evaluation_exception_leak_regression(text, expected_rule):
    s.core.cache.clear_cache()
    try:
        with pytest.raises(parser.UnsafeExpressionError) as caught:
            parser.parse_expression(text)
        assert isinstance(caught.value, ValueError)
        assert caught.value.rule == expected_rule
    finally:
        s.core.cache.clear_cache()


@pytest.mark.parametrize(
    "stage,expected_rule",
    [
        ("magnitude", "R-EVAL-DOMAIN"),
        ("comparison", "R-EVAL-DOMAIN"),
        ("result", "R-EVAL-DOMAIN"),
        ("evaluation", "R-EVAL-DOMAIN"),
        ("parse", "R-PARSE"),
        ("post-validation", "R-EVAL-DOMAIN"),
        ("substitution", "R-EVAL-DOMAIN"),
    ],
    ids=[
        "magnitude",
        "comparison",
        "result",
        "evaluation",
        "parse",
        "post-validation",
        "substitution",
    ],
)
def test_evaluation_exception_conversion(stage, expected_rule):
    # Isolate every conversion so an outer handler cannot mask its absence.
    with pytest.raises(parser.UnsafeExpressionError) as caught, ExitStack() as stack:
        if stage == "magnitude":

            class Value:
                @property
                def is_nonnegative(self):
                    raise RuntimeError("probe")

            parser._magnitude(Value())
        elif stage == "comparison":
            stack.enter_context(
                patch.object(parser, "_magnitude", side_effect=RuntimeError("probe"))
            )
            parser._exceeds_magnitude(s.Integer(1), 2)
        elif stage == "result":

            class Value:
                def has(self, *_):
                    raise RuntimeError("probe")

            parser._check_result(Value())
        elif stage == "evaluation":
            stack.enter_context(
                patch.object(parser, "_check_result", side_effect=RuntimeError("probe"))
            )
            parser._checked_eval(s.Integer(1))
        elif stage == "parse":
            stack.enter_context(
                patch.object(
                    sympy_parser, "parse_expr", side_effect=RuntimeError("probe")
                )
            )
            parser._evaluate_curated("x")
        elif stage == "post-validation":
            stack.enter_context(
                patch.object(s, "preorder_traversal", side_effect=RuntimeError("probe"))
            )
            parser._post_validate(s.Integer(1))
        else:
            stack.enter_context(
                patch.object(s.Symbol, "xreplace", side_effect=RuntimeError("probe"))
            )
            parser.parse_expression("x; x=1")
    assert isinstance(caught.value, ValueError)
    assert caught.value.rule == expected_rule


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "sin(Piecewise((1,x>0),(0,True)))",
            s.sin(s.Piecewise((1, x > 0), (0, True))),
        ),
        ("where(True,x,0)", s.Function("where")(s.true, x, 0)),
        ("Eq(True,False)", s.false),
        ("And(x>0,x<1)", s.And(x > 0, x < 1)),
    ],
    ids=["Piecewise-value", "where-condition", "Equality", "And"],
)
def test_boolean_predicate_positions_accept(text, expected):
    assert parser.parse_expression(text) == expected


def test_boundary_numeric_fixture_truth():
    assert len(("x" + " " * 8191).encode("utf-8")) == 8192
    assert len(("x" + " " * 8192).encode("utf-8")) == 8193
    assert math.factorial(63).bit_length() == 290
    assert 290 * 14 == 4060
    assert 290 * 15 == 4350
    assert (10**400).bit_length() == 1329
    assert (1 << 4095).bit_length() == 4096
    assert (1 << 4096).bit_length() == 4097
    assert (1 << 16).bit_length() * 241 == 4097
    for value, expected in ((s.sin(2048 * s.I), 2954), (s.erf(50 * s.I), 3601)):
        _, _, exponent, bits = s.Abs(value.evalf(15)).evalf(15)._mpf_
        assert exponent + bits == expected


# CPython 3.12 replaced the pure-Python scanner in `tokenize` with the C
# tokenizer, so full-width underscore identifiers tokenize as NAME and an
# unmatched ")" as OP only from there; on Python <= 3.11 `tokenize` rejects
# those inputs first. Both paths reject the input, only the rule differs.
_C_TOKENIZER = sys.version_info >= (3, 12)


MALICIOUS_CASES = [
    ("import-string", "__import__('os')", "R-TOKEN", True),
    ("import-name", "__import__(x)", "R-IDENT-DUNDER", True),
    ("subclasses", "().__class__.__base__.__subclasses__()", "R-TOKEN", True),
    ("class-attribute", "x.__class__", "R-TOKEN", True),
    ("attribute", "x.y", "R-TOKEN", True),
    ("attribute-call", "sin.__call__(x)", "R-TOKEN", True),
    ("lambda-call", "(lambda: 1)()", "R-TOKEN", True),
    ("lambda-keyword", "lambda", "R-IDENT-KEYWORD", True),
    ("list-comprehension", "[i for i in range(3)]", "R-TOKEN", True),
    ("dict-comprehension", "{i: i for i in x}", "R-TOKEN", True),
    ("generator", "(i for i in x)", "R-IDENT-KEYWORD", True),
    ("open-string", "open('f')", "R-TOKEN", True),
    ("open-name", "open(x)", "R-CALL-TARGET", True),
    ("formatted-string", 'f"{x}"', "R-TOKEN", True),
    ("formatted-string-raw", 'rf"{x}"', "R-TOKEN", True),
    ("string-double", '"a"', "R-TOKEN", True),
    ("string-single", "'a'", "R-TOKEN", True),
    ("subscript", "x[0]", "R-TOKEN", True),
    ("slice", "x[1:3]", "R-TOKEN", True),
    ("kwargs-unknown", "f(**k)", "R-CALL-TARGET", True),
    ("star-unknown", "f(*a)", "R-CALL-TARGET", True),
    ("keyword-call", "sin(x=1)", "R-KEYWORD", True),
    ("kwargs-call", "sin(**k)", "R-KEYWORD", True),
    ("star-call", "sin(*a)", "R-KEYWORD", True),
    (
        "fullwidth-dunder",
        "\uff3f\uff3fimport\uff3f\uff3f",
        "R-IDENT-UNICODE" if _C_TOKENIZER else "R-TOKEN",
        True,
    ),
    (
        "nfkc-dunder",
        "_\uff3fx\uff3f\uff3f",
        "R-IDENT-DUNDER" if _C_TOKENIZER else "R-TOKEN",
        True,
    ),
    ("nfkc-lambda", "\uff4c\uff41\uff4d\uff42\uff44\uff41", "R-IDENT-KEYWORD", True),
    ("nfkc-import", "\uff49\uff4d\uff50\uff4f\uff52\uff54", "R-IDENT-KEYWORD", True),
    ("builtins", "__builtins__", "R-IDENT-DUNDER", True),
    ("dunder-suffix", "x__", "R-IDENT-DUNDER", True),
    ("comment", "x # 1", "R-TOKEN", True),
    ("comment-multiline", "x\n#x", "R-TOKEN", True),
    ("substitution-no-equal", "x; y", "R-SUBS-LHS", True),
    ("substitution-number-lhs", "x; 2=3", "R-SUBS-LHS", True),
    ("substitution-extra-equal", "x; y=1=2", "R-SUBS-LHS", True),
    ("assignment-base", "x=1", "R-SUBS-LHS", True),
    ("double-semicolon", "x;;", "R-SEMI", True),
    ("trailing-semicolon", "x;", "R-SEMI", True),
    ("chained-equal", "x==1==2", "R-CHAINED-CMP", True),
    ("chained-less", "0<x<1", "R-CHAINED-CMP", True),
    ("nan", "nan", "R-NONFINITE", True),
    ("oo", "oo", "R-NONFINITE", True),
    ("zoo", "zoo", "R-NONFINITE", True),
    ("negative-oo", "-oo", "R-NONFINITE", True),
    ("inf", "inf", "R-NONFINITE", True),
    ("power-tower", "x**x**x**x**x**x", "R-POW-NEST", True),
    ("power-large", "2**10000", "R-CONST-POW", False),
    ("ten-power", "10**1025", "R-CONST-POW", False),
    ("exp-large", "exp(10**9)", "R-CONST-ARG", False),
    ("exp-floor", "floor(exp(10**9))", "R-CONST-ARG", False),
    ("exp-cos", "cos(exp(10**9))", "R-CONST-ARG", False),
    ("exp-exp", "exp(exp(1.0e9))", "R-CONST-ARG", False),
    ("gamma-power", "gamma(64)**15", "R-RATIONAL-BITS", False),
    ("float-positive", "1e10000", "R-FLOAT-EXP", True),
    ("float-negative", "1e-10000", "R-FLOAT-EXP", True),
    ("bytes", b"x", "R-TYPE", True),
    ("bytes-two", b"a+x", "R-TYPE", True),
    ("none-object", None, "R-TYPE", True),
    ("number-object", 3, "R-TYPE", True),
    ("nul", "x\x00", "R-CTRL", True),
    ("del", "x\x7f", "R-CTRL", True),
    ("control", "x\x01", "R-CTRL", True),
    ("empty", "", "R-EMPTY", True),
    ("whitespace", " \t\n", "R-EMPTY", True),
    ("floor-division", "x // 2", "R-TOKEN", True),
    ("modulo", "x % 3", "R-TOKEN", True),
    ("floor-division-power", "1 // 2**1025", "R-TOKEN", True),
    ("modulo-power", "1 % 2**1025", "R-TOKEN", True),
    ("floor-division-large", "x // 2**1000000000", "R-TOKEN", True),
    ("None-keyword", "None", "R-IDENT-KEYWORD", True),
    ("None-add", "x+None", "R-IDENT-KEYWORD", True),
    ("fullwidth-True", "\uff34\uff52\uff55\uff45", "R-IDENT-KEYWORD", True),
    ("conditional", "a if b else c", "R-IDENT-KEYWORD", True),
    ("boolean-and", "a and b", "R-IDENT-KEYWORD", True),
    ("boolean-or", "a or b", "R-IDENT-KEYWORD", True),
    ("function-value-gamma", "gamma", "R-FUNC-VALUE", True),
    ("function-value-exp", "exp", "R-FUNC-VALUE", True),
    ("helper-value", "Integer", "R-FUNC-VALUE", True),
    ("tuple", "(x,y)", "R-AST-NODE", True),
    ("tuple-call", "sin((x,y))", "R-AST-NODE", True),
    ("Piecewise-shape", "Piecewise(x)", "R-PIECEWISE", True),
    ("Piecewise-triple", "Piecewise((x,True,1))", "R-PIECEWISE", True),
    ("arity-zero", "sin()", "R-ARITY", True),
    ("arity-three", "pow(x,2,3)", "R-ARITY", True),
    (
        "arity-seventeen",
        "Min(" + ",".join(f"s{i}" for i in range(17)) + ")",
        "R-ARITY",
        True,
    ),
    ("Rational-noninteger", "Rational(x,2)", "R-ARITY", True),
    ("Rational-zero", "Rational(1,0)", "R-ARITY", True),
    ("call-expression", "(x+1)(2)", "R-CALL-TARGET", True),
    ("parse-incomplete", "x+", "R-PARSE", True),
    ("parse-operand", "(x+)", "R-PARSE", True),
    ("token-unclosed", "(x", "R-TOKEN", True),
    (
        "token-extra-close",
        "x)",
        "R-PARSE" if _C_TOKENIZER else "R-TOKEN",
        True,
    ),
    ("And-domain", "And(I,1)", "R-PARSE", False),
    ("Or-domain", "Or(I,1)", "R-PARSE", False),
    ("division-zero", "1/0", "R-NONFINITE", False),
    ("division-substitution", "a/b; a=1; b=0", "R-NONFINITE", False),
]
for _name in (
    "exec",
    "eval",
    "Symbol",
    "S",
    "N",
    "integrate",
    "Matrix",
    "Function",
    "Integer",
    "Float",
):
    MALICIOUS_CASES.append((_name + "-call", _name + "(x)", "R-CALL-TARGET", True))
for _name, _text in [
    ("walrus", "(a:=1)"),
    ("arrow", "a->b"),
    ("at", "a@b"),
    ("ellipsis", "..."),
    ("backslash", "x\\y"),
    ("not", "!x"),
]:
    MALICIOUS_CASES.append((_name, _text, "R-TOKEN", True))


@pytest.mark.parametrize(
    "case,text,rule,preparse",
    MALICIOUS_CASES,
    ids=[row[2] + "-reject-" + row[0] for row in MALICIOUS_CASES],
)
def test_malicious(case, text, rule, preparse):
    _assert_rejected(text, rule, preparse)


@pytest.mark.parametrize(
    "text", ["1 // 2**1025", "1 % 2**1025"], ids=["floor", "modulo"]
)
def test_forbidden_operator_precedes_power(text):
    _assert_rejected(text, "R-TOKEN", False)


@pytest.mark.parametrize(
    "text", ["x;y=1", "x=1", "sin(x=1)"], ids=["semicolon", "assignment", "keyword"]
)
def test_substitutions_disabled_R_SEMI_reject(text):
    with patch.object(
        sympy_parser, "parse_expr", side_effect=AssertionError("unexpected parser")
    ):
        with pytest.raises(parser.UnsafeExpressionError) as caught:
            parser.parse_expression(text, allow_substitutions=False)
    assert caught.value.rule == "R-SEMI"


def test_substitutions_disabled_accept():
    assert parser.parse_expression("x+1", allow_substitutions=False) == x + 1


@pytest.mark.parametrize(
    "name,rule",
    [
        ("a" * 65, "R-IDENT-LEN"),
        ("__x", "R-IDENT-DUNDER"),
        ("_\uff3fx\uff3f\uff3f", "R-IDENT-DUNDER"),
        ("\uff4c\uff41\uff4d\uff42\uff44\uff41", "R-IDENT-KEYWORD"),
        ("\uff49\uff4d\uff50\uff4f\uff52\uff54", "R-IDENT-KEYWORD"),
        ("a b", "R-IDENT-UNICODE"),
        (None, "R-IDENT-UNICODE"),
        ("\ufb03" * 22, "R-IDENT-LEN"),
    ],
    ids=[
        "length",
        "dunder",
        "nfkc-dunder",
        "nfkc-lambda",
        "nfkc-import",
        "space",
        "type",
        "nfkc-length",
    ],
)
def test_validate_identifier_reject(name, rule):
    with pytest.raises(parser.UnsafeExpressionError) as caught:
        parser.validate_identifier(name)
    assert caught.value.rule == rule


@pytest.mark.parametrize("name", ["_shape", "gamma_", "O", "\u03c3", "a" * 64])
def test_validate_identifier_accept(name):
    assert parser.validate_identifier(name) == name


@pytest.mark.parametrize(
    "source,expected",
    [
        ("max(1,x)", s.Function("max")(1, x)),
        ("__import__(x)", s.Function("__import__")(x)),
        ("abs(-1)", s.Integer(1)),
    ],
    ids=["max", "import-symbol-only", "abs"],
)
def test_p7_does_not_resolve_builtin_names(source, expected):
    assert parser._evaluate_curated(source) == expected


@pytest.mark.parametrize(
    "source",
    ["1/0", "0/0"],
    ids=["R-NONFINITE-reject-p7b", "R-NONFINITE-reject-p7b-nan"],
)
def test_p7b_nonfinite_isolated(source):
    with pytest.raises(parser.UnsafeExpressionError) as caught:
        parser._evaluate_curated(source)
    assert caught.value.rule == "R-NONFINITE"


def test_p7b_nonfinite_accept():
    assert parser._evaluate_curated("1/2") == s.Rational(1, 2)


@pytest.mark.parametrize(
    "pairs", [16, 17], ids=["R-PIECEWISE-accept-p9", "R-PIECEWISE-reject-p9"]
)
def test_post_validate_piecewise(pairs):
    expr = s.Piecewise(*[(s.Integer(i), x > i) for i in range(pairs)], evaluate=False)
    assert len(expr.args) == pairs
    if pairs == 16:
        assert parser._post_validate(expr) is None
    else:
        with pytest.raises(parser.UnsafeExpressionError) as caught:
            parser._post_validate(expr)
        assert caught.value.rule == "R-PIECEWISE"


@pytest.mark.parametrize(
    "expr,rule",
    [
        (s.Integer(1 << 4096), "R-RATIONAL-BITS"),
        (s.Rational(1, 1 << 4096), "R-RATIONAL-BITS"),
        (s.zoo, "R-NONFINITE"),
        (s.oo, "R-NONFINITE"),
        (s.nan, "R-NONFINITE"),
    ],
    ids=[
        "R-RATIONAL-BITS-reject-p9-integer",
        "R-RATIONAL-BITS-reject-p9-denominator",
        "R-NONFINITE-reject-p9-zoo",
        "R-NONFINITE-reject-p9-oo",
        "R-NONFINITE-reject-p9-nan",
    ],
)
def test_post_validate_reject(expr, rule):
    with pytest.raises(parser.UnsafeExpressionError) as caught:
        parser._post_validate(expr)
    assert caught.value.rule == rule


@pytest.mark.parametrize(
    "expr",
    [s.Integer(1 << 4095), s.Rational(1, 1 << 4095), s.Integer(1)],
    ids=[
        "R-RATIONAL-BITS-accept-p9-integer",
        "R-RATIONAL-BITS-accept-p9-denominator",
        "R-NONFINITE-accept-p9",
    ],
)
def test_post_validate_accept(expr):
    assert parser._post_validate(expr) is None


@pytest.mark.parametrize(
    "count", [2047, 2048], ids=["R-NODES-accept-p9-exact", "R-NODES-reject-p9-exact"]
)
def test_post_validate_node_count(count):
    expr = s.Add(*([x] * count), evaluate=False)
    assert sum(1 for _ in s.preorder_traversal(expr)) == count + 1
    if count == 2047:
        assert parser._post_validate(expr) is None
    else:
        with pytest.raises(parser.UnsafeExpressionError) as caught:
            parser._post_validate(expr)
        assert caught.value.rule == "R-NODES"


def test_rejection_has_no_raw_fallback():
    with patch.object(s, "sympify", side_effect=AssertionError("raw fallback")):
        with pytest.raises(parser.UnsafeExpressionError) as caught:
            parser.parse_expression("And(I,1)")
    assert caught.value.rule == "R-PARSE"


@pytest.mark.parametrize(
    "modules",
    [None, "numpy", "scipy", "numexpr", ["numpy", "scipy"], ("scipy", "numpy")],
    ids=[
        "R-BACKEND-accept-default",
        "R-BACKEND-accept-numpy",
        "R-BACKEND-accept-scipy",
        "R-BACKEND-accept-numexpr",
        "R-BACKEND-accept-list",
        "R-BACKEND-accept-tuple",
    ],
)
def test_backend_accept(modules):
    assert parser.validate_modules(modules) is None


@pytest.mark.parametrize(
    "modules",
    ["os", "tensorflow", ["numpy", "evil"], {"sin": np.sin}, 3, [np.sin]],
    ids=[
        "R-BACKEND-reject-os",
        "R-BACKEND-reject-tensorflow",
        "R-BACKEND-reject-list",
        "R-BACKEND-reject-dict",
        "R-BACKEND-reject-number",
        "R-BACKEND-reject-callable",
    ],
)
def test_backend_reject(modules):
    with pytest.raises(parser.UnsafeExpressionError) as caught:
        parser.lambdify_ordered([x], x, modules=modules)
    assert caught.value.rule == "R-BACKEND"


@pytest.mark.parametrize(
    "expr,modules,rule",
    [
        (x, ["numpy", "numexpr"], "R-BACKEND"),
        (parser.parse_expression("wofz(x)"), "numexpr", "R-BACKEND"),
        ("x", "numpy", "R-ARGS"),
    ],
    ids=[
        "R-BACKEND-reject-numpy-numexpr-combination",
        "R-BACKEND-reject-unsupported-numexpr-function",
        "R-ARGS-reject-non-expression",
    ],
)
def test_lambdify_ordered_exception_conversion(expr, modules, rule):
    with pytest.raises(parser.UnsafeExpressionError) as caught:
        parser.lambdify_ordered([x], expr, modules=modules)
    assert isinstance(caught.value, ValueError)
    assert caught.value.rule == rule


@pytest.mark.parametrize(
    "args,expr",
    [([x, x], x), (["x"], x), ([1], x), ([x], x + y), (x, x)],
    ids=[
        "R-ARGS-reject-duplicate",
        "R-ARGS-reject-string",
        "R-ARGS-reject-number",
        "R-ARGS-reject-uncovered",
        "R-ARGS-reject-sequence",
    ],
)
def test_lambdify_ordered_args_reject(args, expr):
    with pytest.raises(parser.UnsafeExpressionError) as caught:
        parser.lambdify_ordered(args, expr)
    assert caught.value.rule == "R-ARGS"


@pytest.mark.parametrize(
    "args,values",
    [([y, x, a], (2, 7, 11)), ([a, x, y], (11, 7, 2))],
    ids=["R-ARGS-accept", "R-ARGS-accept-reversed"],
)
def test_lambdify_ordered_args_accept(args, values):
    function = parser.lambdify_ordered(args, 3 * y + 5 * x + a)
    np.testing.assert_allclose(function(*values), 52, rtol=1e-12)


@pytest.mark.parametrize("name", ["sin", "cos", "exp", "numpy", "Dummy_0"])
def test_lambdify_ordered_collision(name):
    parameter = s.Symbol(name)
    function = parser.lambdify_ordered(
        [parameter, x], parameter + s.sin(x), modules="numpy"
    )
    np.testing.assert_allclose(
        function(3, np.array([0, np.pi / 2, np.pi])), [3, 4, 3], rtol=1e-12
    )


@pytest.mark.parametrize(
    "modules",
    [
        None,
        "numpy",
        "scipy",
        pytest.param(
            "numexpr",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("numexpr") is None,
                reason="numexpr is not installed",
            ),
        ),
    ],
)
def test_lambdify_ordered_numerical_backends(modules):
    function = parser.lambdify_ordered([x], x * x + 1, modules=modules)
    np.testing.assert_allclose(function(np.array([1, 2, 4])), [2, 5, 17], rtol=1e-12)


def test_lambdify_ordered_where_wofz():
    where = parser.lambdify_ordered(
        [x], parser.parse_expression("where(x>0,x,0)"), "numpy"
    )
    np.testing.assert_allclose(where(np.array([-2, 0, 3])), [0, 0, 3], rtol=1e-12)
    voigt = parser.lambdify_ordered(
        [x], parser.parse_expression("re(wofz(x+1j))"), ["numpy", "scipy"]
    )
    points = np.array([-1, 0, 2])
    np.testing.assert_allclose(voigt(points), np.real(wofz(points + 1j)), rtol=1e-12)


def test_caret_precedence():
    assert parser.parse_expression("x^2+1") == x**2 + 1
    assert parser.parse_expression("1+x^2") == 1 + x**2


def test_determinism_and_no_module_cache():
    keys = {name: id(value) for name, value in vars(parser).items()}
    namespace = parser.NS.copy()
    globals_before = parser.GD.copy()
    first = parser.parse_expression("a*x+b; a=2; b=3")
    assert first == 2 * x + 3
    with pytest.raises(parser.UnsafeExpressionError) as caught:
        parser.parse_expression("sin(x=1)")
    assert caught.value.rule == "R-KEYWORD"
    assert parser.parse_expression("a*x+b; a=2; b=3") == 2 * x + 3
    assert {name: id(value) for name, value in vars(parser).items()} == keys
    assert parser.NS == namespace
    assert parser.GD == globals_before
    assert parser.GD["__builtins__"] == {"isinstance": isinstance}


@pytest.mark.parametrize(
    "source",
    [
        "x.attr",
        "x[0]",
        "x[1:2]",
        "lambda: 1",
        "[i for i in x]",
        "{i for i in x}",
        "{i:i for i in x}",
        "(i for i in x)",
        "{}",
        "{1}",
        "[1]",
        'f"{x}"',
        "x if y else z",
        "x and y",
        "(x:=1)",
        "await x",
        "'text'",
        "b'text'",
        "None",
        "...",
        "x//2",
        "x%2",
        "x^2",
        "x is y",
        "x in y",
        "(x,y)",
    ],
    ids=[
        "Attribute",
        "Subscript",
        "Slice",
        "Lambda",
        "ListComp",
        "SetComp",
        "DictComp",
        "GeneratorExp",
        "Dict",
        "Set",
        "List",
        "JoinedStr",
        "IfExp",
        "BoolOp",
        "NamedExpr",
        "Await",
        "String",
        "Bytes",
        "None",
        "Ellipsis",
        "FloorDiv",
        "Mod",
        "BitXor",
        "Is",
        "In",
        "Tuple",
    ],
)
def test_ast_grammar_rejects_unreachable_nodes(source):
    # The public tokenizer already excludes these spellings; this isolates the
    # independent P5 grammar checks rather than crediting P1 for their coverage.
    tree = ast.parse(source, mode="eval")
    with pytest.raises(parser.UnsafeExpressionError) as caught:
        parser._validate_ast(tree)
    assert caught.value.rule == "R-AST-NODE"


@pytest.mark.parametrize(
    "node",
    [
        ast.Yield(value=ast.Name(id="x", ctx=ast.Load())),
        ast.Starred(value=ast.Name(id="x", ctx=ast.Load()), ctx=ast.Load()),
        ast.FormattedValue(value=ast.Name(id="x", ctx=ast.Load()), conversion=-1),
    ],
    ids=["Yield", "Starred", "FormattedValue"],
)
def test_ast_grammar_rejects_constructed_nodes(node):
    with pytest.raises(parser.UnsafeExpressionError) as caught:
        parser._validate_ast(ast.Expression(body=node))
    assert caught.value.rule == "R-AST-NODE"


@pytest.mark.parametrize(
    "source",
    ["x; unused=exp(10**9)", "x; unused=2**1025"],
    ids=["R-CONST-ARG-reject-unused-rhs", "R-CONST-POW-reject-unused-rhs"],
)
def test_substitution_rhs_checked_before_growth(source):
    _assert_rejected(source, "R-CONST-ARG" if "exp" in source else "R-CONST-POW", False)


@pytest.mark.parametrize(
    "extra",
    ["", "*2"],
    ids=[
        "R-RATIONAL-BITS-accept-float-magnitude",
        "R-RATIONAL-BITS-reject-float-magnitude",
    ],
)
def test_float_result_magnitude_boundary(extra):
    source = "1.0*2**1024*2**1024*2**1024*2**1024" + extra
    if not extra:
        with tripwires():
            assert parser.parse_expression(source) == s.Float(2) ** 4096
    else:
        _assert_rejected(source, "R-RATIONAL-BITS", False)
