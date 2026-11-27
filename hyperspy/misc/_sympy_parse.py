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

"""Compile a validated scientific expression, without a Python namespace.

Token and AST validation precede every SymPy call. Construction is unevaluated;
arithmetic is enabled one node at a time, after checking its evaluated operands.
Substitutions follow the same path, so folding cannot hide a large argument.

The bounds are best effort. They reject oversized literals and obvious
constant-magnitude blowups, they are not a resource bound, and nothing is
isolated in a separate process, so an expression this module accepts can still
consume unbounded time or memory. Parse text from files you trust only.
"""

import ast
import io
import keyword
import tokenize
import unicodedata

import sympy
from sympy.parsing import sympy_parser


class UnsafeExpressionError(ValueError):
    """An expression violates the grammar or a resource bound."""

    def __init__(self, rule: str, message: str):
        self.rule = rule
        super().__init__(f"[{rule}] {message}")


_ARITIES = {
    **dict.fromkeys(
        "sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh exp sqrt "
        "abs Abs erf erfc re im sign floor ceiling gamma wofz Not".split(),
        (1, 1),
    ),
    **dict.fromkeys("log ln Heaviside zeta".split(), (1, 2)),
    **dict.fromkeys(
        "atan2 pow Pow Mod Eq Ne Lt Le Gt Ge beta Rational".split(), (2, 2)
    ),
    **dict.fromkeys("And Or Xor".split(), (2, 16)),
    **dict.fromkeys("Min Max Piecewise".split(), (1, 16)),
    "where": (3, 3),
}


def _unevaluated(function):
    """Keep call syntax as inert as the parser's arithmetic operators."""

    def call(*args, **kwargs):
        return function(*args, evaluate=False)

    return call


_ALIASES = {
    "abs": sympy.Abs,
    "ln": sympy.log,
    "pow": sympy.Pow,
    "where": sympy.Function("where"),
    "wofz": sympy.Function("wofz"),
}
NS = {
    name: _unevaluated(_ALIASES[name] if name in _ALIASES else getattr(sympy, name))
    for name in _ARITIES
}
NS.update(pi=sympy.pi, E=sympy.E, I=sympy.I, Rational=sympy.Rational)
GD = {
    name: getattr(sympy, name)
    for name in (
        "Symbol Function Integer Float Rational I Add Mul Pow Lt Gt Le Ge Eq Ne "
        "And Or Not Xor"
    ).split()
}
# SymPy >= 1.15 rewrites unevaluated divisions into a lambda that calls
# isinstance; expose that single builtin so the restricted eval can run it.
GD["__builtins__"] = {"isinstance": isinstance}
_OPERATORS = set("+ - * / ** ^ ( ) , < > <= >= == != & | ~ ; =".split())
_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitAnd, ast.BitOr)
_UNARY = (ast.UAdd, ast.USub, ast.Invert)
_COMPARE = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
_EXP_LIMITS = {sympy.exp: 2048, sympy.sinh: 2048, sympy.cosh: 2048, sympy.tanh: 2048}
_IMAG_LIMITS = {
    sympy.sin: 2048,
    sympy.cos: 2048,
    sympy.tan: 2048,
    sympy.erf: 50,
    sympy.erfc: 50,
}
# erf and erfc saturate for real arguments beyond a few units; bounding the real
# part keeps oversized arguments out without relying on a backend-specific
# overflow during evaluation.
_REAL_LIMITS = {sympy.erf: 50, sympy.erfc: 50}
_GAMMA_FUNCTIONS = {sympy.gamma, sympy.beta, sympy.zeta}


def validate_identifier(name) -> str:
    """Validate both the written and Python-normalized identifier."""
    if not isinstance(name, str):
        raise UnsafeExpressionError("R-IDENT-UNICODE", "Expected an identifier.")
    for form in (name, unicodedata.normalize("NFKC", name)):
        if not form.isidentifier():
            raise UnsafeExpressionError("R-IDENT-UNICODE", "Invalid identifier.")
        if len(form) > 64:
            raise UnsafeExpressionError(
                "R-IDENT-LEN", "Identifier exceeds 64 characters."
            )
        if form.startswith("__") or form.endswith("__"):
            raise UnsafeExpressionError(
                "R-IDENT-DUNDER", "Double underscores are reserved."
            )
        if keyword.iskeyword(form) and name not in ("True", "False"):
            raise UnsafeExpressionError(
                "R-IDENT-KEYWORD", "Python keywords are not names."
            )
    return name


def validate_modules(modules):
    """Allow only the numerical backends used by HyperSpy expressions."""
    if modules is None:
        return
    choices = (modules,) if isinstance(modules, str) else modules
    if not isinstance(choices, (list, tuple)) or any(
        not isinstance(item, str) or item not in ("numpy", "numexpr", "scipy")
        for item in choices
    ):
        raise UnsafeExpressionError("R-BACKEND", "Unsupported numerical backend.")


def _segments(text, allow_substitutions):
    """Apply P0-P3 and return the token lists and substitution names."""
    if not isinstance(text, str):
        raise UnsafeExpressionError("R-TYPE", "Expected expression text.")
    if not text.strip():
        raise UnsafeExpressionError("R-EMPTY", "The expression is empty.")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise UnsafeExpressionError(
            "R-IDENT-UNICODE", "Invalid Unicode text."
        ) from error
    if size > 8192:
        raise UnsafeExpressionError("R-BYTES", "Expression exceeds 8192 UTF-8 bytes.")
    if any(
        (ord(char) < 32 and char not in "\t\n\r\f\v") or ord(char) == 127
        for char in text
    ):
        raise UnsafeExpressionError("R-CTRL", "Control characters are not allowed.")
    normalized = "".join(" " if char in "\t\n\r\f\v" else char for char in text).strip()
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(normalized).readline):
            if token.type in (tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL):
                continue
            if token.type not in (tokenize.NAME, tokenize.NUMBER, tokenize.OP) or (
                token.type == tokenize.OP and token.string not in _OPERATORS
            ):
                raise UnsafeExpressionError("R-TOKEN", "Unsupported token.")
            tokens.append(token)
    except (tokenize.TokenError, IndentationError) as error:
        raise UnsafeExpressionError("R-TOKEN", "Malformed token stream.") from error

    if not allow_substitutions and any(token.string in (";", "=") for token in tokens):
        raise UnsafeExpressionError("R-SEMI", "Substitutions are not allowed here.")
    parts = [[]]
    depth = 0
    for token in tokens:
        if token.string == ";" and depth == 0:
            parts.append([])
        else:
            parts[-1].append(token)
        depth += (token.string == "(") - (token.string == ")")
    if any(not part for part in parts):
        raise UnsafeExpressionError("R-SEMI", "Empty expression segment.")
    if len(parts) - 1 > 32:
        raise UnsafeExpressionError("R-SUBS", "More than 32 substitutions.")
    names = []
    expressions = []
    for index, part in enumerate(parts):
        assignments = []
        depth = 0
        for position, token in enumerate(part):
            if token.string == "=" and depth == 0:
                assignments.append(position)
            depth += (token.string == "(") - (token.string == ")")
        if index:
            if assignments != [1] or part[0].type != tokenize.NAME or len(part) < 3:
                raise UnsafeExpressionError(
                    "R-SUBS-LHS", "Expected IDENT = expression."
                )
            names.append(part[0].string)
            expressions.append(part[2:])
        else:
            if assignments:
                raise UnsafeExpressionError(
                    "R-SUBS-LHS", "Assignment in the base expression."
                )
            expressions.append(part)
    for token in tokens:
        if token.type == tokenize.NAME:
            validate_identifier(token.string)
    return expressions, names


def _source(tokens):
    """Rewrite caret precedence without rounding any numeric token."""
    rewritten = []
    for token in tokens:
        value = token.string
        if value == "^":
            value = "**"
        elif token.type == tokenize.NAME:
            # Python normalizes callee names too; use the same curated callable.
            normalized = unicodedata.normalize("NFKC", value)
            if normalized in NS:
                value = normalized
        rewritten.append((token.type, value))
    return tokenize.untokenize(rewritten).strip()


def _validate_ast(tree):
    """P5: validate grammar, call targets, arity, and tuple ownership."""
    stack = [(tree, None)]
    while stack:
        node, parent = stack.pop()
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ARITIES:
                raise UnsafeExpressionError("R-CALL-TARGET", "Unknown function target.")
            if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
                raise UnsafeExpressionError(
                    "R-KEYWORD", "Only positional arguments are allowed."
                )
            low, high = _ARITIES[node.func.id]
            if not low <= len(node.args) <= high:
                raise UnsafeExpressionError(
                    "R-ARITY", "Wrong number of function arguments."
                )
            if node.func.id == "Piecewise" and any(
                not isinstance(arg, ast.Tuple) or len(arg.elts) != 2
                for arg in node.args
            ):
                raise UnsafeExpressionError(
                    "R-PIECEWISE", "Piecewise needs expression/condition pairs."
                )
            if node.func.id == "Rational":
                integers = []
                for arg in node.args:
                    sign = 1
                    if isinstance(arg, ast.UnaryOp) and isinstance(
                        arg.op, (ast.UAdd, ast.USub)
                    ):
                        sign = -1 if isinstance(arg.op, ast.USub) else 1
                        arg = arg.operand
                    if not isinstance(arg, ast.Constant) or type(arg.value) is not int:
                        raise UnsafeExpressionError(
                            "R-ARITY", "Rational needs integer literals."
                        )
                    integers.append(sign * arg.value)
                if integers[1] == 0:
                    raise UnsafeExpressionError(
                        "R-ARITY", "Rational needs a nonzero denominator."
                    )
        elif isinstance(node, ast.Name):
            if node.id in ("nan", "inf", "oo", "zoo"):
                raise UnsafeExpressionError(
                    "R-NONFINITE", "Non-finite constants are not allowed."
                )
            callee = isinstance(parent, ast.Call) and parent.func is node
            if not callee and (
                node.id in _ARITIES or (node.id in GD and node.id != "I")
            ):
                raise UnsafeExpressionError(
                    "R-FUNC-VALUE", "A function cannot be used as a value."
                )
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise UnsafeExpressionError(
                    "R-CHAINED-CMP", "Chained comparisons are not allowed."
                )
        elif isinstance(node, ast.Tuple):
            if not (
                isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Name)
                and parent.func.id == "Piecewise"
                and node in parent.args
            ):
                raise UnsafeExpressionError("R-AST-NODE", "Unsupported Tuple node.")
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float, complex, bool):
                raise UnsafeExpressionError("R-AST-NODE", "Unsupported Constant node.")
        elif not isinstance(
            node,
            (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Load)
            + _BINARY
            + _UNARY
            + _COMPARE,
        ):
            raise UnsafeExpressionError(
                "R-AST-NODE", f"Unsupported {type(node).__name__} node."
            )
        stack.extend((child, node) for child in ast.iter_child_nodes(node))


def _validate_resources(tree, tokens):
    """P6: inspect raw shape and literal text before any SymPy construction."""
    nodes = calls = deepest = powers = 0
    # Python normalizes AST identifiers; SymPy preserves ordinary Symbol names.
    # Count the written names so Unicode aliases cannot hide distinct symbols.
    symbols = {
        token.string
        for token in tokens
        if token.type == tokenize.NAME
        and token.string not in ("True", "False")
        and unicodedata.normalize("NFKC", token.string) not in NS
    }
    stack = [(tree, 0, 0)]
    while stack:
        node, depth, power_depth = stack.pop()
        is_expression = isinstance(node, ast.expr)
        nodes += is_expression
        depth += is_expression
        deepest = max(deepest, depth)
        is_power = isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
        is_power |= isinstance(node, ast.Call) and node.func.id in ("pow", "Pow")
        power_depth += is_power
        powers = max(powers, power_depth)
        calls += isinstance(node, ast.Call)
        stack.extend(
            (child, depth, power_depth) for child in ast.iter_child_nodes(node)
        )
    if nodes > 2048:
        raise UnsafeExpressionError("R-NODES", "More than 2048 expression nodes.")
    if deepest > 160:
        raise UnsafeExpressionError("R-DEPTH", "Expression depth exceeds 160.")
    if calls > 128:
        raise UnsafeExpressionError("R-CALLS", "More than 128 function calls.")
    if len(symbols) > 128:
        raise UnsafeExpressionError("R-SYMBOLS", "More than 128 distinct symbols.")
    if powers > 4:
        raise UnsafeExpressionError("R-POW-NEST", "More than four nested powers.")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and node.value.bit_length() > 256
        ):
            raise UnsafeExpressionError(
                "R-INT-BITS", "Integer literal exceeds 256 bits."
            )
    for token in tokens:
        if token.type != tokenize.NUMBER:
            continue
        number = token.string.lower().replace("_", "")
        if number.startswith(("0x", "0o", "0b")) or not any(c in number for c in ".ej"):
            continue
        mantissa, _, exponent = number.rstrip("j").partition("e")
        digits = mantissa.replace(".", "").lstrip("0")
        if len(digits) > 64:
            raise UnsafeExpressionError(
                "R-FLOAT-DIGITS", "Float literal exceeds 64 significant digits."
            )
        if exponent:
            exponent_digits = exponent.lstrip("+-")
            if len(exponent_digits) > 4 or abs(int(exponent)) > 1000:
                raise UnsafeExpressionError(
                    "R-FLOAT-EXP", "Float exponent exceeds 1000."
                )


def _magnitude(value):
    """Return an exact symbolic magnitude for a numeric constant."""
    try:
        if value.is_nonnegative is True:
            magnitude = value
        elif value.is_nonpositive is True:
            magnitude = -value
        else:
            magnitude = sympy.Abs(value, evaluate=False)
        if magnitude.is_number is not True:
            raise TypeError("The constant has no numeric magnitude.")
        return magnitude
    except UnsafeExpressionError:
        raise
    except Exception as error:
        # Reject invalid numeric domains deterministically; never retry with
        # unbounded precision.
        raise UnsafeExpressionError(
            "R-EVAL-DOMAIN", "Cannot evaluate constant magnitude."
        ) from error


def _exceeds_magnitude(value, limit):
    """Compare exactly, rejecting an undecidable near-boundary value."""
    try:
        magnitude = _magnitude(value)
        if not (magnitude.is_Rational or magnitude.is_Float):
            # Preserve the bounded domain probe used for unstable SymPy
            # values (near poles and floor/ceiling precision failures).
            # Its rounded value is never used for the limit comparison.
            magnitude.evalf(15)
        difference = magnitude - limit
        if difference.is_positive is True:
            return True
        if difference.is_nonpositive is True:
            if not (magnitude.is_Rational or magnitude.is_Float):
                numeric = magnitude.evalf(80, strict=True)
                if numeric.is_real is not True or numeric.is_finite is not True:
                    raise TypeError("The constant has no finite real magnitude.")
            return False
        # A strict high-precision result must be negative at two precisions to
        # prove that the value is below the cap. Zero, disagreement, or an
        # undecidable sign is conservatively treated as exceeding it.
        for precision in (80, 160):
            approximate = difference.evalf(precision, strict=True)
            if approximate.is_negative is not True:
                return True
        return False
    except UnsafeExpressionError:
        raise
    except Exception as error:
        raise UnsafeExpressionError(
            "R-EVAL-DOMAIN", "Cannot compare constant magnitude."
        ) from error


def _power_growth_exceeds(base, exponent):
    """Conservatively bound numerator, denominator, and numeric power growth."""
    if base.is_Rational:
        scale = sympy.Integer(max(abs(base.p), base.q).bit_length())
    else:
        magnitude = _magnitude(base)
        if magnitude.is_zero is True:
            return False
        scale = sympy.Abs(sympy.log(magnitude, 2))
    return _exceeds_magnitude(scale * sympy.Abs(exponent), 4096)


def _is_boolean_operand(value):
    """Return whether a value is a Boolean atom, predicate, or relation."""
    return value.is_Boolean is True or isinstance(
        value, sympy.core.relational.Relational
    )


def _check_operand_types(function, children):
    """Keep Boolean values in predicate positions, never numeric positions."""
    name = function.__name__
    if name in ("And", "Or", "Xor", "Not", "Equality", "Unequality"):
        return
    if name == "ExprCondPair":
        invalid = _is_boolean_operand(children[0])
    elif function is sympy.Piecewise:
        invalid = False
    elif name == "where":
        invalid = any(_is_boolean_operand(child) for child in children[1:])
    else:
        invalid = any(_is_boolean_operand(child) for child in children)
    if invalid:
        raise UnsafeExpressionError(
            "R-EVAL-DOMAIN", "Boolean values cannot be used as numeric operands."
        )


def _check_before(function, children):
    """Bound evaluating constructors using their already evaluated operands."""
    if function is sympy.Pow:
        base, exponent = children
        if not exponent.free_symbols:
            if _exceeds_magnitude(exponent, 1024):
                raise UnsafeExpressionError(
                    "R-CONST-POW", "Constant exponent exceeds 1024."
                )
            if not base.free_symbols and _power_growth_exceeds(base, exponent):
                raise UnsafeExpressionError(
                    "R-RATIONAL-BITS", "Power would exceed 4096 bits."
                )
    if function in _EXP_LIMITS and not children[0].free_symbols:
        if _exceeds_magnitude(children[0], _EXP_LIMITS[function]):
            raise UnsafeExpressionError(
                "R-CONST-ARG", "Exponential argument exceeds 2048."
            )
    if function in _IMAG_LIMITS and not children[0].free_symbols:
        imaginary = children[0].as_real_imag()[1]
        if _exceeds_magnitude(imaginary, _IMAG_LIMITS[function]):
            raise UnsafeExpressionError(
                "R-CONST-ARG", "Imaginary argument exceeds its bound."
            )
    if function in _REAL_LIMITS and not children[0].free_symbols:
        # Reject here rather than during evaluation: with the gmpy2 backend the
        # overflow that used to signal this case never occurs.
        if _exceeds_magnitude(children[0].as_real_imag()[0], _REAL_LIMITS[function]):
            raise UnsafeExpressionError(
                "R-EVAL-DOMAIN", "Real argument exceeds its bound."
            )
    if function in _GAMMA_FUNCTIONS:
        for child in children:
            if not child.free_symbols and (
                child.is_real is not True or _exceeds_magnitude(child, 64)
            ):
                raise UnsafeExpressionError(
                    "R-CONST-ARG", "Expected a real argument of magnitude <= 64."
                )


def _check_result(value):
    """P7b post-call limits also protect intermediates that later cancel."""
    try:
        if value.has(sympy.nan, sympy.oo, -sympy.oo, sympy.zoo):
            raise UnsafeExpressionError(
                "R-NONFINITE", "Evaluation produced a non-finite value."
            )
        if not value.free_symbols and value.is_number:
            if value.is_Rational:
                oversized = max(abs(value.p), value.q).bit_length() > 4096
            else:
                oversized = _exceeds_magnitude(value, sympy.Integer(1 << 4096))
            if oversized:
                raise UnsafeExpressionError(
                    "R-RATIONAL-BITS", "Constant result exceeds 4096 bits."
                )
    except UnsafeExpressionError:
        raise
    except Exception as error:
        raise UnsafeExpressionError(
            "R-EVAL-DOMAIN", "Cannot validate function result."
        ) from error


def _checked_eval(tree):
    """P7b: evaluate bottom-up, with no recursion or persistent cache."""
    values = {}
    stack = [(tree, False)]
    while stack:
        node, ready = stack.pop()
        try:
            if node is True or node is False:
                values[id(node)] = sympy.true if node else sympy.false
                continue
            if not node.args:
                result = node
            elif not ready:
                stack.append((node, True))
                stack.extend((child, False) for child in reversed(node.args))
                continue
            else:
                children = tuple(values[id(child)] for child in node.args)
                _check_operand_types(node.func, children)
                _check_before(node.func, children)
                result = node.func(*children)
            _check_result(result)
        except UnsafeExpressionError:
            raise
        except Exception as error:
            raise UnsafeExpressionError(
                "R-EVAL-DOMAIN", "Invalid function arguments."
            ) from error
        values[id(node)] = result
    return values[id(tree)]


def _evaluate_curated(source):
    """Run P7/P7b only; callers must have completed P0-P6 validation."""
    try:
        tree = sympy_parser.parse_expr(
            source,
            local_dict=NS.copy(),
            global_dict=GD.copy(),
            transformations=sympy_parser.standard_transformations
            + (sympy_parser.convert_xor,),
            evaluate=False,
        )
    except UnsafeExpressionError:
        raise
    except Exception as error:
        # SymPy's construction errors have several types. Preserve their cause
        # but expose one deterministic grammar error; never retry with sympify.
        raise UnsafeExpressionError(
            "R-PARSE", "Cannot construct the expression."
        ) from error
    return _checked_eval(tree)


def _post_validate(expr):
    """P9: independently validate the returned tree, including new atoms."""
    try:
        for count, node in enumerate(sympy.preorder_traversal(expr), 1):
            if count > 2048:
                raise UnsafeExpressionError(
                    "R-NODES", "Evaluated tree exceeds 2048 nodes."
                )
            if node in (sympy.nan, sympy.oo, -sympy.oo, sympy.zoo) or (
                node.is_Float and node.is_finite is not True
            ):
                raise UnsafeExpressionError(
                    "R-NONFINITE", "Non-finite atom in the result."
                )
            if node.is_Rational and max(abs(node.p), node.q).bit_length() > 4096:
                raise UnsafeExpressionError(
                    "R-RATIONAL-BITS", "Rational atom exceeds 4096 bits."
                )
            if isinstance(node, sympy.Piecewise) and (
                not 1 <= len(node.args) <= 16
                or any(len(pair) != 2 for pair in node.args)
            ):
                raise UnsafeExpressionError(
                    "R-PIECEWISE", "Piecewise needs 1-16 pairs."
                )
    except UnsafeExpressionError:
        raise
    except Exception as error:
        raise UnsafeExpressionError(
            "R-EVAL-DOMAIN", "Cannot validate evaluated expression."
        ) from error


def parse_expression(text, *, allow_substitutions=True):
    """Validate and evaluate scientific text and its ordered substitutions."""
    segments, names = _segments(text, allow_substitutions)
    sources = [_source(segment) for segment in segments]
    trees = []
    for source in sources:
        try:
            trees.append(ast.parse(source, mode="eval"))
        except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
            raise UnsafeExpressionError(
                "R-PARSE", "Invalid expression syntax."
            ) from error
    for tree in trees:
        _validate_ast(tree)
    for tree, segment in zip(trees, segments):
        _validate_resources(tree, segment)
    expr = _evaluate_curated(sources[0])
    _post_validate(expr)
    for name, source in zip(names, sources[1:]):
        rhs = _evaluate_curated(source)
        _post_validate(rhs)
        try:
            lhs = sympy.Symbol(name)
            current_count = sum(1 for _ in sympy.preorder_traversal(expr))
            rhs_count = sum(1 for _ in sympy.preorder_traversal(rhs))
            after = current_count + expr.count(lhs) * (rhs_count - 1)
            if after > 2048:
                raise UnsafeExpressionError(
                    "R-GROWTH", "Substitution would exceed 2048 nodes."
                )
            with sympy.evaluate(False):
                substituted = expr.xreplace({lhs: rhs})
            expr = _checked_eval(substituted)
        except UnsafeExpressionError:
            raise
        except Exception as error:
            raise UnsafeExpressionError(
                "R-EVAL-DOMAIN", "Invalid substituted function arguments."
            ) from error
        _post_validate(expr)
    return expr


def lambdify_ordered(args, expr, modules=None):
    """Compile with fresh positional names, independent of parameter names."""
    validate_modules(modules)
    try:
        args = tuple(args)
    except TypeError as error:
        raise UnsafeExpressionError(
            "R-ARGS", "Expected an ordered sequence of Symbols."
        ) from error
    if (
        any(not isinstance(arg, sympy.Symbol) for arg in args)
        or len(set(args)) != len(args)
        or not isinstance(expr, sympy.Basic)
        or not expr.free_symbols.issubset(args)
    ):
        raise UnsafeExpressionError(
            "R-ARGS", "Arguments must cover every Symbol exactly once."
        )
    dummies = [sympy.Dummy() for _ in args]
    try:
        return sympy.utilities.lambdify(
            dummies,
            expr.xreplace(dict(zip(args, dummies))),
            modules=modules,
            dummify=False,
        )
    except UnsafeExpressionError:
        raise
    except Exception as error:
        raise UnsafeExpressionError(
            "R-BACKEND", "Backend cannot compile this expression."
        ) from error
