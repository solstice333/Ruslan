from enum import Enum
from typing import \
    Any, \
    TypeVar, \
    Callable, \
    ContextManager, \
    Optional, \
    Union, \
    overload, \
    Sequence, \
    List, \
    Tuple, \
    Dict, \
    Mapping, \
    MutableMapping, \
    NamedTuple, \
    Iterator, \
    Iterable
from types import TracebackType
from abc import ABC, abstractmethod
from anytree import NodeMixin  # type:ignore
from decimal import Decimal
from io import StringIO
from logging import getLogger, Logger

import re
import argparse
import logging
import logging.config
import typing

T = TypeVar('T')
BinOpTok = Union[
    'AddTok',
    'SubTok',
    'MulTok',
    'IntDivTok',
    'FloatDivTok',
    'LogicalAndTok',
    'LogicalOrTok',
    'BitwiseOrTok',
    'BitwiseXorTok',
    'BitwiseAndTok',
    'ModulusTok',
    'EqualTok',
    'NotEqualTok',
    'RightShiftTok',
    'GreaterEqualTok',
    'GreaterTok',
    'LeftShiftTok',
    'LessEqualTok',
    'LessTok'
]
UnOpTok = Union[
    'AddTok',
    'SubTok',
    'LogicalNotTok',
    'BitwiseNotTok'
]


def to_bool(s: str) -> bool:
    s = s.strip()
    assert re.fullmatch(r"true|false", s, re.I)
    return bool(re.fullmatch(r"true", s, re.I))


def to_int(s: str) -> int:
    s = s.strip()
    m = re.fullmatch(r"(0[xX])[0-9a-fA-F]+|\d+", s)
    assert m
    return int(m[0], 16) if m[1] else int(m[0])


def to_lower_snake_case(s: str) -> str:
    s = s.strip()
    s = s[0].lower() + s[1:]
    s = re.sub(r"([A-Z])", r"_\1", s)
    s = re.sub(r"_+", r"_", s)
    return s.lower()


def to_upper_camel_case(s: str) -> str:
    s = s.lower()
    s = re.sub(r"^\w", lambda m: m[0].upper(), s)
    s = re.sub(r"_(\w)", lambda m: m[1].upper(), s)
    s = re.sub(r"_", "", s)
    return s


def assert_with(cond: bool, err: Exception) -> None:
    if not cond:
        raise err


def any_of(vals: Iterable[T], pred: Callable[[T], bool]) -> bool:
    for val in vals:
        if pred(val):
            return True
    return False


@overload
def cast(
        ty: typing.Type[T],
        val: Any,
        exc_fn: Optional[
            Callable[[Iterable[typing.Type], Any, str], Exception]] = None
) -> T:
    pass


@overload
def cast(
        ty: Iterable[typing.Type],
        val: Any,
        exc_fn: Optional[
            Callable[[Iterable[typing.Type], Any, str], Exception]] = None
) -> Any:
    pass


@overload
def cast(
        ty: None,
        val: Any,
        exc_fn: Optional[
            Callable[[Iterable[typing.Type], Any, str], Exception]] = None
) -> None:
    pass


def cast(ty, val, exc_fn=None):
    ty = type(None) if ty is None else ty
    tys = ty if isinstance(ty, Iterable) else [ty]
    cond = any_of(tys, lambda ty2: isinstance(val, ty2))
    default_error_msg = \
        f"val is type {type(val).__name__} which is " \
        f"not a subtype of any of {[ty.__name__ for ty in tys]}"

    if exc_fn is not None:
        assert callable(exc_fn)
        assert_with(cond, exc_fn(tys, val, default_error_msg))
    else:
        assert cond, default_error_msg
    return val


def flattened_str(s: str):
    s = re.sub(r"\n", "", s)
    s = re.sub(r"(?<!:|,)\s+", "", s)
    return s


def setup_logging(verbose_stack: bool, verbose_scope: bool) -> None:
    def log_or_drop(verbose_flag: bool) -> int:
        return logging.DEBUG if verbose_flag else logging.CRITICAL + 1

    if verbose_stack or verbose_scope:
        logging.disable(logging.NOTSET)
        logging.config.dictConfig({
            "version": 1,
            "formatters": {
                "brief": {}
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "brief"
                }
            },
            "loggers": {
                "stack": {
                    "level": log_or_drop(verbose_stack)
                },
                "scope": {
                    "level": log_or_drop(verbose_scope)
                }
            },
            "root": {
                "level": logging.DEBUG,
                "handlers": ["console"]
            }
        })


class TokenTypeValue(NamedTuple):
    pat: str
    re: bool = False
    type: Callable[[str], Any] = str


class Position(NamedTuple):
    line: int
    col: int


class TokenType(Enum):
    PROGRAM = TokenTypeValue(pat=r"[pP][rR][oO][gG][rR][aA][mM]", re=True)
    VAR = TokenTypeValue(pat=r"[vV][aA][rR]", re=True)
    PROCEDURE = TokenTypeValue(
        pat=r"[pP][rR][oO][cC][eE][dD][uU][rR][eE]", re=True)
    INTEGER = TokenTypeValue(pat=r"[iI][nN][tT][eE][gG][eE][rR]", re=True)
    REAL = TokenTypeValue(pat=r"[rR][eE][aA][lL]", re=True)
    BOOLEAN = TokenTypeValue(pat=r"[bB][oO][oO][lL][eE][aA][nN]", re=True)
    INT_DIV = TokenTypeValue(pat=r"[dD][iI][vV]", re=True)
    BEGIN = TokenTypeValue(pat=r"[bB][eE][gG][iI][nN]", re=True)
    END = TokenTypeValue(pat=r"[eE][nN][dD]", re=True)
    IF = TokenTypeValue(pat=r"[iI][fF]", re=True)
    ELSE = TokenTypeValue(pat=r"[eE][lL][sS][eE]", re=True)
    THEN = TokenTypeValue(pat=r"[tT][hH][eE][nN]", re=True)
    TRUE = TokenTypeValue(pat=r"[tT][rR][uU][eE]", type=to_bool, re=True)
    FALSE = TokenTypeValue(pat=r"[fF][aA][lL][sS][eE]", type=to_bool, re=True)

    COMMA = TokenTypeValue(pat=",")
    REAL_CONST = TokenTypeValue(pat=r"\d+\.\d*", re=True, type=float)
    INT_CONST = TokenTypeValue(
        pat=r"0[xX][0-9a-fA-F]+|\d+", re=True, type=to_int)
    LOGICAL_OR = TokenTypeValue(pat=r'||')
    LOGICAL_AND = TokenTypeValue(pat='&&')
    BITWISE_OR = TokenTypeValue(pat='|')
    BITWISE_AND = TokenTypeValue(pat='&')
    BITWISE_XOR = TokenTypeValue(pat='^')
    MODULUS = TokenTypeValue(pat='%')
    EQUAL = TokenTypeValue(pat='==')
    NOT_EQUAL = TokenTypeValue(pat='!=')
    LOGICAL_NOT = TokenTypeValue(pat='!')
    BITWISE_NOT = TokenTypeValue(pat='~')
    RIGHT_SHIFT = TokenTypeValue(pat='>>')
    GREATER_EQUAL = TokenTypeValue(pat='>=')
    GREATER = TokenTypeValue(pat='>')
    LEFT_SHIFT = TokenTypeValue(pat='<<')
    LESS_EQUAL = TokenTypeValue(pat='<=')
    LESS = TokenTypeValue(pat='<')
    ADD = TokenTypeValue(pat='+')
    SUB = TokenTypeValue(pat='-')
    MUL = TokenTypeValue(pat='*')
    FLOAT_DIV = TokenTypeValue(pat='/')
    LPAR = TokenTypeValue(pat='(')
    RPAR = TokenTypeValue(pat=')')
    DOT = TokenTypeValue(pat=".")
    ID = TokenTypeValue(pat=r"[a-zA-Z_]\w*", re=True)
    ASSIGN = TokenTypeValue(pat=":=")
    COLON = TokenTypeValue(pat=":")
    SEMI = TokenTypeValue(pat=";")
    COMMENT = TokenTypeValue(pat=r"\{(?:.|\n)*?\}", re=True)
    NEWLINE = TokenTypeValue(pat=r"\n", re=True)
    EOF = TokenTypeValue(pat=r"$", re=True)

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def pattern(ident: 'TokenType') -> str:
        pat = ident.value.pat or ''
        return pat if ident.value.re else re.escape(pat)


class IToken(ABC):
    def __init__(
            self,
            ty: TokenType,
            value: Union[str, int, float],
            pos: Position
    ) -> None:
        self.type: TokenType = ty
        self._value: Union[str, int, float] = value
        self.pos: Position = pos

    @property  # type:ignore
    @abstractmethod
    def value(self) -> Union[str, int, float, bool]:
        return self._value

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.value})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value}, {self.pos})"

    def __bool__(self) -> bool:
        return bool(self.value)

    def __eq__(self, other) -> bool:
        return self.type == other.type and \
               (
                   self.value.lower() == other.value.lower() \
                       if isinstance(self.value, str) else \
                       self.value == other.value
               ) and \
               self.pos == other.pos

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class ProgramTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.PROGRAM, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class VarTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.VAR, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class ProcedureTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.PROCEDURE, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class CommaTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.COMMA, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class IntegerTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.INTEGER, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class RealTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.REAL, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class BooleanTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.BOOLEAN, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class IntConstTok(IToken):
    def __init__(self, value: int, pos: Position) -> None:
        super().__init__(TokenType.INT_CONST, value, pos)

    @property
    def value(self) -> int:
        return cast(int, self._value)


class RealConstTok(IToken):
    def __init__(self, value: float, pos: Position) -> None:
        super().__init__(TokenType.REAL_CONST, value, pos)

    @property
    def value(self) -> float:
        return cast(float, self._value)


class AddTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.ADD, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class SubTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.SUB, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class MulTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.MUL, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class IntDivTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.INT_DIV, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class FloatDivTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.FLOAT_DIV, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class LparTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.LPAR, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class RparTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.RPAR, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class EofTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.EOF, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class BeginTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.BEGIN, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class EndTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.END, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class DotTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.DOT, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class IdTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.ID, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class AssignTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.ASSIGN, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class ColonTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.COLON, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class SemiTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.SEMI, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class IfTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.IF, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class ElseTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.ELSE, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class ThenTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.THEN, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class LogicalAndTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.LOGICAL_AND, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class LogicalOrTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.LOGICAL_OR, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class BitwiseOrTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.BITWISE_OR, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class BitwiseAndTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.BITWISE_AND, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class BitwiseXorTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.BITWISE_XOR, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class TrueTok(IToken):
    def __init__(self, value: bool, pos: Position) -> None:
        super().__init__(TokenType.TRUE, value, pos)

    @property
    def value(self) -> bool:
        return cast(bool, self._value)


class FalseTok(IToken):
    def __init__(self, value: bool, pos: Position) -> None:
        super().__init__(TokenType.FALSE, value, pos)

    @property
    def value(self) -> bool:
        return cast(bool, self._value)


class ModulusTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.MODULUS, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class EqualTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.EQUAL, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class NotEqualTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.NOT_EQUAL, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class LogicalNotTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.LOGICAL_NOT, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class BitwiseNotTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.BITWISE_NOT, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class RightShiftTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.RIGHT_SHIFT, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class GreaterEqualTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.GREATER_EQUAL, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class GreaterTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.GREATER, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class LeftShiftTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.LEFT_SHIFT, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class LessEqualTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.LESS_EQUAL, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class LessTok(IToken):
    def __init__(self, value: str, pos: Position) -> None:
        super().__init__(TokenType.LESS, value, pos)

    @property
    def value(self) -> str:
        return cast(str, self._value)


class ErrorCode(Enum):
    UNEXPECTED_TOKEN = 'Unexpected token'
    UNEXPECTED_OPERAND_TYPE = 'Unexpected operand type'
    ID_NOT_FOUND = 'Identifier not found'
    DUPLICATE_ID = 'Duplicate id found'
    TYPE_NOT_FOUND = 'Type not found'
    PROC_NOT_FOUND = 'Proc not found'
    BAD_PARAMS = 'Parameter list does not match declaration'


class Error(Exception):
    def __init__(
            self,
            error_code: ErrorCode,
            token: IToken,
            appended_message: Optional[Union[str, Callable[[], str]]] = None,
            message: Optional[Union[str, Callable[[], str]]] = None
    ):
        self.error_code: ErrorCode = error_code
        self.token: IToken = token

        message2 = message() if callable(message) else message
        appended_message2 = appended_message() if callable(appended_message) \
            else appended_message

        if message2 is None:
            message2 = f"{error_code.value} -> {token!r}"
        if appended_message2:
            message2 += f". {appended_message2}"

        self.message = f"{type(self).__name__}: {message2}"
        super().__init__(self.message)


class ParserError(Error):
    pass


class SemanticError(Error):
    pass


class InterpreterError(Error):
    pass


class Lexer(Iterable[IToken]):
    class TokenTypeInfo(NamedTuple):
        tokty: TokenType
        pattern: str
        token: IToken = EofTok("", Position(line=0, col=0))

    __TOKEN_NAME_TO_TTY_INFO: Dict[str, TokenTypeInfo] = {}

    @staticmethod
    def token_ctor(name: str) -> \
            Callable[[Union[str, int, float], Position], IToken]:
        ctor_name = to_upper_camel_case(name) + "Tok"
        return globals()[ctor_name]

    @classmethod
    def _token_name_to_tty_info(cls) -> Dict[str, TokenTypeInfo]:
        if not cls.__TOKEN_NAME_TO_TTY_INFO:
            cls.__TOKEN_NAME_TO_TTY_INFO = \
                {
                    tty.name:
                        cls.TokenTypeInfo(
                            tokty=tty,
                            pattern=TokenType.pattern(tty)
                        )
                    for tty in TokenType
                }
        return cls.__TOKEN_NAME_TO_TTY_INFO

    def __init__(self, text: str) -> None:
        self._text: str = text
        self.linenum: int = 1
        self.newline_anchor: int = -1

    def _iter_tokens(self) -> Iterator[IToken]:
        token_spec = Lexer._token_name_to_tty_info()

        token_pats = [
            rf"(?P<{name}>{tty_info.pattern})"
            for name, tty_info in token_spec.items()
        ]
        token_pat = "|".join(token_pats)

        for m in re.finditer(token_pat, self._text):
            name = m.lastgroup if m.lastgroup else ''
            tty = token_spec[name].tokty

            if any_of(
                    [TokenType.NEWLINE, TokenType.COMMENT],
                    lambda tty_elem: tty == tty_elem
            ):
                if tty == TokenType.NEWLINE:
                    self.linenum += 1
                    self.newline_anchor = m.start()
                continue

            yield Lexer.token_ctor(name)(
                tty.value.type(m[name]),
                Position(
                    line=self.linenum,
                    col=m.start(name) - self.newline_anchor
                )
            )

    def __iter__(self) -> Iterator[IToken]:
        return self._iter_tokens()


class IAST(ABC, NodeMixin):
    def __init__(self, children: Optional[List['IAST']] = None):
        self.children: Tuple['IAST', ...] = tuple(children or [])

    @property
    def kids(self) -> Tuple['IAST', ...]:
        assert isinstance(self.children, tuple)
        return self.children

    @abstractmethod
    def __str__(self) -> str:
        return f"{type(self).__name__}({self.kids})"

    def __repr__(self) -> str:
        return str(self)


class Program(IAST):
    def __init__(self, name: str, block: 'Block') -> None:
        self.name: str = name
        self.block: 'Block' = block
        self.proc_sym: Optional[ProcSymbol] = None
        super().__init__([self.block])

    def __str__(self) -> str:
        return f"{type(self).__name__}(name={self.name})"


class IDecl(IAST):
    pass


class Block(IAST):
    def __init__(
            self,
            declarations: List['IDecl'],
            compound_statement: 'Compound'
    ) -> None:
        self.declarations: List['IDecl'] = declarations
        self.compound_statement: 'Compound' = compound_statement
        super().__init__(self.declarations + [self.compound_statement])

    def __str__(self) -> str:
        return f"{type(self).__name__}()"


class VarDecl(IDecl):
    def __init__(self, var: 'Var', ty: 'Type') -> None:
        self.var: 'Var' = var
        self.type: 'Type' = ty
        super().__init__([self.var, self.type])

    def __str__(self) -> str:
        return f"{type(self).__name__}()"


class Param(VarDecl):
    def __init__(self, var: 'Var', ty: 'Type') -> None:
        super().__init__(var, ty)


class ProcDecl(IDecl):
    def __init__(self, proc_name: str, params: List[Param], block_node: Block):
        self.name: str = proc_name
        self.params: List[Param] = params
        self.block: Block = block_node
        super().__init__(self.params + [self.block])

    def __str__(self) -> str:
        return f"{type(self).__name__}(name={self.name})"


class Type(IAST):
    def __init__(
            self,
            arg: Union[
                IntegerTok,
                RealTok,
                BooleanTok,
                'Type'
            ]
    ) -> None:
        super().__init__()

        if isinstance(arg, (IntegerTok, RealTok, BooleanTok)):
            self.token: Union[IntegerTok, RealTok, BooleanTok] = arg
            self.value: str = cast(str, self.token.value)
        else:
            ty = cast(Type, arg)
            self.token = ty.token
            self.value = ty.value

    def __str__(self) -> str:
        return f"{type(self).__name__}(value={self.value})"


class BinOp(IAST):
    def __init__(
            self,
            left: IAST,
            right: IAST,
            optok: BinOpTok
    ) -> None:
        self.token: BinOpTok = optok
        self.value: str = self.token.value
        self.left: IAST = left
        self.right: IAST = right
        super().__init__([self.left, self.right])

    def __str__(self) -> str:
        return f"{type(self).__name__}(value={self.value})"


class Add(BinOp):
    def __init__(
            self,
            left: IAST,
            right: IAST,
            optok: AddTok
    ) -> None:
        super().__init__(left, right, optok)


class Sub(BinOp):
    def __init__(
            self,
            left: IAST,
            right: IAST,
            optok: SubTok
    ) -> None:
        super().__init__(left, right, optok)


class Mul(BinOp):
    def __init__(
            self,
            left: IAST,
            right: IAST,
            optok: MulTok
    ) -> None:
        super().__init__(left, right, optok)


class IntDiv(BinOp):
    def __init__(
            self,
            left: IAST,
            right: IAST,
            optok: IntDivTok
    ) -> None:
        super().__init__(left, right, optok)


class FloatDiv(BinOp):
    def __init__(
            self,
            left: IAST,
            right: IAST,
            optok: FloatDivTok
    ) -> None:
        super().__init__(left, right, optok)


class LogicalAnd(BinOp):
    def __init__(
            self,
            left: IAST,
            right: IAST,
            optok: LogicalAndTok
    ) -> None:
        super().__init__(left, right, optok)


class LogicalOr(BinOp):
    def __init__(
            self,
            left: IAST,
            right: IAST,
            optok: LogicalOrTok
    ) -> None:
        super().__init__(left, right, optok)


class BitwiseOr(BinOp):
    def __init__(
            self,
            left: IAST,
            right: IAST,
            optok: BitwiseOrTok
    ) -> None:
        super().__init__(left, right, optok)


class BitwiseXor(BinOp):
    def __init__(
            self,
            left: IAST,
            right: IAST,
            optok: BitwiseXorTok
    ) -> None:
        super().__init__(left, right, optok)


class BitwiseAnd(BinOp):
    def __init__(
            self,
            left: IAST,
            right: IAST,
            optok: BitwiseAndTok
    ) -> None:
        super().__init__(left, right, optok)


class Modulus(BinOp):
    def __init__(self, left: IAST, right: IAST, optok: ModulusTok) -> None:
        super().__init__(left, right, optok)


class Equal(BinOp):
    def __init__(self, left: IAST, right: IAST, optok: EqualTok):
        super().__init__(left, right, optok)


class NotEqual(BinOp):
    def __init__(self, left: IAST, right: IAST, optok: NotEqualTok):
        super().__init__(left, right, optok)


class RightShift(BinOp):
    def __init__(self, left: IAST, right: IAST, optok: RightShiftTok) -> None:
        super().__init__(left, right, optok)


class GreaterEqual(BinOp):
    def __init__(self, left: IAST, right: IAST, optok: GreaterEqualTok) -> None:
        super().__init__(left, right, optok)


class Greater(BinOp):
    def __init__(self, left: IAST, right: IAST, optok: GreaterTok) -> None:
        super().__init__(left, right, optok)


class LeftShift(BinOp):
    def __init__(self, left: IAST, right: IAST, optok: LeftShiftTok) -> None:
        super().__init__(left, right, optok)


class LessEqual(BinOp):
    def __init__(self, left: IAST, right: IAST, optok: LessEqualTok) -> None:
        super().__init__(left, right, optok)


class Less(BinOp):
    def __init__(self, left: IAST, right: IAST, optok: LessTok) -> None:
        super().__init__(left, right, optok)


class UnOp(IAST):
    def __init__(self, right: IAST, optok: UnOpTok) -> None:
        self.right: IAST = right
        super().__init__([self.right])
        self.token: UnOpTok = optok
        self.value = self.token.value

    def __str__(self) -> str:
        return f"{type(self).__name__}(value={self.value})"


class LogicalNot(UnOp):
    def __init__(self, right: IAST, optok: LogicalNotTok) -> None:
        super().__init__(right, optok)


class BitwiseNot(UnOp):
    def __init__(self, right: IAST, optok: BitwiseNotTok) -> None:
        super().__init__(right, optok)


class Pos(UnOp):
    def __init__(self, right: IAST, optok: AddTok) -> None:
        super().__init__(right, optok)


class Neg(UnOp):
    def __init__(self, right: IAST, optok: SubTok) -> None:
        super().__init__(right, optok)


class Num(IAST):
    def __init__(self, numtok: Union[IntConstTok, RealConstTok]) -> None:
        super().__init__()
        self.token: Union[IntConstTok, RealConstTok] = numtok
        self.value: Union[int, float] = self.token.value

    def __str__(self) -> str:
        return f"{type(self).__name__}(value={self.value})"


class Bool(IAST):
    def __init__(self, booltok: Union[TrueTok, FalseTok]) -> None:
        super().__init__()
        self.value: bool = booltok.value
        self.token: Union[TrueTok, FalseTok] = booltok

    def __str__(self) -> str:
        return f"{type(self).__name__}(value={self.value})"


class Compound(IAST):
    def __init__(self, children: Optional[List[IAST]] = None) -> None:
        super().__init__(children)

    def __str__(self) -> str:
        return f"{type(self).__name__}()"


class Branch(IAST):
    def __init__(self, cond: IAST, if_blk: IAST, else_blk: IAST) -> None:
        self.cond: IAST = cond
        self.if_blk: IAST = if_blk
        self.else_blk: IAST = else_blk
        super().__init__([self.cond, self.if_blk, self.else_blk])

    def __str__(self) -> str:
        return f"{type(self).__name__}()"


class Var(IAST):
    def __init__(self, idtok: IdTok) -> None:
        super().__init__()
        self.token: IdTok = idtok
        self.value: str = cast(str, self.token.value)

    def __str__(self) -> str:
        return f"{type(self).__name__}(value={self.value})"


class Assign(IAST):
    def __init__(
            self,
            left: Var,
            right: IAST,
            optok: AssignTok
    ) -> None:
        self.left: Var = left
        self.right: IAST = right
        super().__init__([self.left, self.right])
        self.token: AssignTok = optok
        self.value = self.token.value

    def __str__(self) -> str:
        return f"{type(self).__name__}(value={self.value})"


class ProcCall(IAST):
    def __init__(
            self,
            token: IToken,
            actual_params: List[IAST]
    ) -> None:
        self.token: IToken = token
        self.proc_name: str = cast(str, self.token.value)
        self.actual_params: List[IAST] = actual_params
        self.proc_sym: Optional[ProcSymbol] = None
        super().__init__(self.actual_params)

    def __str__(self) -> str:
        return f"{type(self).__name__}(proc_name={self.proc_name})"


class NoOp(IAST):
    def __str__(self) -> str:
        return f"{type(self).__name__}()"


class Eof(NoOp):
    pass


class Parser:
    def __init__(self, lexer: Lexer) -> None:
        self.lexer: Lexer = lexer
        self._it: Iterator[IToken] = iter(self.lexer)
        self.current_token: IToken = next(self._it)
        self._tokbuf: List[IToken] = []

    def _assert(
            self,
            cond,
            errcode: ErrorCode,
            token: IToken,
            msg: Optional[Union[str, Callable[[], str]]] = None
    ) -> None:
        assert_with(
            cond,
            ParserError(
                error_code=errcode,
                token=token,
                appended_message=msg
            )
        )

    def advance(self) -> IToken:
        if self._tokbuf:
            self.current_token = self._tokbuf.pop(0)
        elif self.current_token.type != TokenType.EOF:
            self.current_token = next(self._it)
        return self.current_token

    def peek(self, offset: int = 1) -> IToken:
        if len(self._tokbuf) >= offset:
            return self._tokbuf[offset - 1]

        for i in range(0, offset - len(self._tokbuf)):
            self._tokbuf.append(next(self._it))

        return self._tokbuf[offset - 1]

    def eat(self, toktypes: Union[TokenType, Sequence[TokenType]]) -> IToken:
        if isinstance(toktypes, TokenType):
            toktypes = toktypes,

        assert len(toktypes) > 0, "toktypes must have at least one TokenType"

        def gen_msg() -> str:
            toktypes_s = [toktype.name
                          for toktype in cast([tuple, list], toktypes)]
            toktypes_flat_s = toktypes_s[0] if len(toktypes_s) == 1 \
                else f"one of {toktypes_s}"
            return f"Expected {toktypes_flat_s}"

        self._assert(
            cond=any_of(
                toktypes,
                lambda tokty: self.current_token.type == tokty
            ),
            errcode=ErrorCode.UNEXPECTED_TOKEN,
            token=self.current_token,
            msg=gen_msg
        )

        return self.advance()

    def factor(self) -> IAST:
        """
        factor :
            ADD factor |
            SUB factor |
            LOGICAL_NOT factor |
            BITWISE_NOT factor |
            INT_CONST |
            REAL_CONST |
            True |
            False |
            LPAREN root_expr RPAREN |
            variable
        """
        curtok = self.current_token
        prevtok = curtok
        num_tok_types = [TokenType.INT_CONST, TokenType.REAL_CONST]

        if curtok.type == TokenType.ADD:
            self.eat(TokenType.ADD)
            ast = Pos(self.factor(), cast(AddTok, prevtok))
        elif curtok.type == TokenType.SUB:
            self.eat(TokenType.SUB)
            ast = Neg(self.factor(), cast(SubTok, prevtok))
        elif curtok.type == TokenType.LOGICAL_NOT:
            self.eat(TokenType.LOGICAL_NOT)
            ast = LogicalNot(self.factor(), cast(LogicalNotTok, curtok))
        elif curtok.type == TokenType.BITWISE_NOT:
            self.eat(TokenType.BITWISE_NOT)
            ast = BitwiseNot(self.factor(), cast(BitwiseNotTok, curtok))
        elif any_of(
                num_tok_types,
                lambda numtype: curtok.type == numtype
        ):
            ast = Num(cast([IntConstTok, RealConstTok], curtok))
            self.eat(num_tok_types)

            curtok = self.current_token
            self._assert(
                cond=curtok.type != TokenType.LPAR,
                errcode=ErrorCode.UNEXPECTED_TOKEN,
                token=curtok
            )
        elif curtok.type == TokenType.TRUE:
            ast = Bool(cast(TrueTok, curtok))
            self.eat(TokenType.TRUE)
        elif curtok.type == TokenType.FALSE:
            ast = Bool(cast(FalseTok, curtok))
            self.eat(TokenType.FALSE)
        elif curtok.type == TokenType.LPAR:
            self.eat(TokenType.LPAR)
            ast = self.root_expr()
            self.eat(TokenType.RPAR)
        else:
            ast = self.variable()
        return ast

    def term(self) -> IAST:
        """term : factor ((MUL | INT_DIV | FLOAT_DIV | MODULUS) factor)*"""
        node: IAST = self.factor()

        while True:
            curtok: IToken = self.current_token
            if curtok.type == TokenType.MUL:
                self.eat(TokenType.MUL)
                node = Mul(
                    node, self.factor(), cast(MulTok, curtok))
            elif curtok.type == TokenType.INT_DIV:
                self.eat(TokenType.INT_DIV)
                node = IntDiv(
                    node, self.factor(), cast(IntDivTok, curtok))
            elif curtok.type == TokenType.FLOAT_DIV:
                self.eat(TokenType.FLOAT_DIV)
                node = FloatDiv(
                    node, self.factor(), cast(FloatDivTok, curtok))
            elif curtok.type == TokenType.MODULUS:
                self.eat(TokenType.MODULUS)
                node = Modulus(
                    node, self.factor(), cast(ModulusTok, curtok))
            else:
                break

        return node

    def expr(self) -> IAST:
        """expr: term ((ADD | SUB) term)* """
        node: IAST = self.term()

        while True:
            curtok: IToken = self.current_token
            if curtok.type == TokenType.ADD:
                self.eat(TokenType.ADD)
                node = Add(node, self.term(), cast(AddTok, curtok))
            elif curtok.type == TokenType.SUB:
                self.eat(TokenType.SUB)
                node = Sub(node, self.term(), cast(SubTok, curtok))
            else:
                break

        return node

    def bitwise_left_right_expr(self) -> IAST:
        """
        bitwise_left_right_expr: expr ((LEFT_SHIFT | RIGHT_SHIFT) expr)*
        """
        node: IAST = self.expr()
        while True:
            curtok: IToken = self.current_token
            if curtok.type == TokenType.LEFT_SHIFT:
                self.eat(TokenType.LEFT_SHIFT)
                node = LeftShift(
                    node, self.expr(), cast(LeftShiftTok, curtok))
            elif curtok.type == TokenType.RIGHT_SHIFT:
                self.eat(TokenType.RIGHT_SHIFT)
                node = RightShift(
                    node, self.expr(), cast(RightShiftTok, curtok))
            else:
                break
        return node

    def gt_gte_lt_lte_expr(self) -> IAST:
        """
        gt_gte_lt_lte_expr: bitwise_left_right_expr
            (
                (GREATER | GREATER_EQUAL | LESS | LESS_EQUAL)
                bitwise_left_right_expr
            )*
        """
        node: IAST = self.bitwise_left_right_expr()
        while True:
            curtok: IToken = self.current_token
            if curtok.type == TokenType.GREATER:
                self.eat(TokenType.GREATER)
                node = Greater(
                    node,
                    self.bitwise_left_right_expr(),
                    cast(GreaterTok, curtok)
                )
            elif curtok.type == TokenType.GREATER_EQUAL:
                self.eat(TokenType.GREATER_EQUAL)
                node = GreaterEqual(
                    node,
                    self.bitwise_left_right_expr(),
                    cast(GreaterEqualTok, curtok)
                )
            elif curtok.type == TokenType.LESS:
                self.eat(TokenType.LESS)
                node = Less(
                    node,
                    self.bitwise_left_right_expr(),
                    cast(LessTok, curtok)
                )
            elif curtok.type == TokenType.LESS_EQUAL:
                self.eat(TokenType.LESS_EQUAL)
                node = LessEqual(
                    node,
                    self.bitwise_left_right_expr(),
                    cast(LessEqualTok, curtok)
                )
            else:
                break
        return node

    def equal_not_equal_expr(self) -> IAST:
        """
        equal_not_equal_expr:
            gt_gte_lt_lte_expr ((EQUAL | NOT_EQUAL) gt_gte_lt_lte_expr)*
        """
        node: IAST = self.gt_gte_lt_lte_expr()
        while True:
            curtok: IToken = self.current_token
            if curtok.type == TokenType.EQUAL:
                self.eat(TokenType.EQUAL)
                node = Equal(
                    node,
                    self.gt_gte_lt_lte_expr(),
                    cast(EqualTok, curtok)
                )
            elif curtok.type == TokenType.NOT_EQUAL:
                self.eat(TokenType.NOT_EQUAL)
                node = NotEqual(
                    node,
                    self.gt_gte_lt_lte_expr(),
                    cast(NotEqualTok, curtok)
                )
            else:
                break
        return node

    def bitwise_and_expr(self) -> IAST:
        """
        bitwise_and_expr:
            equal_not_equal_expr (BITWISE_AND equal_not_equal_expr)*
        """
        node: IAST = self.equal_not_equal_expr()
        while True:
            curtok: IToken = self.current_token
            if curtok.type == TokenType.BITWISE_AND:
                self.eat(TokenType.BITWISE_AND)
                node = BitwiseAnd(node, self.equal_not_equal_expr(),
                                  cast(BitwiseAndTok, curtok))
            else:
                break
        return node

    def bitwise_xor_expr(self) -> IAST:
        """
        bitwise_xor_expr:
            bitwise_and_expr (BITWISE_XOR bitwise_and_expr)*
        """
        node: IAST = self.bitwise_and_expr()
        while True:
            curtok: IToken = self.current_token
            if curtok.type == TokenType.BITWISE_XOR:
                self.eat(TokenType.BITWISE_XOR)
                node = BitwiseXor(node, self.bitwise_and_expr(),
                                  cast(BitwiseXorTok, curtok))
            else:
                break
        return node

    def bitwise_or_expr(self) -> IAST:
        """bitwise_or_expr: bitwise_xor_expr (BITWISE_OR bitwise_xor_expr)*"""
        node: IAST = self.bitwise_xor_expr()
        while True:
            curtok: IToken = self.current_token
            if curtok.type == TokenType.BITWISE_OR:
                self.eat(TokenType.BITWISE_OR)
                node = BitwiseOr(node, self.bitwise_xor_expr(),
                                 cast(BitwiseOrTok, curtok))
            else:
                break
        return node

    def logical_and_expr(self) -> IAST:
        """logical_and_expr: bitwise_or_expr (LOGICAL_AND bitwise_or_expr)*"""
        node: IAST = self.bitwise_or_expr()
        while True:
            curtok: IToken = self.current_token
            if curtok.type == TokenType.LOGICAL_AND:
                self.eat(TokenType.LOGICAL_AND)
                node = LogicalAnd(node, self.bitwise_or_expr(),
                                  cast(LogicalAndTok, curtok))
            else:
                break
        return node

    def logical_or_expr(self) -> IAST:
        """logical_or_expr: logical_and_expr (LOGICAL_OR logical_and_expr)*"""
        node: IAST = self.logical_and_expr()
        while True:
            curtok: IToken = self.current_token
            if curtok.type == TokenType.LOGICAL_OR:
                self.eat(TokenType.LOGICAL_OR)
                node = LogicalOr(node, self.logical_and_expr(),
                                 cast(LogicalOrTok, curtok))
            else:
                break
        return node

    def root_expr(self) -> IAST:
        """root_expr: logical_or_expr"""
        return self.logical_or_expr()

    def empty(self) -> NoOp:
        """empty rule"""
        return NoOp()

    def variable(self) -> Var:
        """variable: ID"""
        curtok = self.current_token
        self.eat(TokenType.ID)
        return Var(cast(IdTok, curtok))

    def assignment_statement(self) -> Assign:
        """assignment_statement: variable ASSIGN root_expr"""
        left = self.variable()
        assign_tok = self.current_token
        self.eat(TokenType.ASSIGN)
        right = self.root_expr()
        return Assign(left, right, cast(AssignTok, assign_tok))

    def actual_parameter_list(self) -> List[IAST]:
        """arglist: root_expr (COMMA root_expr)*"""
        actual_params = []
        start = True
        while self.current_token.type != TokenType.RPAR:
            if start:
                start = False
            else:
                self.eat(TokenType.COMMA)
            node = self.root_expr()
            actual_params.append(node)
        return actual_params

    def proccall_statement(self) -> ProcCall:
        """proccall_statement: ID LPAR actual_parameter_list? RPAR"""
        idtok = self.current_token

        self.eat(TokenType.ID)
        self.eat(TokenType.LPAR)
        actual_params = self.actual_parameter_list()
        self.eat(TokenType.RPAR)

        return ProcCall(
            token=idtok,
            actual_params=actual_params
        )

    def if_statement(self) -> IAST:
        """
        if_statement:
            IF LPAR root_expr RPAR THEN statement (ELSE statement)?
        """
        self.eat(TokenType.IF)
        self.eat(TokenType.LPAR)
        cond_node = self.root_expr()
        self.eat(TokenType.RPAR)
        self.eat(TokenType.THEN)
        statement_node = self.statement()

        statement_node2 = NoOp()
        if self.current_token.type == TokenType.ELSE:
            self.eat(TokenType.ELSE)
            statement_node2 = self.statement()

        return Branch(cond_node, statement_node, statement_node2)

    def statement(self) -> IAST:
        """
        statement:
            compound_statement |
            assignment_statement |
            proccall_statement |
            if_statement |
            empty
        """
        tokty = self.current_token.type

        if tokty == TokenType.BEGIN:
            node = self.compound_statement()
        elif tokty == TokenType.ID:
            ahead_tok = self.peek()

            self._assert(
                cond=any_of(
                    [TokenType.ASSIGN, TokenType.LPAR],
                    lambda tty: ahead_tok.type == tty
                ),
                errcode=ErrorCode.UNEXPECTED_TOKEN,
                token=ahead_tok
            )

            node = Eof()
            if ahead_tok.type == TokenType.ASSIGN:
                node = self.assignment_statement()
            elif ahead_tok.type == TokenType.LPAR:
                node = self.proccall_statement()
        elif tokty == TokenType.IF:
            node = self.if_statement()
        else:
            node = self.empty()

        return node

    def statement_list(self) -> List[IAST]:
        """statement_list: statement (SEMI statement)*"""
        statements = []
        statements.append(self.statement())

        while self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            statements.append(self.statement())

        self._assert(
            cond=self.current_token.type != TokenType.ID,
            errcode=ErrorCode.UNEXPECTED_TOKEN,
            token=self.current_token,
            msg=f"Expected {TokenType.SEMI}"
        )

        return statements

    def compound_statement(self) -> Compound:
        """compound_statement: BEGIN statement_list END"""
        self.eat(TokenType.BEGIN)
        nodes = self.statement_list()
        self.eat(TokenType.END)
        return Compound(nodes)

    def type_spec(self) -> Type:
        """type_spec: INTEGER | REAL | BOOLEAN"""
        ty = None
        curtok = self.current_token
        if curtok.type == TokenType.INTEGER:
            ty = Type(cast(IntegerTok, curtok))
            self.eat(TokenType.INTEGER)
        elif curtok.type == TokenType.REAL:
            ty = Type(cast(RealTok, curtok))
            self.eat(TokenType.REAL)
        elif curtok.type == TokenType.BOOLEAN:
            ty = Type(cast(BooleanTok, curtok))
            self.eat(TokenType.BOOLEAN)
        else:  # error message to user
            self.eat(
                [
                    TokenType.INTEGER,
                    TokenType.REAL,
                    TokenType.BOOLEAN
                ]
            )

        return cast(Type, ty)

    def variable_declaration(self) -> List[VarDecl]:
        """
        variable_declaration: variable (COMMA variable)* COLON type_spec
        """
        var_nodes = [self.variable()]
        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            var_nodes.append(self.variable())
        self.eat(TokenType.COLON)
        ty_node = self.type_spec()
        return [
            VarDecl(var_node, Type(ty_node)) \
            for var_node in var_nodes \
            ]

    def formal_parameters(self) -> List[Param]:
        """
        formal_parameters: variable (COMMA variable)* COLON type_spec
        """
        param_nodes = [self.variable()]
        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            param_nodes.append(self.variable())
        self.eat(TokenType.COLON)
        ty_node = self.type_spec()
        return [
            Param(param_node, Type(ty_node)) \
            for param_node in param_nodes \
            ]

    def formal_parameter_list(self) -> List[Param]:
        """
        formal_parameter_list: formal_parameters (SEMI formal_parameters)*
        """
        param_nodes = self.formal_parameters()
        while self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            param_nodes += self.formal_parameters()
        return param_nodes

    def procedure_declaration(self):
        """
        procedure_declaration:
            PROCEDURE variable (LPAR formal_parameter_list RPAR)? SEMI
            block SEMI
        """
        self.eat(TokenType.PROCEDURE)
        var_n: Var = self.variable()
        proc_name: str = var_n.value

        params: List[Param] = []
        if self.current_token.type == TokenType.LPAR:
            self.eat(TokenType.LPAR)
            params = self.formal_parameter_list()
            self.eat(TokenType.RPAR)

        self.eat(TokenType.SEMI)
        block_n: Block = self.block()
        self.eat(TokenType.SEMI)

        return ProcDecl(proc_name, params, block_n)

    def declarations(self) -> List[IDecl]:
        """
        declarations:
            (VAR (variable_declaration SEMI)+)*
            procedure_declaration*
        """
        declarations: List[IDecl] = []

        while self.current_token.type == TokenType.VAR:
            self.eat(TokenType.VAR)
            while self.current_token.type == TokenType.ID:
                declarations += self.variable_declaration()
                self.eat(TokenType.SEMI)

        while self.current_token.type == TokenType.PROCEDURE:
            declarations.append(self.procedure_declaration())

        return declarations

    def block(self) -> Block:
        """block: declarations compound_statement"""
        declaration_nodes = self.declarations()
        compound_statement_node = self.compound_statement()
        return Block(declaration_nodes, compound_statement_node)

    def program(self) -> Program:
        """program : PROGRAM variable SEMI block DOT"""
        self.eat(TokenType.PROGRAM)
        var_node = self.variable()
        prog_name = var_node.value
        self.eat(TokenType.SEMI)
        block_node = self.block()
        self.eat(TokenType.DOT)
        return Program(prog_name, block_node)

    def parse_expr(self) -> IAST:
        return self.root_expr()

    def parse_compound(self) -> Compound:
        return self.compound_statement()

    def parse(self) -> Program:
        prog = self.program()
        self.eat(TokenType.EOF)
        return prog


class Symbol:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.level: int = 0


class BuiltinTypeSymbol(Symbol):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}')"


class VarSymbol(Symbol):
    def __init__(self, name: str, ty: BuiltinTypeSymbol) -> None:
        super().__init__(name)
        self.type: BuiltinTypeSymbol = ty

    def __str__(self) -> str:
        return f"<{self.name}:{self.type}>"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}', type='{self.type}')"


class ProcSymbol(Symbol):
    def __init__(
            self,
            name: str,
            block: Block,
            params: List[VarSymbol] = None
    ) -> None:
        super().__init__(name)
        self.block: Block = block
        self.params: List[VarSymbol] = params if params is not None else []

    def __str__(self) -> str:
        return f"{type(self).__name__}(name={self.name}, params={self.params})"


class ScopedSymbolTable:
    def __init__(
            self,
            name: str,
            level: int,
            encl_scope: Optional['ScopedSymbolTable'] = None
    ) -> None:
        self._symbols: Dict[str, Symbol] = {}
        self.name: str = name
        self.level: int = level
        self.encl_scope: Optional[ScopedSymbolTable] = encl_scope
        self.logger: Logger = getLogger("scope")

    def __str__(self) -> str:
        encl_scope_name = self.encl_scope.name \
            if self.encl_scope is not None else str(None)
        return f"(" + \
               f"name: {self.name}, " + \
               f"level: {self.level}, " + \
               f"encl_scope: {encl_scope_name}, " + \
               f"symbols: {[str(sym) for sym in self._symbols.values()]}" + \
               ")"

    def __repr__(self) -> str:
        header = "Symbol Table Contents"
        header2 = "Scope (Scoped symbol table) contents"
        name_header = "Scope name"
        level_header = "Scope level"

        lines = [
            header,
            "=" * len(header),
            f"{name_header:<15}: {self.name}",
            f"{level_header:<15}: {self.level}",
            f"{header2}",
            "-" * len(header2),
        ]
        lines += [f"{name:7}: {sym!r}" for name, sym in self._symbols.items()]
        return "\n".join(lines)

    def __getitem__(self, name: str) -> Symbol:
        assert isinstance(name, str), f"name msut be a str"
        sym: Optional[Symbol] = self.lookup(name)
        assert sym is not None, f"{name} does not exist"
        return cast(Symbol, sym)

    def __iter__(self) -> Iterator[Symbol]:
        return iter(self._symbols.values())

    def __len__(self) -> int:
        return len(self._symbols)

    def lookup_this_scope(self, name: str) -> Optional[Symbol]:
        self.logger.info(f"Lookup: {name}. (Scope name: {self.name})")
        return self._symbols.get(name.lower())

    def lookup(self, name: str) -> Optional[Symbol]:
        sym = self.lookup_this_scope(name)
        if sym is None and self.encl_scope is not None:
            return self.encl_scope.lookup(name)
        return sym

    def lookup_level(self, name: str) -> Optional[int]:
        if self._symbols.get(name.lower()) is not None:
            return self.level
        elif self.encl_scope is not None:
            return self.encl_scope.lookup_level(name)
        else:
            return None

    def insert(self, sym: Symbol) -> None:
        sym.level = self.level
        self.logger.info(f"Insert: {sym.name}")
        self._symbols[sym.name.lower()] = sym


class INodeVisitor(ABC):
    def _gen_visit_method_name(self, node: IAST) -> str:
        method_name = '_visit_' + to_lower_snake_case(type(node).__name__)
        return method_name.lower()

    def visit(self, node: IAST) -> Union[int, float, bool, None]:
        method_name = self._gen_visit_method_name(node)

        def raise_visit_error(_: IAST) -> None:
            assert False, f"No {method_name} method"

        return getattr(self, method_name, raise_visit_error)(node)

    @abstractmethod
    def _visit_pos(self, node: Pos) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_neg(self, node: Neg) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_add(self, node: Add) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_sub(self, node: Sub) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_mul(self, node: Mul) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_int_div(self, node: IntDiv) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_float_div(self, node: FloatDiv) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_modulus(self, node: Modulus) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_logical_and(
            self, node: LogicalAnd) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_logical_or(
            self, node: LogicalOr) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_logical_not(
            self, node: LogicalNot) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_bitwise_not(
            self, node: BitwiseNot) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_bitwise_or(
            self, node: BitwiseOr) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_bitwise_xor(
            self, node: BitwiseXor) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_bitwise_and(
            self, node: BitwiseAnd) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_equal(self, node: Equal) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_not_equal(self, node: NotEqual) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_greater(self, node: Greater) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_greater_equal(
            self, node: GreaterEqual) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_less(self, node: Less) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_less_equal(
            self, node: LessEqual) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_left_shift(
            self, node: LeftShift) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_right_shift(
            self, node: RightShift) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_num(self, node: Num) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_bool(self, node: Bool) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_compound(self, node: Compound) -> None:
        pass

    @abstractmethod
    def _visit_no_op(self, node: NoOp) -> None:
        pass

    @abstractmethod
    def _visit_assign(self, node: Assign) -> None:
        pass

    @abstractmethod
    def _visit_var(self, node: Var) -> Union[int, float, bool, None]:
        pass

    @abstractmethod
    def _visit_program(self, node: Program) -> None:
        pass

    @abstractmethod
    def _visit_block(self, node: Block) -> None:
        pass

    @abstractmethod
    def _visit_var_decl(self, node: VarDecl) -> None:
        pass

    @abstractmethod
    def _visit_proc_decl(self, node: ProcDecl) -> None:
        pass

    @abstractmethod
    def _visit_proc_call(self, node: ProcCall) -> None:
        pass

    @abstractmethod
    def _visit_type(self, node: Type) -> None:
        pass

    @abstractmethod
    def _visit_branch(self, node: Branch) -> None:
        pass


class IDecoSrcBuilder(ABC):
    @property  # type:ignore
    @abstractmethod
    def value(self) -> str:
        pass

    def build_pre_visit(
            self, scope: Optional[ScopedSymbolTable], node: IAST) -> None:
        methname = f"_build_pre_visit_{type(node).__name__.lower()}"
        getattr(self, methname)(scope, node)

    def build_post_visit(
            self, scope: Optional[ScopedSymbolTable], node: IAST) -> None:
        methname = f"_build_post_visit_{type(node).__name__.lower()}"
        getattr(self, methname)(scope, node)

    def build_in_visit(
            self, scope: Optional[ScopedSymbolTable], node: IAST) -> None:
        methname = f"_build_in_visit_{type(node).__name__.lower()}"
        getattr(self, methname)(scope, node)

    @abstractmethod
    def _build_pre_visit_pos(
            self, scope: Optional[ScopedSymbolTable], node: Pos) -> None:
        pass

    @abstractmethod
    def _build_post_visit_pos(
            self, scope: Optional[ScopedSymbolTable], node: Pos) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_neg(
            self, scope: Optional[ScopedSymbolTable], node: Neg) -> None:
        pass

    @abstractmethod
    def _build_post_visit_neg(
            self, scope: Optional[ScopedSymbolTable], node: Neg) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_add(
            self, scope: Optional[ScopedSymbolTable], node: Add) -> None:
        pass

    @abstractmethod
    def _build_in_visit_add(
            self, scope: Optional[ScopedSymbolTable], node: Add) -> None:
        pass

    @abstractmethod
    def _build_post_visit_add(
            self, scope: Optional[ScopedSymbolTable], node: Add) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_sub(
            self, scope: Optional[ScopedSymbolTable], node: Sub) -> None:
        pass

    @abstractmethod
    def _build_in_visit_sub(
            self, scope: Optional[ScopedSymbolTable], node: Sub) -> None:
        pass

    @abstractmethod
    def _build_post_visit_sub(
            self, scope: Optional[ScopedSymbolTable], node: Sub) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_mul(
            self, scope: Optional[ScopedSymbolTable], node: Mul) -> None:
        pass

    @abstractmethod
    def _build_in_visit_mul(
            self, scope: Optional[ScopedSymbolTable], node: Mul) -> None:
        pass

    @abstractmethod
    def _build_post_visit_mul(
            self, scope: Optional[ScopedSymbolTable], node: Mul) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_intdiv(
            self, scope: Optional[ScopedSymbolTable], node: IntDiv) -> None:
        pass

    @abstractmethod
    def _build_in_visit_intdiv(
            self, scope: Optional[ScopedSymbolTable], node: IntDiv) -> None:
        pass

    @abstractmethod
    def _build_post_visit_intdiv(
            self, scope: Optional[ScopedSymbolTable], node: IntDiv) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_floatdiv(
            self, scope: Optional[ScopedSymbolTable], node: FloatDiv) -> None:
        pass

    @abstractmethod
    def _build_in_visit_floatdiv(
            self, scope: Optional[ScopedSymbolTable], node: FloatDiv) -> None:
        pass

    @abstractmethod
    def _build_post_visit_floatdiv(
            self, scope: Optional[ScopedSymbolTable], node: FloatDiv) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_num(
            self, scope: Optional[ScopedSymbolTable], node: Num) -> None:
        pass

    @abstractmethod
    def _build_post_visit_num(
            self, scope: Optional[ScopedSymbolTable], node: Num) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_compound(
            self, scope: Optional[ScopedSymbolTable], node: Compound) -> None:
        pass

    @abstractmethod
    def _build_post_visit_compound(
            self, scope: Optional[ScopedSymbolTable], node: Compound) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_noop(
            self, scope: Optional[ScopedSymbolTable], node: NoOp) -> None:
        pass

    @abstractmethod
    def _build_post_visit_noop(
            self, scope: Optional[ScopedSymbolTable], node: NoOp) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_assign(
            self, scope: Optional[ScopedSymbolTable], node: Assign) -> None:
        pass

    @abstractmethod
    def _build_in_visit_assign(
            self, scope: Optional[ScopedSymbolTable], node: Assign) -> None:
        pass

    @abstractmethod
    def _build_post_visit_assign(
            self, scope: Optional[ScopedSymbolTable], node: Assign) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_var(
            self, scope: Optional[ScopedSymbolTable], node: Var) -> None:
        pass

    @abstractmethod
    def _build_post_visit_var(
            self, scope: Optional[ScopedSymbolTable], node: Var) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_program(
            self, scope: Optional[ScopedSymbolTable], node: Program) -> None:
        pass

    @abstractmethod
    def _build_in_visit_program(
            self, scope: Optional[ScopedSymbolTable], node: Program) -> None:
        pass

    @abstractmethod
    def _build_post_visit_program(
            self, scope: Optional[ScopedSymbolTable], node: Program) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_block(
            self, scope: Optional[ScopedSymbolTable], node: Block) -> None:
        pass

    @abstractmethod
    def _build_post_visit_block(
            self, scope: Optional[ScopedSymbolTable], node: Block) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_vardecl(
            self, scope: Optional[ScopedSymbolTable], node: VarDecl) -> None:
        pass

    @abstractmethod
    def _build_post_visit_vardecl(
            self, scope: Optional[ScopedSymbolTable], node: VarDecl) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_procdecl(
            self, scope: Optional[ScopedSymbolTable], node: ProcDecl) -> None:
        pass

    @abstractmethod
    def _build_in_visit_procdecl(
            self, scope: Optional[ScopedSymbolTable], node: ProcDecl) -> None:
        pass

    @abstractmethod
    def _build_post_visit_procdecl(
            self, scope: Optional[ScopedSymbolTable], node: ProcDecl) -> None:
        pass

    @abstractmethod
    def _build_pre_visit_type(
            self, scope: Optional[ScopedSymbolTable], node: Type) -> None:
        pass

    @abstractmethod
    def _build_post_visit_type(
            self, scope: Optional[ScopedSymbolTable], node: Type) -> None:
        pass


class DecoSrcBuilder(IDecoSrcBuilder):
    def __init__(self):
        self._value: str = ""
        self._expr: List[str] = []
        self._lvalue: List[str] = []
        self._statement: List[str] = []

    @property
    def value(self) -> str:
        return self._value

    def _indent(self, scope: Optional[ScopedSymbolTable]) -> str:
        level = scope.level if scope is not None else 0
        return " " * 3 * level

    def _writeinl(self, s: str) -> None:
        self._value += s

    def _write(self, scope: Optional[ScopedSymbolTable], s: str) -> None:
        self._writeinl(f"{self._indent(scope)}{s}")

    def _writeln(self, scope: Optional[ScopedSymbolTable], s: str) -> None:
        self._write(scope, f"{s}\n")

    def _build_pre_visit_pos(
            self, scope: Optional[ScopedSymbolTable], node: Pos) -> None:
        self._statement.append("+")

    def _build_post_visit_pos(
            self, scope: Optional[ScopedSymbolTable], node: Pos) -> None:
        pass

    def _build_pre_visit_neg(
            self, scope: Optional[ScopedSymbolTable], node: Neg) -> None:
        self._statement.append("-")

    def _build_post_visit_neg(
            self, scope: Optional[ScopedSymbolTable], node: Neg) -> None:
        pass

    def _build_pre_visit_add(
            self, scope: Optional[ScopedSymbolTable], node: Add) -> None:
        pass

    def _build_in_visit_add(
            self, scope: Optional[ScopedSymbolTable], node: Add) -> None:
        self._statement.append("+")

    def _build_post_visit_add(
            self, scope: Optional[ScopedSymbolTable], node: Add) -> None:
        pass

    def _build_pre_visit_sub(
            self, scope: Optional[ScopedSymbolTable], node: Sub) -> None:
        pass

    def _build_in_visit_sub(
            self, scope: Optional[ScopedSymbolTable], node: Sub) -> None:
        self._statement.append("-")

    def _build_post_visit_sub(
            self, scope: Optional[ScopedSymbolTable], node: Sub) -> None:
        pass

    def _build_pre_visit_mul(
            self, scope: Optional[ScopedSymbolTable], node: Mul) -> None:
        pass

    def _build_in_visit_mul(
            self, scope: Optional[ScopedSymbolTable], node: Mul) -> None:
        self._statement.append("*")

    def _build_post_visit_mul(
            self, scope: Optional[ScopedSymbolTable], node: Mul) -> None:
        pass

    def _build_pre_visit_intdiv(
            self, scope: Optional[ScopedSymbolTable], node: IntDiv) -> None:
        pass

    def _build_in_visit_intdiv(
            self, scope: Optional[ScopedSymbolTable], node: IntDiv) -> None:
        self._statement.append("DIV")

    def _build_post_visit_intdiv(
            self, scope: Optional[ScopedSymbolTable], node: IntDiv) -> None:
        pass

    def _build_pre_visit_floatdiv(
            self, scope: Optional[ScopedSymbolTable], node: FloatDiv) -> None:
        pass

    def _build_in_visit_floatdiv(
            self, scope: Optional[ScopedSymbolTable], node: FloatDiv) -> None:
        self._statement.append("/")

    def _build_post_visit_floatdiv(
            self, scope: Optional[ScopedSymbolTable], node: FloatDiv) -> None:
        pass

    def _build_pre_visit_num(
            self, scope: Optional[ScopedSymbolTable], node: Num) -> None:
        self._statement.append(str(node.value))

    def _build_post_visit_num(
            self, scope: Optional[ScopedSymbolTable], node: Num) -> None:
        pass

    def _build_pre_visit_compound(
            self, scope: Optional[ScopedSymbolTable], node: Compound) -> None:
        self._writeln(scope, f"begin")

    def _build_post_visit_compound(
            self, scope: Optional[ScopedSymbolTable], node: Compound) -> None:
        s = f"end;" if scope is not None and scope.level > 1 else f"end"
        self._write(scope, s)

    def _build_pre_visit_noop(
            self, scope: Optional[ScopedSymbolTable], node: NoOp) -> None:
        pass

    def _build_post_visit_noop(
            self, scope: Optional[ScopedSymbolTable], node: NoOp) -> None:
        pass

    def _build_pre_visit_assign(
            self, scope: Optional[ScopedSymbolTable], node: Assign) -> None:
        self._expr = []
        self._lvalue = []
        self._statement = []

    def _build_in_visit_assign(
            self, scope: Optional[ScopedSymbolTable], node: Assign) -> None:
        self._expr = self._statement
        self._statement = []

    def _build_post_visit_assign(
            self, scope: Optional[ScopedSymbolTable], node: Assign) -> None:
        self._lvalue = self._statement
        self._statement = []
        lvalue = " ".join(self._lvalue)
        expr = " ".join(self._expr)
        self._writeln(scope, f"{lvalue} := {expr}")

    def _build_pre_visit_var(
            self, scope: Optional[ScopedSymbolTable], node: Var) -> None:
        scope2 = cast(ScopedSymbolTable, scope)
        sym = scope2.lookup(node.value)
        var_sym = cast(VarSymbol, sym)
        ty_sym = var_sym.type

        var_name = var_sym.name
        var_lv = var_sym.level
        ty_name = ty_sym.name
        ty_lv = ty_sym.level

        self._statement.append(f"<{var_name}{var_lv}:{ty_name}{ty_lv}>")

    def _build_post_visit_var(
            self, scope: Optional[ScopedSymbolTable], node: Var) -> None:
        pass

    def _build_pre_visit_program(
            self, scope: Optional[ScopedSymbolTable], node: Program) -> None:
        pass

    def _build_in_visit_program(
            self, scope: Optional[ScopedSymbolTable], node: Program) -> None:
        scope2 = cast(ScopedSymbolTable, scope)
        sym = scope2.lookup(node.name)
        prog_sym = cast(ProcSymbol, sym)
        self._writeln(
            scope2.encl_scope,
            f"program {prog_sym.name}{prog_sym.level};"
        )

    def _build_post_visit_program(
            self, scope: Optional[ScopedSymbolTable], node: Program) -> None:
        self._writeinl(f".    {{END OF {node.name}}}")

    def _build_pre_visit_block(
            self, scope: Optional[ScopedSymbolTable], node: Block) -> None:
        pass

    def _build_post_visit_block(
            self, scope: Optional[ScopedSymbolTable], node: Block) -> None:
        pass

    def _build_pre_visit_vardecl(
            self, scope: Optional[ScopedSymbolTable], node: VarDecl) -> None:
        pass

    def _build_post_visit_vardecl(
            self, scope: Optional[ScopedSymbolTable], node: VarDecl) -> None:
        scope2 = cast(ScopedSymbolTable, scope)
        sym = scope2.lookup(node.var.value)
        var_sym = cast(VarSymbol, sym)
        self._writeln(
            scope,
            f"var {var_sym.name}{var_sym.level} : "
            f"{var_sym.type.name}{var_sym.type.level};"
        )

    def _build_pre_visit_procdecl(
            self, scope: Optional[ScopedSymbolTable], node: ProcDecl) -> None:
        pass

    def _build_in_visit_procdecl(
            self, scope: Optional[ScopedSymbolTable], node: ProcDecl) -> None:
        scope2 = cast(ScopedSymbolTable, scope)
        sym = scope2.lookup(node.name)
        proc_sym = cast(ProcSymbol, sym)
        args = ""
        if proc_sym.params:
            args += "("
            for param in proc_sym.params:
                args += \
                    f"{param.name}{param.level} : " \
                    f"{param.type.name}{param.type.level}"
            args += ")"
        self._writeln(
            scope2.encl_scope,
            f"procedure {proc_sym.name}{proc_sym.level}{args};"
        )

    def _build_post_visit_procdecl(
            self, scope: Optional[ScopedSymbolTable], node: ProcDecl) -> None:
        self._writeinl(f"    {{END OF {node.name}}}\n")

    def _build_pre_visit_type(
            self, scope: Optional[ScopedSymbolTable], node: Type) -> None:
        pass

    def _build_post_visit_type(
            self, scope: Optional[ScopedSymbolTable], node: Type) -> None:
        pass


class SemanticAnalyzer(INodeVisitor, ContextManager['SemanticAnalyzer']):
    def __init__(self, s2s: bool = False):
        self.s2s: bool = s2s
        self.logger = getLogger("scope")
        self._dsb: DecoSrcBuilder = DecoSrcBuilder()
        self._closed: bool = False

        self.current_scope: Optional[ScopedSymbolTable] = \
            ScopedSymbolTable("builtins", 0, None)
        self.logger.info(f"ENTER scope {self.current_scope.name}")
        self.current_scope.insert(BuiltinTypeSymbol("INTEGER"))
        self.current_scope.insert(BuiltinTypeSymbol("REAL"))
        self.current_scope.insert(BuiltinTypeSymbol("BOOLEAN"))

    def __enter__(self) -> 'SemanticAnalyzer':
        return self

    def __exit__(
            self,
            exc_ty: Optional[typing.Type[BaseException]],
            exc_val: Optional[BaseException],
            tb: Optional[TracebackType]
    ) -> None:
        self.close()

    def _assert(
            self,
            cond,
            errcode: ErrorCode,
            token: IToken,
            msg: Optional[str] = None
    ) -> None:
        assert_with(
            cond,
            SemanticError(
                error_code=errcode,
                token=token,
                appended_message=msg
            )
        )

    @property
    def safe_current_scope(self) -> ScopedSymbolTable:
        assert self.current_scope is not None
        return cast(ScopedSymbolTable, self.current_scope)

    def visit(self, node: IAST) -> Union[int, float, bool, None]:
        if self.s2s:
            self._dsb.build_pre_visit(self.current_scope, node)
        val = super().visit(node)
        if self.s2s:
            self._dsb.build_post_visit(self.current_scope, node)
        return val

    def _build_in_visit(self, node: IAST):
        if self.s2s:
            self._dsb.build_in_visit(self.current_scope, node)

    def _visit_bin_op(self, node: BinOp) -> None:
        self.visit(node.left)
        self._build_in_visit(node)
        self.visit(node.right)

    def _visit_un_op(self, node: UnOp) -> None:
        self.visit(node.right)

    def _visit_pos(self, node: Pos) -> None:
        self._visit_un_op(node)

    def _visit_neg(self, node: Neg) -> None:
        self._visit_un_op(node)

    def _visit_logical_not(self, node: LogicalNot) -> None:
        self._visit_un_op(node)

    def _visit_bitwise_not(self, node: BitwiseNot) -> None:
        self._visit_un_op(node)

    def _visit_add(self, node: Add) -> None:
        self._visit_bin_op(node)

    def _visit_sub(self, node: Sub) -> None:
        self._visit_bin_op(node)

    def _visit_mul(self, node: Mul) -> None:
        self._visit_bin_op(node)

    def _visit_int_div(self, node: IntDiv) -> None:
        self._visit_bin_op(node)

    def _visit_float_div(self, node: FloatDiv) -> None:
        self._visit_bin_op(node)

    def _visit_logical_and(self, node: LogicalAnd) -> None:
        self._visit_bin_op(node)

    def _visit_logical_or(self, node: LogicalOr) -> None:
        self._visit_bin_op(node)

    def _visit_bitwise_or(self, node: BitwiseOr) -> None:
        self._visit_bin_op(node)

    def _visit_bitwise_xor(self, node: BitwiseXor) -> None:
        self._visit_bin_op(node)

    def _visit_bitwise_and(self, node: BitwiseAnd) -> None:
        self._visit_bin_op(node)

    def _visit_modulus(self, node: Modulus) -> None:
        self._visit_bin_op(node)

    def _visit_equal(self, node: Equal) -> None:
        self._visit_bin_op(node)

    def _visit_not_equal(self, node: NotEqual) -> None:
        self._visit_bin_op(node)

    def _visit_greater(self, node: Greater) -> None:
        self._visit_bin_op(node)

    def _visit_greater_equal(self, node: GreaterEqual) -> None:
        self._visit_bin_op(node)

    def _visit_less(self, node: Less) -> None:
        self._visit_bin_op(node)

    def _visit_less_equal(self, node: LessEqual) -> None:
        self._visit_bin_op(node)

    def _visit_left_shift(self, node: LeftShift) -> None:
        self._visit_bin_op(node)

    def _visit_right_shift(self, node: RightShift) -> None:
        self._visit_bin_op(node)

    def _visit_num(self, node: Num) -> None:
        pass

    def _visit_bool(self, node: Bool) -> None:
        pass

    def _visit_compound(self, node: Compound) -> None:
        for n in node.kids:
            self.visit(n)

    def _visit_no_op(self, node: NoOp) -> None:
        return None

    def _visit_assign(self, node: Assign) -> None:
        self.visit(node.right)
        self._build_in_visit(node)
        self.visit(node.left)

    def _visit_var(self, node: Var) -> None:
        var_name = node.value
        self._assert(
            cond=self.safe_current_scope.lookup(var_name) is not None,
            errcode=ErrorCode.ID_NOT_FOUND,
            token=node.token,
            msg=f"{var_name} is not declared"
        )

    def _visit_program(self, node: Program) -> None:
        assert isinstance(self.current_scope, ScopedSymbolTable)

        proc_sym = ProcSymbol(
            name=node.name,
            block=node.block
        )
        self.current_scope.insert(proc_sym)
        node.proc_sym = proc_sym

        self.logger.info(f"ENTER scope {node.name}")
        global_scope = ScopedSymbolTable(
            name=node.name,
            level=proc_sym.level + 1,
            encl_scope=self.current_scope
        )
        self.current_scope = global_scope

        self._build_in_visit(node)
        self.visit(node.block)

        self.logger.info(global_scope)
        self.current_scope = self.current_scope.encl_scope
        self.logger.info(f"LEAVE scope {global_scope.name}")

    def _visit_block(self, node: Block) -> None:
        for n in node.kids:
            self.visit(n)

    def _visit_var_decl(self, node: VarDecl) -> None:
        ty = node.type
        type_name = ty.value
        type_sym = self.safe_current_scope.lookup(type_name)
        self._assert(
            cond=type_sym is not None,
            errcode=ErrorCode.TYPE_NOT_FOUND,
            token=ty.token
        )

        var = node.var
        var_name = var.value
        var_sym = VarSymbol(
            name=var_name,
            ty=cast(BuiltinTypeSymbol, type_sym)
        )
        self._assert(
            cond=self.safe_current_scope.lookup_this_scope(
                var_sym.name) is None,
            errcode=ErrorCode.DUPLICATE_ID,
            token=node.var.token,
        )

        self.safe_current_scope.insert(var_sym)

    def _visit_proc_decl(self, node: ProcDecl) -> None:
        proc_name = node.name
        proc_symbol = ProcSymbol(
            name=proc_name,
            block=node.block
        )
        self.safe_current_scope.insert(proc_symbol)

        scope_name = proc_name
        self.logger.info(f"ENTER scope {scope_name}")
        proc_scope = ScopedSymbolTable(
            name=scope_name,
            level=proc_symbol.level + 1,
            encl_scope=self.current_scope
        )
        self.current_scope = proc_scope

        for param in node.params:
            param_name = param.var.value
            param_ty = self.current_scope.lookup(param.type.value)

            assert param_ty is not None
            assert isinstance(param_ty, BuiltinTypeSymbol)

            param_sym = self.safe_current_scope.lookup_this_scope(param_name)

            self._assert(
                cond=param_sym is None,
                errcode=ErrorCode.DUPLICATE_ID,
                token=param.var.token
            )

            var_sym = VarSymbol(
                name=param_name,
                ty=cast(BuiltinTypeSymbol, param_ty)
            )
            proc_symbol.params.append(var_sym)
            self.current_scope.insert(var_sym)

        self._build_in_visit(node)
        self.visit(node.block)

        self.logger.info(proc_scope)
        self.current_scope = self.current_scope.encl_scope
        self.logger.info(f"LEAVE scope {proc_scope.name}")

    def _visit_proc_call(self, node: ProcCall) -> None:
        if node.proc_name.lower() != 'writeln':
            proc_sym = self.safe_current_scope.lookup(node.proc_name)

            self._assert(
                cond=proc_sym is not None,
                errcode=ErrorCode.ID_NOT_FOUND,
                token=node.token
            )

            self._assert(
                cond=isinstance(proc_sym, ProcSymbol),
                errcode=ErrorCode.PROC_NOT_FOUND,
                token=node.token,
                msg=f"{proc_sym} is not a ProcSymbol"
            )

            proc_sym2 = cast(ProcSymbol, proc_sym)
            exp_num_args = len(proc_sym2.params)
            act_num_args = len(node.actual_params)
            self._assert(
                cond=act_num_args == exp_num_args,
                errcode=ErrorCode.BAD_PARAMS,
                token=node.token,
                msg=f"Expected {exp_num_args} args, got {act_num_args} args"
            )

            node.proc_sym = proc_sym2

        for child in node.kids:
            self.visit(child)

    def _visit_type(self, node: Type) -> None:
        pass

    def _visit_branch(self, node: Branch) -> None:
        for node in node.kids:
            self.visit(node)

    def analyze(self, node: IAST) -> None:
        assert not self._closed
        self._dsb = DecoSrcBuilder()
        self.visit(node)

    def deco_src(self) -> str:
        return self._dsb.value

    def close(self):
        self.logger.info(self.current_scope)
        scope = self.current_scope
        self.current_scope = self.current_scope.encl_scope
        self.logger.info(f"LEAVE scope {scope.name}")
        self._closed = True


class FrameType(Enum):
    PROGRAM = 'PROGRAM'
    PROCEDURE = 'PROCEDURE'


class Frame(MutableMapping[str, Union[int, float, bool]]):
    def __init__(
            self,
            name: str,
            ty: FrameType,
            nesting_lv: int,
            members: Optional[Mapping[str, Union[int, float, bool]]] = None
    ) -> None:
        self.name: str = name
        self.ty: FrameType = ty
        assert nesting_lv >= 0
        self.nesting_lv: int = nesting_lv
        self._members: Dict[str, Union[int, float, bool]] = \
            dict(members) if members else {}

    def __getitem__(self, name) -> Union[int, float, bool]:
        return self._members[name]

    def __setitem__(self, name, value) -> None:
        self._members[name] = value

    def __delitem__(self, name) -> None:
        del self._members[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __str__(self) -> str:
        buf = StringIO()
        print(f"{type(self).__name__}(", file=buf)
        print(f"  name: {self.name},", file=buf)
        print(f"  ty: {self.ty},", file=buf)
        print(f"  nesting_lv: {self.nesting_lv},", file=buf)
        print("  members: {", file=buf)
        for i, (k, v) in enumerate(self._members.items()):
            entry = f"    {k}: {v}"
            if i != len(self._members) - 1:
                entry += ","
            print(entry, file=buf)
        print("  }", file=buf)
        print(")", end="", file=buf)
        return buf.getvalue()

    def __repr__(self) -> str:
        return flattened_str(str(self))


class RuntimeStack:
    def __init__(
            self,
            frames: Optional[Iterable[Frame]] = None,
            free_frames=False
    ) -> None:
        self._frames: List[Frame] = [] if frames is None else list(frames)
        self._free_frames: Optional[List[Frame]] = [] if free_frames else None
        self.logger = getLogger("stack")

    def _assert_new_frame(self, frame) -> None:
        assert frame.nesting_lv == self.next_level(), \
            f"got {frame.nesting_lv}, expected {self.next_level()}"
        if len(self._frames) > 0:
            assert frame.ty == FrameType.PROCEDURE
        else:
            assert frame.ty == FrameType.PROGRAM

    def next_frame_type(self) -> FrameType:
        return FrameType.PROGRAM \
            if len(self._frames) == 0 \
            else FrameType.PROCEDURE

    def next_level(self) -> int:
        return 1 if len(self._frames) == 0 else self.peek().nesting_lv + 1

    def emplace_frame(
            self,
            proc_sym: ProcSymbol,
            members: Optional[Mapping[str, Union[int, float, bool]]] = None
    ) -> None:
        self.push(
            Frame(
                name=proc_sym.name,
                ty=self.next_frame_type(),
                nesting_lv=proc_sym.level + 1,
                members=members
            )
        )

    def push(self, frame: Frame) -> None:
        self._assert_new_frame(frame)
        self.logger.info(f"ENTER: {frame.ty.value} {frame.name}")
        self._frames.append(frame)
        self.logger.info(self)

    def pop(self) -> Frame:
        freed_soon = self.peek()
        self.logger.info(f"LEAVE: {freed_soon.ty.value} {freed_soon.name}")
        self.logger.info(self)
        freed = self._frames.pop()
        if self._free_frames is not None:
            self._free_frames.append(freed)
        return freed

    def peek(self) -> Frame:
        return self._frames[-1]

    def __str__(self) -> str:
        buf = StringIO()
        print(f"{type(self).__name__}(", file=buf)
        print("  frames: [", file=buf)
        for i, frame in enumerate(reversed(self._frames)):
            frame_s = re.sub(r"^|\n", r"\g<0>    ", str(frame))
            if i != len(self._frames) - 1:
                frame_s += ","
            print(frame_s, file=buf)
        print("  ]", file=buf)
        print(")", file=buf, end="")
        return buf.getvalue()

    def __repr__(self) -> str:
        return flattened_str(str(self))


class Interpreter(INodeVisitor):
    def __init__(self, free_frames=False):
        self.rts: RuntimeStack = RuntimeStack(free_frames=free_frames)
        self.logger = getLogger("stack")

    def _visit_pos(self, node: Pos) -> Union[int, float]:
        return +cast(
            [int, float],
            self.visit(node.right),
            lambda ty, val, msg: InterpreterError(
                error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                token=node.token,
                appended_message=msg
            )
        )

    def _visit_neg(self, node: Neg) -> Union[int, float]:
        return -cast(
            [int, float],
            self.visit(node.right),
            lambda ty, val, msg: InterpreterError(
                error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                token=node.token,
                appended_message=msg
            )
        )

    def _visit_logical_not(
            self, node: LogicalNot) -> Union[int, float, bool, None]:
        return not cast(
            [int, float, bool],
            self.visit(node.right),
            lambda ty, val, msg: InterpreterError(
                error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                token=node.token,
                appended_message=msg
            )
        )

    def _visit_bitwise_not(
            self, node: BitwiseNot) -> Union[int, float, bool, None]:
        return ~cast(
            [int, bool],
            self.visit(node.right),
            lambda ty, val, msg: InterpreterError(
                error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                token=node.token,
                appended_message=msg
            )
        )

    def _visit_add(self, node: Add) -> Union[int, float]:
        return \
            cast(
                [int, float],
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            + \
            cast(
                [int, float],
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_sub(self, node: Sub) -> Union[int, float]:
        return \
            cast(
                [int, float],
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            - \
            cast(
                [int, float],
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_mul(self, node: Mul) -> Union[int, float]:
        return \
            cast(
                [int, float],
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            * \
            cast(
                [int, float],
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_int_div(self, node: IntDiv) -> int:
        return \
            cast(
                [int, float],
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            // \
            cast(
                [int, float],
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_float_div(self, node: FloatDiv) -> float:
        return \
            cast(
                [int, float],
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            / \
            cast(
                [int, float],
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_logical_and(self, node: LogicalAnd) -> bool:
        return bool(self.visit(node.left) and self.visit(node.right))

    def _visit_logical_or(self, node: LogicalOr) -> bool:
        return bool(self.visit(node.left) or self.visit(node.right))

    def _visit_bitwise_or(self, node: BitwiseOr) -> int:
        return \
            cast(
                int,
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            | \
            cast(
                int,
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_bitwise_xor(self, node: BitwiseXor) -> int:
        return \
            cast(
                int,
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            ^ \
            cast(
                int,
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_bitwise_and(self, node: BitwiseAnd) -> int:
        return \
            cast(
                int,
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            & \
            cast(
                int,
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_modulus(self, node: Modulus) -> int:
        int1 = cast(
            int,
            self.visit(node.left),
            lambda ty, val, msg: InterpreterError(
                error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                token=node.token,
                appended_message=msg
            )
        )
        int2 = cast(
            int,
            self.visit(node.right),
            lambda ty, val, msg: InterpreterError(
                error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                token=node.token,
                appended_message=msg
            )
        )
        return int(Decimal(int1) % Decimal(int2))

    def _visit_equal(self, node: Equal) -> bool:
        return \
            cast(
                [int, float, bool],
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            == \
            cast(
                [int, float, bool],
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_not_equal(self, node: NotEqual) -> bool:
        return \
            cast(
                [int, float, bool],
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            != \
            cast(
                [int, float, bool],
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_greater(self, node: Greater) -> bool:
        return \
            cast(
                [int, float],
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            > \
            cast(
                [int, float],
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_greater_equal(self, node: GreaterEqual) -> bool:
        return \
            cast(
                [int, float],
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            >= \
            cast(
                [int, float],
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_less(self, node: Less) -> bool:
        return \
            cast(
                [int, float],
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            < \
            cast(
                [int, float],
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_less_equal(self, node: LessEqual) -> bool:
        return \
            cast(
                [int, float], self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            <= \
            cast(
                [int, float], self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_left_shift(self, node: LeftShift) -> int:
        return \
            cast(
                int,
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            << \
            cast(
                int,
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_right_shift(self, node: RightShift) -> int:
        return \
            cast(
                int,
                self.visit(node.left),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            ) \
            >> \
            cast(
                int,
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    def _visit_num(self, node: Num) -> Union[int, float]:
        return node.value

    def _visit_bool(self, node: Bool) -> bool:
        return node.value

    def _visit_compound(self, node: Compound) -> None:
        for child in node.kids:
            self.visit(child)

    def _visit_no_op(self, node: NoOp) -> None:
        pass

    def _visit_assign(self, node: Assign) -> None:
        self.rts.peek()[node.left.value] = \
            cast(
                [int, float, bool],
                self.visit(node.right),
                lambda ty, val, msg: InterpreterError(
                    error_code=ErrorCode.UNEXPECTED_OPERAND_TYPE,
                    token=node.token,
                    appended_message=msg
                )
            )

    # TODO: lookup in other frames
    def _visit_var(self, node: Var) -> Union[int, float, bool]:
        name = node.value
        val = self.rts.peek().get(name.lower())
        assert val is not None
        return val

    def _visit_program(self, node: Program) -> None:
        prog_sym = cast(ProcSymbol, node.proc_sym)
        self.rts.emplace_frame(prog_sym)
        self.visit(prog_sym.block)
        self.rts.pop()

    def _visit_block(self, node: Block) -> None:
        for ast in node.kids:
            self.visit(ast)

    def _visit_var_decl(self, node: VarDecl) -> None:
        pass

    def _visit_proc_decl(self, node: ProcDecl) -> None:
        pass

    def _visit_proc_call(self, node: ProcCall) -> None:
        if node.proc_name.lower() == 'writeln':
            data = "".join(
                [str(self.visit(param)) for param in node.actual_params])
            print(data)
        else:
            proc_sym = cast(ProcSymbol, node.proc_sym)
            self.rts.emplace_frame(proc_sym, members={
                param.name: cast([int, float, bool], self.visit(arg))
                for param, arg in zip(proc_sym.params, node.actual_params)
            })
            self.visit(proc_sym.block)
            self.rts.pop()

    def _visit_type(self, node: Type) -> None:
        pass

    def _visit_branch(self, node: Branch) -> None:
        if self.visit(node.cond):
            self.visit(node.if_blk)
        else:
            self.visit(node.else_blk)

    def interpret_compound(self, ast: Compound) -> None:
        proc_sym = ProcSymbol("global", Block([], ast))
        self.rts.emplace_frame(proc_sym)
        self.visit(proc_sym.block)
        self.rts.pop()

    def interpret(self, ast: IAST) -> Union[int, float, None]:
        return self.visit(ast)


def main() -> None:
    argparser = argparse.ArgumentParser(description="simple pascal interpreter")
    argparser.add_argument(
        "--verbose-stack",
        action="store_true",
        help="verbose output related to runtime stack events"
    )
    argparser.add_argument(
        "--verbose-scope",
        action="store_true",
        help="verbose output related to semantic analysis"
    )
    argparser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="enables all verbose flags (--stack-verbose, --scope-verbose)"
    )
    argparser.add_argument(
        "-s", "--src-to-src",
        action="store_true",
        help="translate source in FILE to decorated source and print"
    )
    argparser.add_argument("FILE", help="pascal source file")
    args = argparser.parse_args()

    if args.verbose:
        args.verbose_stack = True
        args.verbose_scope = True

    setup_logging(
        verbose_stack=args.verbose_stack,
        verbose_scope=args.verbose_scope
    )

    with open(args.FILE) as pascal_file:
        text = pascal_file.read()

    lexer: Lexer = Lexer(text)
    parser: Parser = Parser(lexer)
    ast: IAST = parser.parse()

    kwargs = {}
    if args.src_to_src:
        kwargs["s2s"] = True

    with SemanticAnalyzer(**kwargs) as lyz:  # type:SemanticAnalyzer
        lyz.analyze(ast)

    if args.src_to_src:
        print(lyz.deco_src())
    else:
        interpreter: Interpreter = Interpreter()
        interpreter.interpret(ast)


logging.disable(logging.CRITICAL)
if __name__ == '__main__':
    main()
