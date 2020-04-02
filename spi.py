from enum import Enum
from typing import \
    Any, \
    Callable, \
    Optional, \
    Union, \
    List, \
    Dict, \
    Mapping, \
    NamedTuple, \
    Iterator
from abc import ABC, abstractmethod
from anytree import RenderTree # type:ignore
from anytree import \
    PostOrderIter, \
    NodeMixin
from re import Pattern, RegexFlag

import re

class TypeIdValue(NamedTuple):
    pat: str
    re: bool = False
    type: Callable[[str], Any] = str


class TypeId(Enum):
    INT = TypeIdValue(pat=r"\d+", re=True, type=int)
    ADD = TypeIdValue(pat='+')
    SUB = TypeIdValue(pat='-')
    MUL = TypeIdValue(pat='*')
    DIV = TypeIdValue(pat='/')
    LPAR = TypeIdValue(pat='(')
    RPAR = TypeIdValue(pat=')')
    EOF = TypeIdValue(pat=r"$", re=True)
    BEGIN = TypeIdValue(pat="BEGIN")
    END = TypeIdValue(pat="END")
    DOT = TypeIdValue(pat=".")
    ID = TypeIdValue(pat=r"[a-zA-Z]\w*", re=True)
    ASSIGN = TypeIdValue(pat=":=")
    SEMI = TypeIdValue(pat=";")

    def __repr__(self) -> str:
        return str(self)
 
    @classmethod
    def members(cls) -> Mapping[str, 'TypeId']:
        return cls.__members__

    @staticmethod
    def pattern(ident: 'TypeId') -> str:
        pat = ident.value.pat or ''
        return pat if ident.value.re else re.escape(pat)


class Token:
    def __init__(self, type: TypeId, value: Union[str, int, None]) -> None:
        self.type = type
        self.value = value

    def __str__(self) -> str:
        """String representation of the class instance.

        Examples:
            Token(INTEGER, 3)
            Token(ADD, '+')
            Token(MUL, '*')
        """
        return f"Token({self.type}, {self.value})"

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return bool(self.value)

    def __eq__(self, other) -> bool:
        return self.type == other.type and self.value == other.value

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class Lexer:
    class TypeIdInfo(NamedTuple):
        typeid: TypeId
        pattern: str
        token: Token=Token(TypeId.EOF, None)

    RES_KW: List[TypeId] = [
        TypeId.BEGIN,
        TypeId.END
    ]

    __RES_KW_TO_TID_INFO: Dict[str, TypeIdInfo] = {}
    __TOKEN_NAME_TO_TID_INFO: Dict[str, TypeIdInfo] = {}

    @classmethod
    def _RES_KW_TO_TID_INFO(cls) -> Dict[str, TypeIdInfo]:
        if not cls.__RES_KW_TO_TID_INFO:
            cls.__RES_KW_TO_TID_INFO = \
                { 
                    name: 
                    cls.TypeIdInfo(
                        typeid=tid,
                        pattern=TypeId.pattern(tid),
                        token=Token(tid, TypeId.pattern(tid))
                    )
                    for name, tid in TypeId.members().items() 
                    if tid in cls.RES_KW
                }
        return cls.__RES_KW_TO_TID_INFO

    @classmethod
    def _TOKEN_NAME_TO_TID_INFO(cls) -> Dict[str, TypeIdInfo]:
        if not cls.__TOKEN_NAME_TO_TID_INFO:
            cls.__TOKEN_NAME_TO_TID_INFO = \
                { 
                    name: 
                    cls.TypeIdInfo(
                        tid,
                        TypeId.pattern(tid)
                    )
                    for name, tid in TypeId.members().items()
                }
            cls.__TOKEN_NAME_TO_TID_INFO.update(cls._RES_KW_TO_TID_INFO())
        return cls.__TOKEN_NAME_TO_TID_INFO

    def __init__(self, text: str) -> None:
        self._token_gen: Iterator[Token] = self._iter_tokens()
        self._text: str = text

    def _iter_tokens(self) -> Iterator[Token]:
        token_spec = Lexer._TOKEN_NAME_TO_TID_INFO()
        token_pats = [
            rf"(?P<{name}>{tid_info.pattern})" 
            for name, tid_info in token_spec.items() 
        ]
        token_pat = "|".join(token_pats)

        for m in re.finditer(token_pat, self._text):
            name = m.lastgroup if m.lastgroup else ''
            tid = token_spec[name].typeid
            if token_spec[name].token:
                yield token_spec[name].token
            else:
                yield Token(tid, tid.value.type(m[name]))

        yield Token(TypeId.EOF, None)

    def get_next_token(self) -> Token:
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """

        return next(self._token_gen)


class AST(ABC, NodeMixin):
    @property
    @abstractmethod
    def token(self) -> Token:
        pass

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.token})"

    def __repr__(self) -> str:
        return str(self)


class BinOp(AST):
    def __init__(self, left: AST, right: AST, optok: Token) -> None:
        self.left = left
        self._token = self.op = optok
        self.right = right
        self.children = [self.left, self.right]

    @property
    def token(self) -> Token:
        return self._token


class Add(BinOp):
    def __init__(
        self, 
        left: AST, 
        right: AST, 
        opchar: str=TypeId.ADD.value.pat
    ) -> None:
        super().__init__(left, right, Token(TypeId.ADD, opchar))

class Sub(BinOp):
     def __init__(
        self, 
        left: AST, 
        right: AST, 
        opchar: str=TypeId.SUB.value.pat
    ) -> None:
        super().__init__(left, right, Token(TypeId.SUB, opchar))   


class Mul(BinOp):
     def __init__(
        self, 
        left: AST, 
        right: AST, 
        opchar: str=TypeId.MUL.value.pat
    ) -> None:
        super().__init__(left, right, Token(TypeId.MUL, opchar))


class Div(BinOp):
     def __init__(
        self, 
        left: AST, 
        right: AST, 
        opchar: str=TypeId.DIV.value.pat
    ) -> None:
        super().__init__(left, right, Token(TypeId.DIV, opchar))


class UnOp(AST):
    def __init__(self, right: AST, optok: Token ) -> None:
        self.right = right
        self.children = [self.right]
        self._token = optok

    @property
    def token(self):
        return self._token

class Pos(UnOp):
    def __init__(
        self, 
        right: AST,
        opchar: str=TypeId.ADD.value.pat
    ) -> None:
        super().__init__(right, Token(TypeId.ADD, opchar))


class Neg(UnOp):
    def __init__(
        self,
        right: AST,
        opchar: str=TypeId.SUB.value.pat
    ) -> None:
        super().__init__(right, Token(TypeId.SUB, opchar))


class Num(AST):
    def __init__(self, numtok: Token) -> None:
        self._token = numtok
        self.value = numtok.value

    @property
    def token(self):
        return self._token


class Eof(AST):
    def __init__(self) -> None:
        self._token = Token(TypeId.EOF, None)

    @property
    def token(self) -> Token:
        return self._token
    
    
class Parser:
    def __init__(self, lexer: Lexer) -> None:
        self.lexer = lexer
        # set current token to the first token taken from the input
        self.current_token = self.lexer.get_next_token()

    def error(self, msg: str) -> None:
        raise RuntimeError(f"Invalid syntax: {msg}")

    def eat(self, token_type: TypeId) -> None:
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(
                f"was expecting {token_type}, got {self.current_token.type}")

    def factor(self, lpar_b: bool=False) -> AST:
        """factor : (ADD | SUB) factor | INTEGER | LPAREN expr RPAREN"""
        token: Token = self.current_token

        if token.type == TypeId.ADD:
            self.eat(TypeId.ADD) 
            return Pos(self.factor(lpar_b))
        elif token.type == TypeId.SUB:
            self.eat(TypeId.SUB)
            return Neg(self.factor(lpar_b))
        elif token.type == TypeId.INT:
            self.eat(TypeId.INT)
            if self.current_token.type == TypeId.LPAR:
                self.error(f"found {self.current_token}")
            elif self.current_token.type == TypeId.RPAR and not lpar_b:
                self.error(f"{TypeId.RPAR} with no matching {TypeId.LPAR}")
            return Num(token)
        elif token.type == TypeId.LPAR:
            self.eat(TypeId.LPAR)
            node: AST = self.expr(True)
            self.eat(TypeId.RPAR)
            return node

        self.error(str(token))
        return Eof()

    def term(self, lpar_b: bool=False) -> AST:
        """term : factor ((MUL | DIV) factor)*"""
        node: AST = self.factor(lpar_b)

        while True:
            token: Token = self.current_token
            if token.type == TypeId.MUL:
                self.eat(TypeId.MUL)
                node = Mul(node, self.factor(lpar_b))
            elif token.type == TypeId.DIV:
                self.eat(TypeId.DIV)
                node = Div(node, self.factor(lpar_b))
            else:
                break

        return node

    def expr(self, lpar_b: bool=False) -> AST:
        """
        expr   : term ((ADD | SUB) term)*
        term   : factor ((MUL | DIV) factor)*
        factor : (POS | NEG)* factor | INTEGER | LPAREN expr RPAREN
        """
        node: AST = self.term(lpar_b)

        while True:
            token: Token = self.current_token
            if token.type == TypeId.ADD:
                self.eat(TypeId.ADD)
                node = Add(node, self.term(lpar_b))
            elif token.type == TypeId.SUB:
                self.eat(TypeId.SUB)
                node = Sub(node, self.term(lpar_b))
            else:
                break

        return node

    def parse(self) -> AST:
        return self.expr()


class NodeVisitor:
    def _gen_visit_method_name(self, node: AST) -> str:
        method_name = 'visit_' + type(node).__name__
        return method_name.lower()

    def visit(self, node: AST) -> int:
        method_name = self._gen_visit_method_name(node)
        return getattr(self, method_name, self.raise_visit_error)(node)

    def raise_visit_error(self, node: AST) -> None:
        method_name = self._gen_visit_method_name(node)
        raise RuntimeError(f"No {method_name} method")


class Interpreter(NodeVisitor):
    def __init__(self, text: str):
        self._text = text.strip()

    def visit_pos(self, node: AST) -> int:
        return +self.visit(node.right)

    def visit_neg(self, node: AST) -> int:
        return -self.visit(node.right)

    def visit_add(self, node: AST) -> int:
        return self.visit(node.left) + self.visit(node.right)

    def visit_sub(self, node: AST) -> int:
        return self.visit(node.left) - self.visit(node.right)

    def visit_mul(self, node: AST) -> int:
        return self.visit(node.left) * self.visit(node.right)

    def visit_div(self, node: AST) -> int:
        return int(self.visit(node.left) / self.visit(node.right))

    def visit_num(self, node: AST) -> int:
        return node.value

    def interpret(self) -> Union[int, str]:
        if not self._text:
            return ""
        lexer: Lexer = Lexer(self._text)
        parser: Parser = Parser(lexer)
        ast: AST = parser.parse()
        return self.visit(ast)


def main() -> None:
    while True:
        try:
            text: str = input('spi> ')
        except EOFError:
            break
        if not text:
            continue

        interpreter: Interpreter = Interpreter(text)
        result: Union[int, str] = interpreter.interpret()
        print(result)


if __name__ == '__main__':
    main()
