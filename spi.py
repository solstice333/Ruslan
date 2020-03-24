from enum import Enum, auto
from typing import Optional, Union, List, Dict, overload
from abc import ABC, abstractmethod
from anytree import PostOrderIter, RenderTree, NodeMixin # type:ignore


class TypeId(Enum):
    INT = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    LPAR = auto()
    RPAR = auto()
    EOF = auto()

    def __repr__(self) -> str:
        return str(self)
 
    @classmethod
    def operators(cls) -> List['TypeId']:
        return [
            cls.ADD, 
            cls.SUB, 
            cls.MUL, 
            cls.DIV
        ]

    @classmethod
    def parens(cls) -> List['TypeId']:
        return [
            cls.LPAR,
            cls.RPAR
        ]


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


class Lexer:
    _CHAR_TO_TYPEID = { 
        '+': TypeId.ADD,
        '-': TypeId.SUB,
        '*': TypeId.MUL,
        '/': TypeId.DIV,
        '(': TypeId.LPAR,
        ')': TypeId.RPAR
    }

    def __init__(self, text: str) -> None:
        self.text: str = text
        self.pos: int = 0
        self.current_char: Optional[str] = self.text[self.pos]

    def error(self) -> None:
        raise RuntimeError('Invalid character')

    def advance(self) -> None:
        """Advance the `pos` pointer and set the `current_char` variable."""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self) -> None:
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def integer(self) -> int:
        """Return a (multidigit) integer consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def get_next_token(self) -> Token:
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            elif self.current_char.isdigit():
                return Token(TypeId.INT, self.integer())

            elif self.current_char in Lexer._CHAR_TO_TYPEID.keys():
                tok = Token(
                    Lexer._CHAR_TO_TYPEID[self.current_char], 
                    self.current_char
                )
                self.advance()
                return tok
            
            self.error()

        return Token(TypeId.EOF, None)


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
        opchar: str='+'
    ) -> None:
        super().__init__(left, right, Token(TypeId.ADD, opchar))

class Sub(BinOp):
     def __init__(
        self, 
        left: AST, 
        right: AST, 
        opchar: str='-'
    ) -> None:
        super().__init__(left, right, Token(TypeId.SUB, opchar))   


class Mul(BinOp):
     def __init__(
        self, 
        left: AST, 
        right: AST, 
        opchar: str='*'
    ) -> None:
        super().__init__(left, right, Token(TypeId.MUL, opchar))


class Div(BinOp):
     def __init__(
        self, 
        left: AST, 
        right: AST, 
        opchar: str='/'
    ) -> None:
        super().__init__(left, right, Token(TypeId.DIV, opchar))


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

    def error(self) -> None:
        raise RuntimeError('Invalid syntax')

    def eat(self, token_type: TypeId) -> None:
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def factor(self) -> AST:
        """factor : INTEGER | LPAREN expr RPAREN"""
        token: Token = self.current_token
        if token.type == TypeId.INT:
            self.eat(TypeId.INT)
            return Num(token)
        elif token.type == TypeId.LPAR:
            self.eat(TypeId.LPAR)
            node: AST = self.expr()
            self.eat(TypeId.RPAR)
            return node

        self.error()
        return Eof()

    def term(self) -> AST:
        """term : factor ((MUL | DIV) factor)*"""
        node: AST = self.factor()

        while self.current_token.type in (TypeId.MUL, TypeId.DIV):
            token: Token = self.current_token
            if token.type == TypeId.MUL:
                self.eat(TypeId.MUL)
                node = Mul(node, self.factor())
            elif token.type == TypeId.DIV:
                self.eat(TypeId.DIV)
                node = Div(node, self.factor())

        return node

    def expr(self) -> AST:
        """
        expr   : term ((ADD | SUB) term)*
        term   : factor ((MUL | DIV) factor)*
        factor : INTEGER | LPAREN expr RPAREN
        """
        node: AST = self.term()

        while self.current_token.type in (TypeId.ADD, TypeId.SUB):
            token: Token = self.current_token
            if token.type == TypeId.ADD:
                self.eat(TypeId.ADD)
                node = Add(node, self.term())
            elif token.type == TypeId.SUB:
                self.eat(TypeId.SUB)
                node = Sub(node, self.term())

        return node

    def parse(self) -> AST:
        return self.expr()


class NodeVisitor:
    def visit(self, node) -> int:
        method_name = 'visit_' + type(node).__name__
        method_name = method_name.lower()
        return getattr(self, method_name, self.raise_visit_error)

    def raise_visit_error(self, node) -> None:
        raise RuntimeError('No visit_{} method'.format(type(node).__name__))


class Interpreter(NodeVisitor):
    def visit_add(self, node) -> int:
        return self.visit(node.left) + self.visit(node.right)

    def visit_sub(self, node) -> int:
        return self.visit(node.left) - self.visit(node.right)

    def visit_mul(self, node) -> int:
        return self.visit(node.left) * self.visit(node.right)

    def visit_div(self, node) -> int:
        return int(self.visit(node.left) / self.visit(node.right))

    def visit_num(self, node) -> int:
        return node.value

    def interpret(self, ast) -> int:
        return self.visit(ast)


def main() -> None:
    while True:
        try:
            text: str = input('spi> ')
        except EOFError:
            break
        if not text:
            continue

        lexer: Lexer = Lexer(text)
        parser: Parser = Parser(lexer)
        interpreter: Interpreter = Interpreter()
        result: int = interpreter.interpret(parser.parse())
        print(result)


if __name__ == '__main__':
    main()
