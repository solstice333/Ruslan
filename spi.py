from enum import Enum
from typing import \
    Any, \
    TypeVar, \
    Callable, \
    Optional, \
    Union, \
    List, \
    Dict, \
    Mapping, \
    NamedTuple, \
    Iterator, \
    Iterable, \
    cast
from abc import ABC, abstractmethod
from anytree import RenderTree # type:ignore
from anytree import \
    PostOrderIter, \
    NodeMixin
from re import Pattern, RegexFlag

import re
import argparse

T = TypeVar('T')


class TypeIdValue(NamedTuple):
    pat: str
    re: bool = False
    type: Callable[[str], Any] = str


class TypeId(Enum):
    PROGRAM = TypeIdValue(pat=r"[pP][rR][oO][gG][rR][aA][mM]", re=True)
    VAR = TypeIdValue(pat=r"[vV][aA][rR]", re=True)
    PROCEDURE = TypeIdValue(
        pat=r"[pP][rR][oO][cC][eE][dD][uU][rR][eE]", re=True)
    COMMA = TypeIdValue(pat=",")
    INTEGER = TypeIdValue(pat=r"[iI][nN][tT][eE][gG][eE][rR]", re=True)
    REAL = TypeIdValue(pat=r"[rR][eE][aA][lL]", re=True)
    REAL_CONST = TypeIdValue(pat=r"\d+\.\d*", re=True, type=float)
    INT_CONST = TypeIdValue(pat=r"\d+", re=True, type=int)
    ADD = TypeIdValue(pat='+')
    SUB = TypeIdValue(pat='-')
    MUL = TypeIdValue(pat='*')
    INT_DIV = TypeIdValue(pat=r"[dD][iI][vV]", re=True)
    FLOAT_DIV = TypeIdValue(pat='/')
    LPAR = TypeIdValue(pat='(')
    RPAR = TypeIdValue(pat=')')
    EOF = TypeIdValue(pat=r"$", re=True)
    BEGIN = TypeIdValue(pat=r"[bB][eE][gG][iI][nN]", re=True)
    END = TypeIdValue(pat=r"[eE][nN][dD]", re=True)
    DOT = TypeIdValue(pat=".")
    ID = TypeIdValue(pat=r"[a-zA-Z_]\w*", re=True)
    ASSIGN = TypeIdValue(pat=":=")
    COLON = TypeIdValue(pat=":")
    SEMI = TypeIdValue(pat=";")
    COMMENT = TypeIdValue(pat=r"\{.*\}", re=True)
    NEWLINE = TypeIdValue(pat=r"\n", re=True)

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
    def __init__(
        self, 
        ty: TypeId, 
        value: Union[str, int, float, None]
    ) -> None:
        self.type: TypeId = ty
        self.value: Union[str, int, float, None] = value

    def __str__(self) -> str:
        """String representation of the class instance.

        Examples:
            Token(INT_CONST, 3)
            Token(ADD, '+')
            Token(MUL, '*')
        """
        return f"Token({self.type}, {self.value})"

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return bool(self.value)

    def __eq__(self, other) -> bool:
        return self.type == other.type and \
            self.value.lower() == other.value.lower() \
            if isinstance(self.value, str) else \
            self.value == other.value

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class Lexer(Iterable[Token]):
    class TypeIdInfo(NamedTuple):
        typeid: TypeId
        pattern: str
        token: Token=Token(TypeId.EOF, None)

    RES_KW: List[TypeId] = [
        TypeId.PROGRAM,
        TypeId.VAR,
        TypeId.PROCEDURE,
        TypeId.INT_DIV,
        TypeId.INTEGER,
        TypeId.REAL,
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
                        typeid=tid,
                        pattern=TypeId.pattern(tid)
                    )
                    for name, tid in TypeId.members().items()
                }
            cls.__TOKEN_NAME_TO_TID_INFO.update(cls._RES_KW_TO_TID_INFO())
        return cls.__TOKEN_NAME_TO_TID_INFO

    def __init__(self, text: str) -> None:
        self._text: str = text
        self.linenum: int = 1

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

            if any_of(
                [TypeId.NEWLINE, TypeId.COMMENT], 
                lambda tid_elem: tid == tid_elem
            ):
                if tid == TypeId.NEWLINE:
                    self.linenum += 1
                continue

            if token_spec[name].token:
                yield token_spec[name].token
            else:
                yield Token(tid, tid.value.type(m[name]))

        yield Token(TypeId.EOF, None)

    def __iter__(self) -> Iterator[Token]:
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """

        return self._iter_tokens()


class IAST(ABC, NodeMixin):
    @property # type:ignore
    @abstractmethod
    def token(self) -> Token:
        pass

    @property # type:ignore
    @abstractmethod
    def linenum(self) -> int:
        pass

    @linenum.setter # type:ignore
    @abstractmethod
    def linenum(self, num: int) -> None:
        pass


class AST(IAST):
    def __init__(self):
        self._linenum = 0

    @property
    def linenum(self) -> int:
        return self._linenum

    @linenum.setter
    def linenum(self, num: int) -> None:
        self._linenum = num 

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.token})"

    def __repr__(self) -> str:
        return str(self)


class Program(AST):
    def __init__(self, name: str, block: 'Block') -> None:
        super().__init__()
        self._token = Token(TypeId.EOF, type(self).__name__)
        self.name: str = name
        self.block: 'Block' = block
        self.children: List[AST] = [self.block]

    @property
    def token(self) -> Token:
        return self._token


class Block(AST):
    def __init__(
        self, 
        declarations: List['VarDecl'], 
        compound_statement: 'Compound'
    ) -> None:
        super().__init__()
        self._token = Token(TypeId.EOF, type(self).__name__)
        self.declarations: List['VarDecl'] = declarations
        self.compound_statement: 'Compound' = compound_statement
        self.children: List[AST] = self.declarations + [self.compound_statement]

    @property
    def token(self) -> Token:
        return self._token


class VarDecl(AST)    :
    def __init__(self, var: 'Var', ty: 'Type') -> None:
        super().__init__()
        self._token = Token(TypeId.EOF, type(self).__name__)
        self.var: 'Var' = var
        self.type: 'Type' = ty
        self.children: List[AST] = [self.var, self.type]

    @property
    def token(self) -> Token:
        return self._token


class ProcedureDecl(AST):
    def __init__(self, proc_name: str, block_node: Block):
        self.proc_name = proc_name
        self.block_node = block_node
        self.children = [self.block_node]


class Type(AST):
    def __init__(self, tytok: Token) -> None:
        super().__init__()
        self._token: Token = tytok
        self.type: TypeId = self._token.type
        self.value: str  = self.type.name

    @classmethod
    def copy(cls, other: 'Type') -> 'Type':
        return cls(other._token)

    @property
    def token(self) -> Token:
        return self._token
    

class BinOp(AST):
    def __init__(self, left: AST, right: AST, optok: Token) -> None:
        super().__init__()
        self._token: Token = optok
        self.op: Token = self._token
        self.left: AST = left
        self.right: AST = right
        self.children: List[AST] = [self.left, self.right]

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


class IntDiv(BinOp):
     def __init__(
        self, 
        left: AST, 
        right: AST, 
        opchar: str=TypeId.INT_DIV.value.pat
    ) -> None:
        super().__init__(left, right, Token(TypeId.INT_DIV, opchar))


class FloatDiv(BinOp):
    def __init__(
        self, 
        left: AST, 
        right: AST, 
        opchar: str=TypeId.FLOAT_DIV.value.pat
    ) -> None:
        super().__init__(left, right, Token(TypeId.FLOAT_DIV, opchar))


class UnOp(AST):
    def __init__(self, right: AST, optok: Token ) -> None:
        super().__init__()
        self.right: AST = right
        self.children: List[AST] = [self.right]
        self._token: Token = optok

    @property
    def token(self) -> Token:
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
    def __init__(self, val: Union[int, float]) -> None:
        super().__init__()
        self._token: Token = Token(TypeId.EOF, None)
        self.value: Union[int, float] = 0

        if isinstance(val, int):
            self._token = Token(TypeId.INT_CONST, val)
            self.value = cast(int, self._token.value)
        elif isinstance(val, float):
            self._token = Token(TypeId.REAL_CONST, val)
            self.value = cast(float, self._token.value)
        else:
            raise TypeError("val must be int or float")

    @property
    def token(self) -> Token:
        return self._token


class Compound(AST):
    """Represents a 'BEGIN ... END' block"""
    def __init__(self, children: Optional[List[AST]]=None) -> None:
        super().__init__()
        self._token: Token = Token(TypeId.EOF, "Compound")
        self.children: List[AST] = children or []

    @property
    def token(self) -> Token:
        return self._token


class Var(AST):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._token: Token = Token(TypeId.ID, name.lower())
        self.value: str = cast(str, self._token.value)

    @property
    def token(self):
        return self._token


class Assign(AST):
    def __init__(
        self, 
        left: Var, 
        right: AST, 
        opchar: str=TypeId.ASSIGN.value.pat
    ) -> None:
        super().__init__()
        self.left: Var = left
        self.right: AST = right
        self.children: List[AST] = [self.left, self.right]
        self._token: Token = Token(TypeId.ASSIGN, opchar)

    @property
    def token(self):
        return self._token
    

class NoOp(AST):
    def __init__(self) -> None:
        super().__init__()
        self._token: Token = Token(TypeId.EOF, "NoOp")

    @property
    def token(self):
        return self._token
    

class Eof(AST):
    def __init__(self) -> None:
        super().__init__()
        self._token: Token = Token(TypeId.EOF, None)

    @property
    def token(self) -> Token:
        return self._token
    

class Symbol:
    def __init__(
        self, name: str, ty: Optional['BuiltinTypeSymbol']=None) -> None:
        self.name: str = name
        self.type: Optional['BuiltinTypeSymbol'] = ty


class BuiltinTypeSymbol(Symbol):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()


class VarSymbol(Symbol):
    def __init__(self, name: str, ty: BuiltinTypeSymbol) -> None:
        super().__init__(name, ty)

    def __str__(self) -> str:
        return f"<{self.name}:{self.type}>"

    def __repr__(self) -> str:
        return self.__str__()


class SymbolTable:
    def _init_builtins(self):
        self.define(BuiltinTypeSymbol(TypeId.INTEGER.name))
        self.define(BuiltinTypeSymbol(TypeId.REAL.name))

    def __init__(self) -> None:
        self._symbols: Dict[str, Symbol] = {}
        self._init_builtins()

    def __str__(self) -> str:
        return f"Symbols: {list(self._symbols.values())}"

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, name: str) -> Symbol:
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        sym: Optional[Symbol] = self.lookup(name)
        if sym is None:
            raise KeyError(f"{name} does not exist in {type(self).__name__}")
        return cast(Symbol, sym)

    def __iter__(self) -> Iterator[Symbol]:
        return iter(self._symbols.values())

    def __len__(self) -> int:
        return len(self._symbols)

    def lookup(self, name: str) -> Optional[Symbol]:
        return self._symbols.get(name)

    def define(self, sym: Symbol) -> None:
        self._symbols[sym.name] = sym


class Parser:
    def __init__(self, lexer: Lexer) -> None:
        self.lexer: Lexer = lexer
        self._it: Iterator[Token] = iter(self.lexer)
        self.current_token: Token = next(self._it)

    @property
    def linenum(self):
        return self.lexer.linenum

    def error(self, msg: str) -> None:
        raise RuntimeError(f"Invalid syntax: {msg}, line {self.linenum}")

    def eat(self, token_type: TypeId) -> None:
        if self.current_token.type == token_type:
            self.current_token = next(self._it)
        else:
            self.error(
                f"was expecting {token_type}, got {self.current_token.type}")

    def factor(self, lpar_b: bool=False) -> AST:
        """
        factor : 
            ADD factor | 
            SUB factor | 
            INT_CONST | 
            REAL_CONST | 
            LPAREN expr RPAREN | 
            variable
        """

        def assert_no_lpar_rpar(): 
            token = self.current_token
            if token.type == TypeId.LPAR:
                self.error(f"found {token}")
            elif token.type == TypeId.RPAR and not lpar_b:
                self.error(f"{TypeId.RPAR} with no matching {TypeId.LPAR}")

        token: Token = self.current_token

        if token.type == TypeId.ADD:
            self.eat(TypeId.ADD) 
            return Pos(self.factor(lpar_b))
        elif token.type == TypeId.SUB:
            self.eat(TypeId.SUB)
            return Neg(self.factor(lpar_b))
        elif token.type == TypeId.INT_CONST:
            numtok = token
            self.eat(TypeId.INT_CONST)
            assert_no_lpar_rpar()
            return Num(cast(int, numtok.value))
        elif token.type == TypeId.REAL_CONST:
            numtok = token
            self.eat(TypeId.REAL_CONST)
            assert_no_lpar_rpar()
            return Num(cast(float, numtok.value))
        elif token.type == TypeId.LPAR:
            self.eat(TypeId.LPAR)
            node: AST = self.expr(True)
            self.eat(TypeId.RPAR)
            return node
        else:
            return self.variable()


    def term(self, lpar_b: bool=False) -> BinOp:
        """term : factor ((MUL | INT_DIV | FLOAT_DIV) factor)*"""
        node: AST = self.factor(lpar_b)

        while True:
            token: Token = self.current_token
            if token.type == TypeId.MUL:
                self.eat(TypeId.MUL)
                node = Mul(node, self.factor(lpar_b))
            elif token.type == TypeId.INT_DIV:
                self.eat(TypeId.INT_DIV)
                node = IntDiv(node, self.factor(lpar_b))
            elif token.type == TypeId.FLOAT_DIV:
                self.eat(TypeId.FLOAT_DIV)
                node = FloatDiv(node, self.factor(lpar_b))
            else:
                break

        return node

    def expr(self, lpar_b: bool=False) -> BinOp:
        """expr: term ((ADD | SUB) term)* """
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

    def empty(self) -> NoOp:
        """empty rule"""
        return NoOp()

    def variable(self) -> Var:
        """variable: ID"""
        node = Var(cast(str, self.current_token.value))
        node.linenum = self.linenum
        self.eat(TypeId.ID)
        return node

    def assignment_statement(self) -> Assign:
        """assignment_statement: variable ASSIGN expr"""
        left = self.variable()
        self.eat(TypeId.ASSIGN)
        right = self.expr()
        return Assign(left, right)

    def statement(self) -> AST:
        """statement: compound_statement | assignment_statement | empty"""
        tokty = self.current_token.type

        if tokty == TypeId.BEGIN:
            node = self.compound_statement()
        elif tokty == TypeId.ID:
            node = self.assignment_statement()
        else:
            node = self.empty()

        return node

    def statement_list(self) -> List[AST]:
        """statement_list: statement (SEMI statement)*"""
        statements = []
        statements.append(self.statement())

        while self.current_token.type == TypeId.SEMI:
            self.eat(TypeId.SEMI)
            statements.append(self.statement())

        if self.current_token.type == TypeId.ID:
            self.error(f"found {self.current_token}. Expected semi-colon")

        return statements

    def compound_statement(self) -> Compound:
        """compound_statement: BEGIN statement_list END"""
        self.eat(TypeId.BEGIN)
        nodes = self.statement_list()
        self.eat(TypeId.END)
        return Compound(nodes)


    def type_spec(self) -> Type:
        """type_spec: INTEGER | REAL"""

        tok = Type(self.current_token)
        if self.current_token.type == TypeId.INTEGER:
            self.eat(TypeId.INTEGER)
        elif self.current_token.type == TypeId.REAL:
            self.eat(TypeId.REAL)
        else:
            self.error(f"expected a type spec. Got {self.current_token}")
        return tok

    def variable_declaration(self) -> List[VarDecl]:
        """
        variable_declaration: variable (COMMA variable)* COLON type_spec
        """
        var_nodes = [self.variable()]
        while self.current_token.type == TypeId.COMMA:
            self.eat(TypeId.COMMA)
            var_nodes.append(self.variable())
        self.eat(TypeId.COLON)
        ty_node = self.type_spec()
        return [
            VarDecl(var_node, Type.copy(ty_node)) \
            for var_node in var_nodes \
        ]

    def declarations(self) -> List[AST]:
        """
        declarations: 
            VAR (variable_declaration SEMI)+ | 
            (PROCEDURE variable SEMI block SEMI)* | 
            empty
        """
        declarations: List[AST] = []

        if self.current_token.type == TypeId.VAR:
            self.eat(TypeId.VAR)
            while self.current_token.type == TypeId.ID:
                declarations += self.variable_declaration()
                self.eat(TypeId.SEMI)

        while self.current_token.type == TypeId.PROCEDURE:
            self.eat(TypeId.PROCEDURE)
            var_n: Var = self.variable()
            proc_name: str = var_n.value
            self.eat(TypeId.SEMI)
            block_n: Block = self.block()
            self.eat(TypeId.SEMI)
            declarations.append(ProcedureDecl(proc_name, block_n))

        return declarations

    def block(self) -> Block:
        """block: declarations compound_statement"""
        declaration_nodes = self.declarations()
        compound_statement_node = self.compound_statement()
        return Block(declaration_nodes, compound_statement_node)

    def program(self) -> Program:
        """program : PROGRAM variable SEMI block DOT"""
        self.eat(TypeId.PROGRAM)
        var_node = self.variable()
        prog_name = var_node.value
        self.eat(TypeId.SEMI)
        block_node = self.block()
        self.eat(TypeId.DOT)
        return Program(prog_name, block_node)

    def parse_expr(self) -> Union[BinOp, NoOp]:
        return self.expr()

    def parse_compound(self) -> Compound:
        return self.compound_statement()

    def parse(self) -> Program:
        prog = self.program()
        if self.current_token.type != TypeId.EOF:
            self.error(f"expected EOF. Got {self.current_token}")
        return prog


class NodeVisitor(ABC):
    def _gen_visit_method_name(self, node: AST) -> str:
        method_name = '_visit_' + type(node).__name__
        return method_name.lower()

    def visit(self, node: AST) -> Union[int, float, None]:
        method_name = self._gen_visit_method_name(node)
        return getattr(self, method_name, self.raise_visit_error)(node)

    def raise_visit_error(self, node: AST) -> None:
        method_name = self._gen_visit_method_name(node)
        raise RuntimeError(f"No {method_name} method")

    @abstractmethod
    def _visit_pos(self, node: Pos) -> Union[int, float, None]:
        pass

    @abstractmethod
    def _visit_neg(self, node: Neg) -> Union[int, float, None]:
        pass

    @abstractmethod
    def _visit_add(self, node: AST) -> Union[int, float, None]:
        pass

    @abstractmethod
    def _visit_sub(self, node: AST) -> Union[int, float, None]:
        pass

    @abstractmethod
    def _visit_mul(self, node: AST) -> Union[int, float, None]:
        pass

    @abstractmethod
    def _visit_intdiv(self, node: AST) -> Union[int, float, None]:
        pass

    @abstractmethod
    def _visit_floatdiv(self, node: AST) -> Union[int, float, None]:
        pass

    @abstractmethod
    def _visit_num(self, node: Num) -> Union[int, float, None]:
        pass

    @abstractmethod
    def _visit_compound(self, node: Compound) -> None:
        pass

    @abstractmethod
    def _visit_noop(self, node: NoOp) -> None:
        pass

    @abstractmethod
    def _visit_assign(self, node: Assign) -> None:
        pass

    @abstractmethod
    def _visit_var(self, node: Var) -> Union[int, float, None]:
        pass

    @abstractmethod
    def _visit_program(self, node: Program) -> None:
        pass

    @abstractmethod
    def _visit_block(self, node: Block) -> None:
        pass

    @abstractmethod
    def _visit_vardecl(self, node: VarDecl) -> None:
        pass

    @abstractmethod
    def _visit_type(self, node: Type) -> None:
        pass


class SymbolTableBuilder(NodeVisitor):
    def __init__(self):
        self.table = SymbolTable()

    def _visit_binop(self, node: AST) -> None:
        self.visit(node.left)
        self.visit(node.right)

    def _visit_unop(self, node: UnOp) -> None:
        self.visit(node.right)

    def _visit_pos(self, node: Pos) -> None:
        self._visit_unop(node)

    def _visit_neg(self, node: Neg) -> None:
        self._visit_unop(node)

    def _visit_add(self, node: AST) -> None:
        self._visit_binop(node)

    def _visit_sub(self, node: AST) -> None:
        self._visit_binop(node)

    def _visit_mul(self, node: AST) -> None:
        self._visit_binop(node)

    def _visit_intdiv(self, node: AST) -> None:
        self._visit_binop(node)

    def _visit_floatdiv(self, node: AST) -> None:
        self._visit_binop(node)

    def _visit_num(self, node: Num) -> None:
        pass

    def _visit_compound(self, node: Compound) -> None:
        for n in node.children:
            self.visit(n)

    def _visit_noop(self, node: NoOp) -> None:
        return None

    def _visit_assign(self, node: Assign) -> None:
        assignee = node.left
        var_name = assignee.value
        if self.table.lookup(var_name) is None:
            linenum = assignee.linenum
            raise NameError(
                f"{var_name}" + (f" at line {linenum}" if linenum else "")
            )
        self.visit(node.right)

    def _visit_var(self, node: Var) -> None:
        var_name = node.value
        if self.table.lookup(var_name) is None:
            linenum = node.linenum
            raise NameError(
                f"{var_name}" + (f" at line {linenum}" if linenum else "")
            )

    def _visit_program(self, node: Program) -> None:
        self.visit(node.block)

    def _visit_block(self, node: Block) -> None:
        for n in node.children:
            self.visit(n)

    def _visit_vardecl(self, node: VarDecl) -> None:
        type_name = node.type.value
        type_sym = self.table.lookup(type_name)
        var_name = node.var.value
        var_sym = VarSymbol(var_name, type_sym)
        self.table.define(var_sym)

    def _visit_type(self, node: Type) -> None:
        raise NotImplementedError()

    def build(self, node: AST) -> None:
        self.visit(node)


class Interpreter(NodeVisitor):
    def __init__(self):
        self.GLOBAL_SCOPE: Dict[str, Union[int, float]] = {}

    def _visit_pos(self, node: Pos) -> Union[int, float]:
        return +cast(Union[int, float], self.visit(node.right))

    def _visit_neg(self, node: Neg) -> Union[int, float]:
        return -cast(Union[int, float], self.visit(node.right))

    def _visit_add(self, node: AST) -> Union[int, float]:
        return \
            cast(Union[int, float], self.visit(node.left)) + \
            cast(Union[int, float], self.visit(node.right))

    def _visit_sub(self, node: AST) -> Union[int, float]:
        return \
            cast(Union[int, float], self.visit(node.left)) - \
            cast(Union[int, float], self.visit(node.right))

    def _visit_mul(self, node: AST) -> Union[int, float]:
        return \
            cast(Union[int, float], self.visit(node.left)) * \
            cast(Union[int, float], self.visit(node.right))

    def _visit_intdiv(self, node: AST) -> Union[int, float]:
        return \
            cast(Union[int, float], self.visit(node.left)) // \
            cast(Union[int, float], self.visit(node.right))

    def _visit_floatdiv(self, node: AST) -> Union[int, float]:
        return \
            cast(Union[int, float], self.visit(node.left)) / \
            cast(Union[int, float], self.visit(node.right))

    def _visit_num(self, node: Num) -> Union[int, float]:
        return node.value

    def _visit_compound(self, node: Compound) -> None:
        for child in node.children:
            self.visit(child)

    def _visit_noop(self, node: NoOp) -> None:
        pass

    def _visit_assign(self, node: Assign) -> None:
        self.GLOBAL_SCOPE[node.left.value] = \
            cast(Union[int, float], self.visit(node.right))

    def _visit_var(self, node: Var) -> Union[int, float]:
        name = node.value
        val = self.GLOBAL_SCOPE.get(name.lower())
        if val is None:
            raise NameError(repr(name))
        return val

    def _visit_program(self, node: Program) -> None:
        self.visit(node.block)

    def _visit_block(self, node: Block) -> None:
        for ast in node.children:
            self.visit(ast)

    def _visit_vardecl(self, node: VarDecl) -> None:
        pass

    def _visit_type(self, node: Type) -> None:
        pass

    def interpret(self, ast: AST) -> Union[int, float, None]:
        return self.visit(ast)


def any_of(vals: Iterable[T], pred: Callable[[T], bool]):
    for val in vals:
        if pred(val):
            return True
    return False

def main() -> None:
    argparser = argparse.ArgumentParser(description="simple pascal interpreter")
    argparser.add_argument("FILE", help="pascal file")
    args = argparser.parse_args()

    with open(args.FILE) as pascal_file:
        text = pascal_file.read()

    lexer: Lexer = Lexer(text)
    parser: Parser = Parser(lexer)
    ast: AST = parser.parse()

    st_bldr: SymbolTableBuilder = SymbolTableBuilder()
    st_bldr.build(ast)
    print(st_bldr.table)

    interpreter: Interpreter = Interpreter()
    interpreter.interpret(ast)
    print(interpreter.GLOBAL_SCOPE)  


if __name__ == '__main__':
    main()
