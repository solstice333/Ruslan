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
import logging

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
        self._token = Token(TypeId.EOF, None)

    @property
    def token(self) -> Token:
        return self._token

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
        self.name: str = name
        self.block: 'Block' = block
        self.children: List[AST] = [self.block]


class Block(AST):
    def __init__(
        self, 
        declarations: List['VarDecl'], 
        compound_statement: 'Compound'
    ) -> None:
        super().__init__()
        self.declarations: List['VarDecl'] = declarations
        self.compound_statement: 'Compound' = compound_statement
        self.children: List[AST] = self.declarations + [self.compound_statement]


class VarDecl(AST):
    def __init__(self, var: 'Var', ty: 'Type') -> None:
        super().__init__()
        self.var: 'Var' = var
        self.type: 'Type' = ty
        self.children: List[AST] = [self.var, self.type]


class Param(VarDecl):
    def __init__(self, var: 'Var', ty: 'Type') -> None:
        super().__init__(var, ty)


class ProcDecl(AST):
    def __init__(self, proc_name: str, params: List[Param], block_node: Block):
        super().__init__()
        self._token: Token = Token(TypeId.EOF, proc_name)
        self.name: str = proc_name
        self.params: List[Param] = params
        self.block_node: Block = block_node
        self.children: List[AST] = self.params + [self.block_node]


class Type(AST):
    def __init__(self, tytok: Token) -> None:
        super().__init__()
        self._token: Token = tytok
        self.type: TypeId = self._token.type
        self.value: str  = self.type.name

    @classmethod
    def copy(cls, other: 'Type') -> 'Type':
        tok = cls(other._token)
        tok.linenum = other.linenum
        return tok


class BinOp(AST):
    def __init__(self, left: AST, right: AST, optok: Token) -> None:
        super().__init__()
        self._token: Token = optok
        self.op: Token = self._token
        self.left: AST = left
        self.right: AST = right
        self.children: List[AST] = [self.left, self.right]


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
        self.value: Union[int, float] = 0
        self._token: Token = Token(TypeId.EOF, None)

        if isinstance(val, int):
            self.value = cast(int, val)
            self._token = Token(TypeId.INT_CONST, self.value)
        elif isinstance(val, float):
            self.value = cast(float, val)
            self._token = Token(TypeId.REAL_CONST, self.value)
        else:
            raise TypeError("val must be int or float")


class Compound(AST):
    """Represents a 'BEGIN ... END' block"""
    def __init__(self, children: Optional[List[AST]]=None) -> None:
        super().__init__()
        self.children: List[AST] = children or []


class Var(AST):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._token: Token = Token(TypeId.ID, name.lower())
        self.value: str = cast(str, self._token.value)


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


class NoOp(AST):
    def __init__(self) -> None:
        super().__init__()


class Eof(AST):
    def __init__(self) -> None:
        super().__init__()
        self._token: Token = Token(TypeId.EOF, None)


class Symbol:
    def __init__(self, name: str) -> None:
        self.name: str = name


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
    def __init__(self, name: str, params: List[VarSymbol]=None) -> None:
        super().__init__(name)
        self.params: List[VarSymbol] = params if params is not None else []

    def __str__(self) -> str:
        return f"{type(self).__name__}(name={self.name}, params={self.params})"

class ScopedSymbolTable:
    def __init__(
        self, 
        name: str, 
        level: int, 
        encl_scope: Optional['ScopedSymbolTable']=None
    ) -> None:
        self._symbols: Dict[str, Symbol] = {}
        self.name: str = name
        self.level: int = level
        self.encl_scope: Optional[ScopedSymbolTable] = encl_scope

        self.insert(BuiltinTypeSymbol(TypeId.INTEGER.name))
        self.insert(BuiltinTypeSymbol(TypeId.REAL.name))

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
            "="*len(header),
            f"{name_header:<15}: {self.name}",
            f"{level_header:<15}: {self.level}",
            f"{header2}",
            "-"*len(header2),
        ]
        lines += [f"{name:7}: {sym!r}" for name, sym in self._symbols.items()]
        return "\n".join(lines)

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
        logging.info(f"Lookup: {name}")
        return self._symbols.get(name)

    def insert(self, sym: Symbol) -> None:
        logging.info(f"Insert: {sym.name}")
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

        tok.linenum = self.linenum
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

    def formal_parameters(self) -> List[Param]:
        """
        formal_parameters: variable (COMMA variable)* COLON type_spec
        """
        param_nodes = [self.variable()]
        while self.current_token.type == TypeId.COMMA:
            self.eat(TypeId.COMMA)
            param_nodes.append(self.variable())
        self.eat(TypeId.COLON)
        ty_node = self.type_spec()
        return [
            Param(param_node, Type.copy(ty_node)) \
            for param_node in param_nodes \
        ]

    def formal_parameter_list(self) -> List[Param]:
        """
        formal_parameter_list: formal_parameters (SEMI formal_parameters)*
        """
        param_nodes = self.formal_parameters()
        while self.current_token.type == TypeId.SEMI:
            self.eat(TypeId.SEMI)
            param_nodes += self.formal_parameters()
        return param_nodes

    def declarations(self) -> List[AST]:
        """
        declarations: 
            (VAR (variable_declaration SEMI)+)* | 
            (PROCEDURE variable (LPAR formal_parameter_list RPAR)? 
                SEMI block SEMI)* | 
            empty
        """
        declarations: List[AST] = []

        while self.current_token.type == TypeId.VAR:
            self.eat(TypeId.VAR)
            while self.current_token.type == TypeId.ID:
                declarations += self.variable_declaration()
                self.eat(TypeId.SEMI)

        while self.current_token.type == TypeId.PROCEDURE:
            self.eat(TypeId.PROCEDURE)
            var_n: Var = self.variable()
            proc_name: str = var_n.value

            params: List[Param] = []
            if self.current_token.type == TypeId.LPAR:
                self.eat(TypeId.LPAR)
                params = self.formal_parameter_list()
                self.eat(TypeId.RPAR)

            self.eat(TypeId.SEMI)
            block_n: Block = self.block()
            self.eat(TypeId.SEMI)
            declarations.append(ProcDecl(proc_name, params, block_n))

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
    def _visit_procdecl(self, node: ProcDecl) -> None:
        pass

    @abstractmethod
    def _visit_type(self, node: Type) -> None:
        pass


class SemanticAnalyzer(NodeVisitor):
    def __init__(self):
        self.current_scope: Optional[ScopedSymbolTable] = None

    @property
    def safe_current_scope(self) -> ScopedSymbolTable:
        assert self.current_scope is not None
        return cast(ScopedSymbolTable, self.current_scope)

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
        self.visit(node.left)
        self.visit(node.right)

    def _visit_var(self, node: Var) -> None:
        var_name = node.value
        if self.safe_current_scope.lookup(var_name) is None:
            linenum = node.linenum
            raise NameError(
                f"{var_name}" + (f" at line {linenum}" if linenum else ""))

    def _visit_program(self, node: Program) -> None:
        scope_name = "global"
        logging.info(f"ENTER scope {scope_name}")
        global_scope = ScopedSymbolTable(scope_name, 1, self.current_scope)
        self.current_scope = global_scope

        self.visit(node.block)

        logging.info(global_scope)
        self.current_scope = self.current_scope.encl_scope
        logging.info(f"LEAVE scope {global_scope.name}")

    def _visit_block(self, node: Block) -> None:
        for n in node.children:
            self.visit(n)

    def _visit_vardecl(self, node: VarDecl) -> None:
        var = node.var
        ty = node.type

        type_name = ty.value
        type_sym = self.safe_current_scope.lookup(type_name)

        if type_sym is None:
            raise TypeError(
                f"{type_name} not in symbol table at line {ty.linenum}")

        var_name = var.value
        var_sym = VarSymbol(var_name, cast(BuiltinTypeSymbol, type_sym))

        if self.safe_current_scope.lookup(var_sym.name) is not None:
            linenum = node.var.linenum
            raise NameError(
                f"duplicate identifier {var_sym.name} " + \
                f"found at line {linenum}"
            )

        self.safe_current_scope.insert(var_sym)

    def _visit_procdecl(self, node: ProcDecl) -> None:
        proc_name = node.name
        proc_symbol = ProcSymbol(proc_name)
        self.safe_current_scope.insert(proc_symbol)

        scope_name = proc_name
        logging.info(f"ENTER scope {scope_name}")
        proc_scope = ScopedSymbolTable(
            scope_name, self.safe_current_scope.level + 1, self.current_scope)
        self.current_scope = proc_scope

        for param in node.params:
            param_name = param.var.value
            param_ty = self.current_scope.lookup(param.type.value)

            assert param_ty is not None
            assert isinstance(param_ty, BuiltinTypeSymbol)

            if self.safe_current_scope.lookup(param_name) is not None:
                linenum = param.var.linenum
                raise NameError(
                    f"duplicate identifier {param_name} " + \
                    f"found at line {linenum}"
                )

            var_sym = VarSymbol(param_name, cast(BuiltinTypeSymbol, param_ty))
            proc_symbol.params.append(var_sym)
            self.current_scope.insert(var_sym)

        self.visit(node.block_node)

        logging.info(proc_scope)
        self.current_scope = self.current_scope.encl_scope
        logging.info(f"LEAVE scope {proc_scope.name}")

    def _visit_type(self, node: Type) -> None:
        raise NotImplementedError()

    def analyze(self, node: AST) -> None:
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

    def _visit_procdecl(self, node: ProcDecl) -> None:
        pass

    def _visit_type(self, node: Type) -> None:
        pass

    def interpret(self, ast: AST) -> Union[int, float, None]:
        val = self.visit(ast)
        logging.info(f"global runtime memory: {self.GLOBAL_SCOPE}")
        return val


def any_of(vals: Iterable[T], pred: Callable[[T], bool]):
    for val in vals:
        if pred(val):
            return True
    return False

def setup_logging(verbose: bool) -> None:
    logging.disable(logging.NOTSET)
    logging.basicConfig(format="{message}", style="{")
    if verbose:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

def main() -> None:
    argparser = argparse.ArgumentParser(description="simple pascal interpreter")
    argparser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="verbose debugging output"
    )
    argparser.add_argument("FILE", help="pascal file")
    args = argparser.parse_args()

    setup_logging(args.verbose)

    with open(args.FILE) as pascal_file:
        text = pascal_file.read()

    lexer: Lexer = Lexer(text)
    parser: Parser = Parser(lexer)
    ast: AST = parser.parse()

    st_bldr: SemanticAnalyzer = SemanticAnalyzer()
    st_bldr.analyze(ast)

    interpreter: Interpreter = Interpreter()
    interpreter.interpret(ast)

logging.disable(logging.CRITICAL)
if __name__ == '__main__':
    main()
