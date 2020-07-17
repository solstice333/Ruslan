import sys
import os
import os.path
import unittest
import io
import subprocess

from pprint import pprint
from anytree import PostOrderIter, RenderTree


def append_spi_dir_to_sys_path():
    spi_path = "spi.py"
    while not os.path.isfile(spi_path):
        old_spi_path = spi_path
        spi_path = os.path.join("..", spi_path)
        assert old_spi_path != spi_path
    spi_path = os.path.dirname(spi_path)
    sys.path.append(os.path.realpath(spi_path))


append_spi_dir_to_sys_path()
from spi import *


def to_list_of_lines(s):
    return s.strip().splitlines()


def make_parser(text):
    lexer = Lexer(text)
    return Parser(lexer)


def make_parser_from_file(path):
    with open(path) as f:
        text = f.read()
    return make_parser(text)


def make_expr_ast(txt):
    p = make_parser(txt)
    return p.parse_expr()


def make_prog_ast(txt):
    parser = make_parser(txt)
    return parser.parse()


def make_compound_ast(txt):
    parser = make_parser(txt)
    return parser.parse_compound()


def make_prog_ast_from_file(path):
    with open(path) as f:
        txt = f.read()
    return make_prog_ast(txt)


def make_compound_ast_from_file(path):
    with open(path) as f:
        txt = f.read()
    return make_compound_ast(txt)


class Float:
    def __init__(self, val):
        self.val = val

    def eq(self, eps, other):
        if isinstance(other, float):
            other = Float(other)
        return abs(self.val - other.val) < eps


class StringBuffers(NamedTuple):
    scope: io.StringIO
    stack: io.StringIO


class StringBufferLogger():
    DEFAULT_FORMATTER = logging.Formatter()

    def __init__(self, name):
        self._name = name
        self._logger = getLogger(self._name)
        self._sbuf = None
        self._handler = None

    @property
    def sbuf(self):
        return self._sbuf

    def reset_sbuf(self):
        if self._handler is not None:
            self._logger.removeHandler(self._handler)
        self._sbuf = io.StringIO()
        self._handler = logging.StreamHandler(self._sbuf)
        self._handler.setFormatter(type(self).DEFAULT_FORMATTER)
        self._logger.addHandler(self._handler)


class StringBufferLoggers():
    _initd = False
    _scope = None
    _stack = None

    @classmethod
    def _reset_sbufs(cls):
        cls._scope.reset_sbuf()
        cls._stack.reset_sbuf()

    @classmethod
    def config(cls):
        logging.config.dictConfig({
            "version": 1,
            "loggers": {
                "stack": {
                    "level": logging.DEBUG
                },
                "scope": {
                    "level": logging.DEBUG
                }
            }
        })
        cls._scope = StringBufferLogger("scope")
        cls._stack = StringBufferLogger("stack")
        cls._initd = True

    def __enter__(self):
        cls = type(self)
        logging.disable(logging.NOTSET)
        if not cls._initd:
            cls.config()
        cls._reset_sbufs()
        return StringBuffers(scope=cls._scope.sbuf, stack=cls._stack.sbuf)

    def __exit__(self, exc_ty, exc_val, tb):
        logging.disable()


class LexerTestCase(unittest.TestCase):
    def makeLexer(self, text):
        lexer = Lexer(text)
        return lexer

    def test_lexer_integer(self):
        lexer = self.makeLexer('234')
        token = next(iter(lexer))
        self.assertEqual(token.type, TokenType.INT_CONST)
        self.assertEqual(token.value, 234)

    def test_lexer_mul(self):
        lexer = self.makeLexer('*')
        token = next(iter(lexer))
        self.assertEqual(token.type, TokenType.MUL)
        self.assertEqual(token.value, '*')

    def test_lexer_div(self):
        lexer = self.makeLexer(' dIv ')
        token = next(iter(lexer))
        self.assertEqual(token.type, TokenType.INT_DIV)
        self.assertEqual(token.value, 'dIv')

    def test_lexer_plus(self):
        lexer = self.makeLexer('+')
        token = next(iter(lexer))
        self.assertEqual(token.type, TokenType.ADD)
        self.assertEqual(token.value, '+')

    def test_lexer_minus(self):
        lexer = self.makeLexer('-')
        token = next(iter(lexer))
        self.assertEqual(token.type, TokenType.SUB)
        self.assertEqual(token.value, '-')

    def test_lexer_lparen(self):
        lexer = self.makeLexer('(')
        token = next(iter(lexer))
        self.assertEqual(token.type, TokenType.LPAR)
        self.assertEqual(token.value, '(')

    def test_lexer_rparen(self):
        lexer = self.makeLexer(')')
        token = next(iter(lexer))
        self.assertEqual(token.type, TokenType.RPAR)
        self.assertEqual(token.value, ')')

    def test_lexer_ident(self):
        pass

    def test_lexer_res_kw_single_char_var(self):
        lexer = self.makeLexer("BEGIN a := 0; END.")
        tokens = list(iter(lexer))

        self.assertEqual(len(tokens), 8)
        self.assertEqual(tokens[0], BeginTok("BEGIN", Position(1, 1)))
        self.assertEqual(tokens[1], IdTok("a", Position(1, 7)))
        self.assertEqual(tokens[2], AssignTok(":=", Position(1, 9)))
        self.assertEqual(tokens[3], IntConstTok(0, Position(1, 12)))
        self.assertEqual(tokens[4], SemiTok(";", Position(1, 13)))
        self.assertEqual(tokens[5], EndTok("END", Position(1, 15)))
        self.assertEqual(tokens[6], DotTok(".", Position(1, 18)))
        self.assertEqual(tokens[7], EofTok("", Position(1, 19)))

    def test_lexer_res_kw_multi_char_var(self):
        lexer = self.makeLexer("BEGIN foo_bar123 := 0; END.")
        tokens = list(iter(lexer))

        self.assertEqual(len(tokens), 8)
        self.assertEqual(tokens[0], BeginTok("BEGIN", Position(1, 1)))
        self.assertEqual(tokens[1], IdTok("foo_bar123", Position(1, 7)))
        self.assertEqual(tokens[2], AssignTok(":=", Position(1, 18)))
        self.assertEqual(tokens[3], IntConstTok(0, Position(1, 21)))
        self.assertEqual(tokens[4], SemiTok(";", Position(1, 22)))
        self.assertEqual(tokens[5], EndTok("END", Position(1, 24)))
        self.assertEqual(tokens[6], DotTok(".", Position(1, 27)))
        self.assertEqual(tokens[7], EofTok("", Position(1, 28)))

    def test_lexer_vardecl(self):
        lexer = self.makeLexer("VAR x: INTEGER\nVAR y: REAL")
        tokens = list(iter(lexer))
        self.assertEqual(len(tokens), 9)
        self.assertEqual(tokens[0], VarTok("VAR", Position(1, 1)))
        self.assertEqual(tokens[1], IdTok("x", Position(1, 5)))
        self.assertEqual(tokens[2], ColonTok(":", Position(1, 6)))
        self.assertEqual(tokens[3], IntegerTok("INTEGER", Position(1, 8)))
        self.assertEqual(tokens[4], VarTok("VAR", Position(2, 1)))
        self.assertEqual(tokens[5], IdTok("y", Position(2, 5)))
        self.assertEqual(tokens[6], ColonTok(":", Position(2, 6)))
        self.assertEqual(tokens[7], RealTok("REAL", Position(2, 8)))
        self.assertEqual(tokens[8], EofTok("", Position(2, 12)))


class ParserTestCase(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_parser1(self):
        p = make_parser("BEGIN x := 11; y := 2 + x END.")
        ast = p.parse_compound()
        act = [str(el) for el in PostOrderIter(ast)]

        exp = [
            "Var(value=x)",
            "Num(value=11)",
            "Assign(value=:=)",
            "Var(value=y)",
            "Num(value=2)",
            "Var(value=x)",
            "Add(value=+)",
            "Assign(value=:=)",
            "Compound()"
        ]

        self.assertEqual(act, exp)

    def test_parser2(self):
        p = make_parser(
            "BEGIN\n" + \
            "    BEGIN\n" + \
            "        number := 2;\n" + \
            "        a := number;\n" + \
            "        b := 10 * a + 10 * number div 4;\n" + \
            "        c := a - - b\n" + \
            "    END;\n" + \
            "    x := 11;\n" + \
            "END.\n"
        )
        ast = p.parse_compound()

        act = [str(el) for el in PostOrderIter(ast)]
        exp = [
            'Var(value=number)',
            'Num(value=2)',
            'Assign(value=:=)',
            'Var(value=a)',
            'Var(value=number)',
            'Assign(value=:=)',
            'Var(value=b)',
            'Num(value=10)',
            'Var(value=a)',
            'Mul(value=*)',
            'Num(value=10)',
            'Var(value=number)',
            'Mul(value=*)',
            'Num(value=4)',
            'IntDiv(value=div)',
            'Add(value=+)',
            'Assign(value=:=)',
            'Var(value=c)',
            'Var(value=a)',
            'Var(value=b)',
            'Neg(value=-)',
            'Sub(value=-)',
            'Assign(value=:=)',
            'Compound()',
            'Var(value=x)',
            'Num(value=11)',
            'Assign(value=:=)',
            'NoOp()',
            'Compound()'
        ]

        self.assertEqual(act, exp)

    def test_parser3(self):
        with open("part10.pas") as f:
            p = make_parser(f.read())
        ast = p.parse()
        act = [str(el) for el in PostOrderIter(ast)]
        exp = [
            'Var(value=a)',
            'Type(value=INTEGER)',
            'VarDecl()',
            'Var(value=b)',
            'Type(value=INTEGER)',
            'VarDecl()',
            'Var(value=y)',
            'Type(value=REAL)',
            'VarDecl()',
            'Var(value=a)',
            'Num(value=2)',
            'Assign(value=:=)',
            'Var(value=b)',
            'Num(value=10)',
            'Var(value=a)',
            'Mul(value=*)',
            'Num(value=10)',
            'Var(value=a)',
            'Mul(value=*)',
            'Num(value=4)',
            'IntDiv(value=DIV)',
            'Add(value=+)',
            'Assign(value=:=)',
            'Var(value=y)',
            'Num(value=20)',
            'Num(value=7)',
            'FloatDiv(value=/)',
            'Num(value=3.14)',
            'Add(value=+)',
            'Assign(value=:=)',
            'NoOp()',
            'Compound()',
            'Block()',
            'Program(name=Part10AST)'
        ]
        self.assertEqual(act, exp)

    def test_parser_proc(self):
        with open("part12.pas") as f:
            p = make_parser(f.read())

        ast = p.parse()
        act = [str(el) for el in PostOrderIter(ast)]
        exp = [
            'Var(value=a)',
            'Type(value=INTEGER)',
            'VarDecl()',
            'Var(value=a)',
            'Type(value=REAL)',
            'VarDecl()',
            'Var(value=k)',
            'Type(value=INTEGER)',
            'VarDecl()',
            'Var(value=a)',
            'Type(value=INTEGER)',
            'VarDecl()',
            'Var(value=z)',
            'Type(value=INTEGER)',
            'VarDecl()',
            'Var(value=z)',
            'Num(value=777)',
            'Assign(value=:=)',
            'NoOp()',
            'Compound()',
            'Block()',
            'ProcDecl(name=P2)',
            'NoOp()',
            'Compound()',
            'Block()',
            'ProcDecl(name=P1)',
            'Var(value=a)',
            'Num(value=10)',
            'Assign(value=:=)',
            'NoOp()',
            'Compound()',
            'Block()',
            'Program(name=Part12)'
        ]

        self.assertEqual(act, exp)

    def test_fail_parse(self):
        with open("bar.pas") as f:
            p = make_parser(f.read())
        with self.assertRaises(ParserError) as e:
            p.parse_compound()

        errcode = e.exception.error_code
        errtok = e.exception.token
        self.assertEqual(errcode, ErrorCode.UNEXPECTED_TOKEN)
        self.assertEqual(errtok, IdTok("_a", Position(line=6, col=17)))

    def test_fail_parse_type_spec(self):
        with open("bad_type_spec.pas") as f:
            p = make_parser(f.read())
        with self.assertRaises(ParserError) as e:
            p.parse()

        errcode = e.exception.error_code
        errtok = e.exception.token
        self.assertEqual(errcode, ErrorCode.UNEXPECTED_TOKEN)
        self.assertEqual(errtok, IdTok("INTEGE", Position(line=3, col=8)))
        self.assertEqual(
            "ParserError: Unexpected token -> " + \
            "IdTok(INTEGE, Position(line=3, col=8)). " + \
            "Expected one of ['INTEGER', 'REAL', 'BOOLEAN']",
            e.exception.message
        )

    def test_fail_bad_prog_end(self):
        with open("bad_prog_end.pas") as f:
            p = make_parser((f.read()))
        with self.assertRaises(ParserError) as e:
            p.parse()

        errcode = e.exception.error_code
        errtok = e.exception.token
        self.assertEqual(ErrorCode.UNEXPECTED_TOKEN, errcode)
        self.assertEqual(DotTok(".", Position(line=4, col=1)), errtok)
        self.assertEqual("ParserError: Unexpected token -> " + \
                         "DotTok(., Position(line=4, col=1)). " + \
                         "Expected EOF",
                         e.exception.message)

    def test_parser_proc_sig(self):
        ast = make_prog_ast_from_file("part14.pas")
        act = [str(el) for el in PostOrderIter(ast)]
        exp = [
            'Var(value=x)',
            'Type(value=real)',
            'VarDecl()',
            'Var(value=y)',
            'Type(value=real)',
            'VarDecl()',
            'Var(value=a)',
            'Type(value=integer)',
            'Param()',
            'Var(value=y)',
            'Type(value=integer)',
            'VarDecl()',
            'Var(value=x)',
            'Var(value=a)',
            'Var(value=x)',
            'Add(value=+)',
            'Var(value=y)',
            'Add(value=+)',
            'Assign(value=:=)',
            'NoOp()',
            'Compound()',
            'Block()',
            'ProcDecl(name=Alpha)',
            'NoOp()',
            'Compound()',
            'Block()',
            'Program(name=Main)'
        ]
        self.assertEqual(act, exp)

    def test_parser_proc_sig2(self):
        ast = make_prog_ast_from_file("part14_2.pas")
        act = [str(el) for el in PostOrderIter(ast)]
        exp = [
            'Var(value=x)',
            'Type(value=real)',
            'VarDecl()',
            'Var(value=y)',
            'Type(value=real)',
            'VarDecl()',
            'Var(value=a)',
            'Type(value=integer)',
            'Param()',
            'Var(value=b)',
            'Type(value=integer)',
            'Param()',
            'Var(value=c)',
            'Type(value=real)',
            'Param()',
            'Var(value=y)',
            'Type(value=integer)',
            'VarDecl()',
            'Var(value=x)',
            'Var(value=a)',
            'Var(value=x)',
            'Add(value=+)',
            'Var(value=y)',
            'Add(value=+)',
            'Assign(value=:=)',
            'NoOp()',
            'Compound()',
            'Block()',
            'ProcDecl(name=Alpha)',
            'NoOp()',
            'Compound()',
            'Block()',
            'Program(name=Main)'
        ]
        self.assertEqual(act, exp)

    def test_peek_and_advance(self):
        p = make_parser_from_file("foo.pas")
        tok = p.advance()
        self.assertEqual(BeginTok("BEGIN", Position(line=3, col=5)), tok)
        tok = p.advance()
        self.assertEqual(IdTok("number", Position(line=4, col=9)), tok)
        tok = p.peek()
        self.assertEqual(AssignTok(":=", Position(line=4, col=16)), tok)
        tok = p.advance()
        self.assertEqual(AssignTok(":=", Position(line=4, col=16)), tok)
        tok = p.advance()
        self.assertEqual(IntConstTok(2, Position(line=4, col=33)), tok)
        tok = p.peek(18)
        self.assertEqual(IdTok("c", Position(line=7, col=9)), tok)
        tok = p.advance()
        self.assertEqual(SemiTok(";", Position(line=4, col=34)), tok)
        tok = p.peek(17)
        self.assertEqual(IdTok("c", Position(line=7, col=9)), tok)
        tok = p.peek(18)
        self.assertEqual(AssignTok(":=", Position(line=7, col=11)), tok)
        tok = p.advance()
        self.assertEqual(IdTok("_a", Position(line=5, col=9)), tok)
        tok = p.peek(30)
        self.assertEqual(EofTok("", Position(line=12, col=1)), tok)

        with self.assertRaises(StopIteration):
            p.peek(31)

        tok = p.peek()
        self.assertEqual(AssignTok(":=", Position(line=5, col=12)), tok)
        tok = p.peek()
        self.assertEqual(AssignTok(":=", Position(line=5, col=12)), tok)

    def test_proccall(self):
        ast = make_prog_ast_from_file("part16.pas")
        act = [str(n) for n in list(PostOrderIter(ast))]
        exp = [
            'Var(value=a)',
            'Type(value=integer)',
            'Param()',
            'Var(value=b)',
            'Type(value=integer)',
            'Param()',
            'Var(value=x)',
            'Type(value=integer)',
            'VarDecl()',
            'Var(value=x)',
            'Var(value=a)',
            'Var(value=b)',
            'Add(value=+)',
            'Num(value=2)',
            'Mul(value=*)',
            'Assign(value=:=)',
            'NoOp()',
            'Compound()',
            'Block()',
            'ProcDecl(name=Alpha)',
            'Num(value=3)',
            'Num(value=5)',
            'Add(value=+)',
            'Num(value=7)',
            'ProcCall(proc_name=Alpha)',
            'NoOp()',
            'Compound()',
            'Block()',
            'Program(name=Main)'
        ]
        self.assertEqual(exp, act)

    def test_bool(self):
        ast = make_prog_ast_from_file("bool_test.pas")
        exp = [
            'Var(value=foo)',
            'Type(value=boolean)',
            'VarDecl()',
            'Var(value=bar)',
            'Type(value=boolean)',
            'VarDecl()',
            'Var(value=foo)',
            'Bool(value=True)',
            'Assign(value=:=)',
            'Var(value=bar)',
            'Bool(value=False)',
            'Assign(value=:=)',
            'Compound()',
            'Block()',
            'Program(name=bool_test)'
        ]
        act = [str(n) for n in PostOrderIter(ast)]
        self.assertEqual(exp, act)

    def test_cond(self):
        ast = make_prog_ast_from_file("cond_test.pas")
        act = [str(n) for n in PostOrderIter(ast)]
        exp = [
            'Var(value=foo_res)',
            'Type(value=integer)',
            'VarDecl()',
            'Var(value=foo_res2)',
            'Type(value=integer)',
            'VarDecl()',
            'Var(value=foo_res3)',
            'Type(value=integer)',
            'VarDecl()',
            'Var(value=foo_t)',
            'Type(value=boolean)',
            'VarDecl()',
            'Var(value=foo_f)',
            'Type(value=boolean)',
            'VarDecl()',
            'Var(value=foo_res)',
            'Num(value=0)',
            'Assign(value=:=)',
            'Var(value=foo_res2)',
            'Num(value=0)',
            'Assign(value=:=)',
            'Var(value=foo_res3)',
            'Num(value=0)',
            'Assign(value=:=)',
            'Var(value=foo_t)',
            'Bool(value=True)',
            'Assign(value=:=)',
            'Var(value=foo_f)',
            'Bool(value=False)',
            'Assign(value=:=)',
            'Bool(value=True)',
            'Var(value=foo_res3)',
            'Var(value=foo_res3)',
            'Num(value=1)',
            'Add(value=+)',
            'Assign(value=:=)',
            'NoOp()',
            'Branch()',
            'Bool(value=False)',
            'Var(value=foo_res)',
            'Num(value=0)',
            'Assign(value=:=)',
            'Num(value=1)',
            'Num(value=2)',
            'Mul(value=*)',
            'Num(value=3)',
            'Num(value=4)',
            'Add(value=+)',
            'LogicalAnd(value=&&)',
            'Bool(value=False)',
            'LogicalOr(value=||)',
            'Var(value=foo_res)',
            'Num(value=1)',
            'Assign(value=:=)',
            'Var(value=foo_res2)',
            'Num(value=1)',
            'Assign(value=:=)',
            'Branch()',
            'Branch()',
            'Bool(value=True)',
            'Bool(value=False)',
            'Bool(value=False)',
            'LogicalAnd(value=&&)',
            'LogicalOr(value=||)',
            'Var(value=foo_res)',
            'Var(value=foo_res)',
            'Num(value=2)',
            'Add(value=+)',
            'Assign(value=:=)',
            'Var(value=foo_res)',
            'Var(value=foo_res)',
            'Num(value=4)',
            'Add(value=+)',
            'Assign(value=:=)',
            'Compound()',
            'Var(value=foo_res2)',
            'Var(value=foo_res2)',
            'Num(value=2)',
            'Add(value=+)',
            'Assign(value=:=)',
            'Var(value=foo_res2)',
            'Var(value=foo_res2)',
            'Num(value=4)',
            'Add(value=+)',
            'Assign(value=:=)',
            'Compound()',
            'Branch()',
            'Var(value=foo_f)',
            'Var(value=foo_res)',
            'Var(value=foo_res)',
            'Num(value=8)',
            'Add(value=+)',
            'Assign(value=:=)',
            'Var(value=foo_res2)',
            'Var(value=foo_res2)',
            'Num(value=8)',
            'Add(value=+)',
            'Assign(value=:=)',
            'Branch()',
            'Var(value=foo_t)',
            'Num(value=1)',
            'Num(value=2)',
            'Mul(value=*)',
            'LogicalAnd(value=&&)',
            'Bool(value=False)',
            'LogicalAnd(value=&&)',
            'Var(value=foo_res)',
            'Var(value=foo_res)',
            'Num(value=16)',
            'Add(value=+)',
            'Assign(value=:=)',
            'Var(value=foo_res2)',
            'Var(value=foo_res2)',
            'Num(value=16)',
            'Add(value=+)',
            'Assign(value=:=)',
            'Branch()',
            'NoOp()',
            'Compound()',
            'Block()',
            'Program(name=cond_test)'
        ]
        self.assertEqual(exp, act)


class InterpreterTestCase(unittest.TestCase):
    def setUp(self):
        self.interpreter = Interpreter(free_frames=True)

    def make_compound_ast(self, txt):
        p = make_parser(txt)
        return p.parse_compound()

    def test_expression1(self):
        ast = make_expr_ast('3')
        result = self.interpreter.interpret(ast)
        self.assertEqual(result, 3)

    def test_expression2(self):
        ast = make_expr_ast('2 + 7 * 4')
        result = self.interpreter.interpret(ast)
        self.assertEqual(result, 30)

    def test_expression3(self):
        ast = make_expr_ast('7 - 8 div 4')
        result = self.interpreter.interpret(ast)
        self.assertEqual(result, 5)

    def test_expression4(self):
        ast = make_expr_ast('14 + 2 * 3 - 6 div 2')
        result = self.interpreter.interpret(ast)
        self.assertEqual(result, 17)

    def test_expression5(self):
        ast = make_expr_ast('7 + 3 * (10 div (12 div (3 + 1) - 1))')
        result = self.interpreter.interpret(ast)
        self.assertEqual(result, 22)

    def test_expression6(self):
        ast = make_expr_ast(
            '7 + 3 * (10 div (12 div (3 + 1) - 1)) div (2 + 3) - 5 - 3 + (8)'
        )
        result = self.interpreter.interpret(ast)
        self.assertEqual(result, 10)

    def test_expression7(self):
        ast = make_expr_ast('7 + (((3 + 2)))')
        result = self.interpreter.interpret(ast)
        self.assertEqual(result, 12)

    def test_expression8(self):
        ast = make_expr_ast('- 3')
        result = self.interpreter.interpret(ast)
        self.assertEqual(result, -3)

    def test_expression9(self):
        ast = make_expr_ast('+ 3')
        result = self.interpreter.interpret(ast)
        self.assertEqual(result, 3)

    def test_expression10(self):
        ast = make_expr_ast('5 - - - + - 3')
        result = self.interpreter.interpret(ast)
        self.assertEqual(result, 8)

    def test_expression11(self):
        ast = make_expr_ast('5 - - - + - (3 + 4) - +2')
        result = self.interpreter.interpret(ast)
        self.assertEqual(result, 10)

    def test_no_expression(self):
        with self.assertRaises(ParserError) as e:
            ast = make_expr_ast('   ')
        errcode = e.exception.error_code
        errtok = e.exception.token
        self.assertEqual(errcode, ErrorCode.UNEXPECTED_TOKEN)
        self.assertEqual(errtok, EofTok("", Position(line=1, col=4)))

    def test_expression_invalid_syntax1(self):
        with self.assertRaises(ParserError):
            ast = make_expr_ast('10 *')

    def test_expression_invalid_syntax2(self):
        with self.assertRaises(ParserError):
            ast = make_expr_ast('1 (1 + 2)')

    def test_expression_compound(self):
        ast = make_compound_ast_from_file("foo.pas")
        self.interpreter.interpret_compound(ast)
        mem = self.interpreter.rts._free_frames[-1]
        self.assertEqual(
            {
                'number': 2,
                '_a': 2,
                'b': 25,
                'c': 27,
                'x': 11
            },
            mem
        )

    def test_part10_program(self):
        with open("part10.pas") as pas:
            txt = pas.read()

        ast = make_prog_ast(txt)
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        self.assertEqual(len(self.interpreter.rts._free_frames[-1]), 3)
        self.assertEqual(self.interpreter.rts._free_frames[-1]['a'], 2)
        self.assertEqual(self.interpreter.rts._free_frames[-1]['b'], 25)

        self.assertTrue(
            Float(self.interpreter.rts._free_frames[-1]['y']).eq(0.01,
                                                                 Float(5.99))
        )

    def test_part12_program(self):
        ast = make_prog_ast_from_file("part12.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        self.assertEqual(len(self.interpreter.rts._free_frames[-1]), 1)
        self.assertEqual(self.interpreter.rts._free_frames[-1]['a'], 10)

    def test_bool_expr(self):
        ast = make_prog_ast_from_file("bool_test2.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        self.assertEqual(len(self.interpreter.rts._free_frames[-1]), 5)
        self.assertEqual(self.interpreter.rts._free_frames[-1]['foo'], True)
        self.assertEqual(self.interpreter.rts._free_frames[-1]['res'], True)
        self.assertEqual(self.interpreter.rts._free_frames[-1]['res2'], True)
        self.assertEqual(self.interpreter.rts._free_frames[-1]['res3'], False)
        self.assertEqual(self.interpreter.rts._free_frames[-1]['res4'], True)

    def test_cond(self):
        ast = make_prog_ast_from_file("cond_test.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        self.assertEqual(5, len(self.interpreter.rts._free_frames[-1]))
        self.assertEqual(7, self.interpreter.rts._free_frames[-1]['foo_res'])
        self.assertEqual(24, self.interpreter.rts._free_frames[-1]['foo_res2'])
        self.assertEqual(1, self.interpreter.rts._free_frames[-1]['foo_res3'])

    def test_bitwise_or(self):
        ast = make_prog_ast_from_file("bitwise_or.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        mem = self.interpreter.rts._free_frames[-1]
        self.assertEqual({'bar': 1,
                          'baz': 3,
                          'foo': 5,
                          'res': 9}, mem)

    def test_bitwise_xor(self):
        ast = make_prog_ast_from_file("bitwise_xor.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        mem = self.interpreter.rts._free_frames[-1]
        self.assertEqual({'res': 11}, mem)

    def test_bitwise_and(self):
        ast = make_prog_ast_from_file("bitwise_and.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        mem = self.interpreter.rts._free_frames[-1]
        self.assertEqual({'res': 5}, mem)

    def test_equal_not_equal(self):
        ast = make_prog_ast_from_file("equal_not_equal.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        mem = self.interpreter.rts._free_frames[-1]
        self.assertEqual({'res': 2, 'res2': 2}, mem)

    def test_greater_greater_equal_less_less_equal(self):
        ast = make_prog_ast_from_file(
            "greater_greater_equal_less_less_equal.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        mem = self.interpreter.rts._free_frames[-1]
        self.assertEqual({'res': False,
                          'res10': 3,
                          'res11': False,
                          'res12': 2,
                          'res13': True,
                          'res14': 3,
                          'res15': True,
                          'res16': 3,
                          'res2': 2,
                          'res3': True,
                          'res4': 3,
                          'res5': True,
                          'res6': 2,
                          'res7': True,
                          'res8': 3,
                          'res9': True}, mem)

    def test_left_right_shift(self):
        ast = make_prog_ast_from_file("left_right_shift.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        mem = self.interpreter.rts._free_frames[-1]
        self.assertEqual({'res': 16,
                          'res2': 1,
                          'res3': 4,
                          'res4': 5}, mem)

    def test_modulus(self):
        ast = make_prog_ast_from_file("modulus.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        mem = self.interpreter.rts._free_frames[-1]
        self.assertEqual({'res': 4,
                          'res2': 3}, mem)

    def test_logical_not(self):
        ast = make_prog_ast_from_file("logical_not.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        mem = self.interpreter.rts._free_frames[-1]
        self.assertEqual({'res': 4,
                          'res2': False,
                          'res3': True,
                          'res4': True}, mem)

    def test_bitwise_not(self):
        ast = make_prog_ast_from_file("bitwise_not.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        mem = self.interpreter.rts._free_frames[-1]
        self.assertEqual({'res': 6,
                          'res2': -8,
                          'res3': -1,
                          'res4': -1,
                          'res5': 3,
                          'res6': -3}, mem)

    def test_bad_operand(self):
        ast = make_prog_ast_from_file("bad_operand.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        with self.assertRaises(InterpreterError) as e:
            self.interpreter.interpret(ast)
        exc = e.exception
        self.assertEqual(
            "InterpreterError: Unexpected operand type -> "
            "BitwiseNotTok(~, Position(line=6, col=10)). val is "
            "type float which is not a subtype of any of "
            "['int', 'bool']",
            exc.message
        )

    def test_expr_stress(self):
        ast = make_prog_ast_from_file("expr_stress.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        mem = self.interpreter.rts._free_frames[-1]
        self.assertEqual({'res': -8}, mem)

    def test_proc_call(self):
        ast = make_prog_ast_from_file("part18.pas")
        with SemanticAnalyzer() as lyz:
            lyz.analyze(ast)
        self.interpreter.interpret(ast)
        alpha_mem = self.interpreter.rts._free_frames[-2]
        self.assertEqual({'a': 8, 'b': 7, 'x': 30}, alpha_mem._members)
        self.assertEqual("Alpha", alpha_mem.name)
        self.assertEqual(FrameType.PROCEDURE, alpha_mem.ty)
        self.assertEqual(2, alpha_mem.nesting_lv)


class SemanticAnalyzerTestCase(unittest.TestCase):
    def _get_scopes_from_str(self, s):
        return [el for el in s.splitlines() if re.match(r"\s*\(name", el)]

    def test_builder(self):
        ast = make_prog_ast_from_file("part11.pas")

        with StringBufferLoggers() as sbs:
            with SemanticAnalyzer() as lyz:
                lyz.visit(ast)

        self.assertNotEqual(sbs.scope.getvalue(), "")
        scopes = self._get_scopes_from_str(sbs.scope.getvalue())
        self.assertEqual(len(scopes), 2)
        self.assertEqual(
            "(" + \
            "name: Part11, " + \
            "level: 1, " + \
            "encl_scope: builtins, " + \
            "symbols: ['<x:INTEGER>', '<y:REAL>']" + \
            ")",
            scopes[0],
        )
        self.assertEqual(
            "(" + \
            "name: builtins, " + \
            "level: 0, " + \
            "encl_scope: None, " + \
            "symbols: ['INTEGER', 'REAL', 'BOOLEAN', " + \
            "'ProcSymbol(name=Part11, params=[])']" + \
            ")",
            scopes[1],
        )

    def test_builder_name_error(self):
        ast = make_prog_ast_from_file("name_err.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(SemanticError) as e:
            lyz.visit(ast)
        self.assertEqual(
            IdTok("b", Position(line=6, col=13)),
            e.exception.token
        )
        self.assertEqual(ErrorCode.ID_NOT_FOUND, e.exception.error_code)
        self.assertEqual(
            "SemanticError: Identifier not found "
            "-> IdTok(b, Position(line=6, col=13)). "
            "b is not declared",
            e.exception.message
        )

    def test_builder_name_error2(self):
        ast = make_prog_ast_from_file("name_err2.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(SemanticError) as e:
            lyz.visit(ast)
        self.assertEqual(ErrorCode.ID_NOT_FOUND, e.exception.error_code)
        self.assertEqual(IdTok("a", Position(7, 4)), e.exception.token)
        self.assertEqual(
            "SemanticError: Identifier not found -> "
            "IdTok(a, Position(line=7, col=4)). a is not declared",
            e.exception.message
        )

    def test_builder_part12(self):
        ast = make_prog_ast_from_file("part12.pas")

        with StringBufferLoggers() as sbs:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        scopes = self._get_scopes_from_str(sbs.scope.getvalue())

        self.assertEqual(len(scopes), 4)
        self.assertEqual(
            "(" + \
            "name: P2, " + \
            "level: 3, " + \
            "encl_scope: P1, " + \
            "symbols: ['<a:INTEGER>', '<z:INTEGER>']" + \
            ")",
            scopes[0]
        )
        self.assertEqual(
            "(" + \
            "name: P1, " + \
            "level: 2, " + \
            "encl_scope: Part12, " + \
            "symbols: [" + \
            "'<a:REAL>', " + \
            "'<k:INTEGER>', " + \
            "'ProcSymbol(name=P2, params=[])'" + \
            "]" + \
            ")",
            scopes[1]
        )
        self.assertEqual(
            "(" + \
            "name: Part12, " + \
            "level: 1, " + \
            "encl_scope: builtins, " + \
            "symbols: [" + \
            "'<a:INTEGER>', " + \
            "'ProcSymbol(name=P1, params=[])'" + \
            "]" + \
            ")",
            scopes[2]
        )
        self.assertEqual(
            "(" + \
            "name: builtins, " + \
            "level: 0, " + \
            "encl_scope: None, " + \
            "symbols: ['INTEGER', 'REAL', 'BOOLEAN', " + \
            "'ProcSymbol(name=Part12, params=[])']" + \
            ")",
            scopes[3]
        )

    def test_dup_var(self):
        ast = make_prog_ast_from_file("dup_var_err.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(SemanticError) as e:
            lyz.analyze(ast)
        sem_err = e.exception
        self.assertEqual(ErrorCode.DUPLICATE_ID, sem_err.error_code)
        self.assertEqual(IdTok("y", Position(3, 8)), sem_err.token)
        self.assertEqual(
            "SemanticError: Duplicate id found -> "
            "IdTok(y, Position(line=3, col=8))",
            sem_err.message
        )

    def test_part14_decl_only_chained_scope(self):
        ast = make_prog_ast_from_file("part14_decl_only.pas")

        with StringBufferLoggers() as sbs:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        actual = to_list_of_lines(sbs.scope.getvalue())

        expected = [
            "ENTER scope builtins",
            "Insert: INTEGER",
            "Insert: REAL",
            "Insert: BOOLEAN",
            "Insert: Main",
            "ENTER scope Main",
            "Lookup: real. (Scope name: Main)",
            "Lookup: real. (Scope name: builtins)",
            "Lookup: x. (Scope name: Main)",
            "Insert: x",
            "Lookup: real. (Scope name: Main)",
            "Lookup: real. (Scope name: builtins)",
            "Lookup: y. (Scope name: Main)",
            "Insert: y",
            "Insert: Alpha",
            "ENTER scope Alpha",
            "Lookup: integer. (Scope name: Alpha)",
            "Lookup: integer. (Scope name: Main)",
            "Lookup: integer. (Scope name: builtins)",
            "Lookup: a. (Scope name: Alpha)",
            "Insert: a",
            "Lookup: integer. (Scope name: Alpha)",
            "Lookup: integer. (Scope name: Main)",
            "Lookup: integer. (Scope name: builtins)",
            "Lookup: y. (Scope name: Alpha)",
            "Insert: y",
            "(name: Alpha, level: 2, " + \
            "encl_scope: Main, symbols: ['<a:INTEGER>', '<y:INTEGER>'])",
            "LEAVE scope Alpha",
            "(name: Main, level: 1, encl_scope: builtins, " + \
            "symbols: ['<x:REAL>', '<y:REAL>', " + \
            "\"ProcSymbol(name=Alpha, " + \
            "params=[VarSymbol(name='a', type='INTEGER')])\"])",
            "LEAVE scope Main",
            "(name: builtins, level: 0, encl_scope: None, " + \
            "symbols: ['INTEGER', 'REAL', 'BOOLEAN', " + \
            "'ProcSymbol(name=Main, params=[])'])",
            "LEAVE scope builtins"
        ]

        self.assertEqual(expected, actual)

    def test_part14_dup_param(self):
        ast = make_prog_ast_from_file("part14_dup_param.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(SemanticError) as e:
            lyz.analyze(ast)
        sem_err = e.exception
        self.assertEqual(ErrorCode.DUPLICATE_ID, sem_err.error_code)
        self.assertEqual(IdTok("a", Position(4, 33)), sem_err.token)
        self.assertEqual(
            "SemanticError: Duplicate id found -> "
            "IdTok(a, Position(line=4, col=33))",
            sem_err.message
        )

    def test_part14_sibling_scopes(self):
        ast = make_prog_ast_from_file(
            "part14_decl_only_sibling_scopes.pas")

        with StringBufferLoggers() as sbs:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        scopes = self._get_scopes_from_str(sbs.scope.getvalue())

        self.assertEqual(len(scopes), 4)
        self.assertEqual(
            scopes[0],
            "(" + \
            "name: AlphaA, " + \
            "level: 2, " + \
            "encl_scope: Main, " + \
            "symbols: ['<a:INTEGER>', '<y:INTEGER>']" + \
            ")"
        )
        self.assertEqual(
            scopes[1],
            "(" + \
            "name: AlphaB, " + \
            "level: 2, " + \
            "encl_scope: Main, " + \
            "symbols: ['<a:INTEGER>', '<b:INTEGER>']" + \
            ")"
        )
        self.assertEqual(
            scopes[2],
            "(" + \
            "name: Main, " + \
            "level: 1, " + \
            "encl_scope: builtins, " + \
            "symbols: [" + \
            "'<x:REAL>', " + \
            "'<y:REAL>', " + \
            "\"ProcSymbol(" + \
            "name=AlphaA, " + \
            "params=[VarSymbol(name='a', type='INTEGER')]" + \
            ")\", " + \
            "\"ProcSymbol(" + \
            "name=AlphaB, " + \
            "params=[VarSymbol(name='a', type='INTEGER')]" + \
            ")\"" + \
            "]" + \
            ")"
        )
        self.assertEqual(
            scopes[3],
            "(" + \
            "name: builtins, " + \
            "level: 0, " + \
            "encl_scope: None, " + \
            "symbols: ['INTEGER', 'REAL', 'BOOLEAN', " + \
            "'ProcSymbol(name=Main, params=[])']" + \
            ")",
        )

    def test_part14_var_ref(self):
        ast = make_prog_ast_from_file("part14_var_ref.pas")

        with StringBufferLoggers() as sbs:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        actual = to_list_of_lines(sbs.scope.getvalue())

        expect = [
            "ENTER scope builtins",
            "Insert: INTEGER",
            "Insert: REAL",
            "Insert: BOOLEAN",
            "Insert: Main",
            "ENTER scope Main",
            "Lookup: real. (Scope name: Main)",
            "Lookup: real. (Scope name: builtins)",
            "Lookup: x. (Scope name: Main)",
            "Insert: x",
            "Lookup: real. (Scope name: Main)",
            "Lookup: real. (Scope name: builtins)",
            "Lookup: y. (Scope name: Main)",
            "Insert: y",
            "Insert: Alpha",
            "ENTER scope Alpha",
            "Lookup: integer. (Scope name: Alpha)",
            "Lookup: integer. (Scope name: Main)",
            "Lookup: integer. (Scope name: builtins)",
            "Lookup: a. (Scope name: Alpha)",
            "Insert: a",
            "Lookup: integer. (Scope name: Alpha)",
            "Lookup: integer. (Scope name: Main)",
            "Lookup: integer. (Scope name: builtins)",
            "Lookup: y. (Scope name: Alpha)",
            "Insert: y",
            "Lookup: a. (Scope name: Alpha)",
            "Lookup: x. (Scope name: Alpha)",
            "Lookup: x. (Scope name: Main)",
            "Lookup: y. (Scope name: Alpha)",
            "Lookup: x. (Scope name: Alpha)",
            "Lookup: x. (Scope name: Main)",
            "(name: Alpha, level: 2, encl_scope: Main, " + \
            "symbols: ['<a:INTEGER>', '<y:INTEGER>'])",
            "LEAVE scope Alpha",
            "(name: Main, level: 1, encl_scope: builtins, " + \
            "symbols: ['<x:REAL>', '<y:REAL>', " + \
            "\"ProcSymbol(name=Alpha, " + \
            "params=[VarSymbol(name='a', type='INTEGER')])\"])",
            "LEAVE scope Main",
            "(name: builtins, level: 0, encl_scope: None, " + \
            "symbols: ['INTEGER', 'REAL', 'BOOLEAN', " + \
            "'ProcSymbol(name=Main, params=[])'])",
            "LEAVE scope builtins"
        ]

        self.assertEqual(expect, actual)

    def test_part16_argcheck(self):
        ast = make_prog_ast_from_file("part16_badarg.pas")

        with self.assertRaises(SemanticError) as e:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        exc = e.exception
        self.assertEqual(
            "SemanticError: Parameter list does not match "
            "declaration -> IdTok(alpha, Position(line=9, col=3)). "
            "Expected 2 args, got 0 args",
            exc.message
        )
        self.assertEqual(ErrorCode.BAD_PARAMS, exc.error_code)

        ast = make_prog_ast_from_file("part16_badarg2.pas")

        with self.assertRaises(SemanticError) as e:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        exc = e.exception
        self.assertEqual(
            "SemanticError: Proc not found -> "
            "IdTok(beta, Position(line=9, col=3)). "
            "<beta:INTEGER> is not a ProcSymbol",
            exc.message
        )
        self.assertEqual(ErrorCode.PROC_NOT_FOUND, exc.error_code)

        ast = make_prog_ast_from_file("part16_badarg3.pas")

        with self.assertRaises(SemanticError) as e:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        exc = e.exception
        self.assertEqual(
            "SemanticError: Parameter list does not match declaration "
            "-> IdTok(alpha, Position(line=9, col=3)). "
            "Expected 2 args, got 1 args",
            exc.message
        )
        self.assertEqual(ErrorCode.BAD_PARAMS, exc.error_code)

        ast = make_prog_ast_from_file("part16_badarg4.pas")

        with self.assertRaises(SemanticError) as e:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        exc = e.exception
        self.assertEqual(
            "SemanticError: Parameter list does not match declaration "
            "-> IdTok(alpha, Position(line=9, col=3)). "
            "Expected 2 args, got 3 args",
            exc.message
        )
        self.assertEqual(ErrorCode.BAD_PARAMS, exc.error_code)


class DecoSrcBuilderTestCase(unittest.TestCase):
    def test_deco_src_part11(self):
        ast = make_prog_ast_from_file("part11.pas")
        lyz = SemanticAnalyzer(s2s=True)
        lyz.analyze(ast)
        actual = lyz.deco_src()
        expect = \
            "program Part110;\n   var x1 : INTEGER0;\n   " + \
            "var y1 : REAL0;\n   begin\n   end.    {END OF Part11}"
        self.assertEqual(actual, expect)

    def test_deco_src_part14(self):
        ast = make_prog_ast_from_file("part14_s2s.pas")
        lyz = SemanticAnalyzer(s2s=True)
        lyz.analyze(ast)
        actual = lyz.deco_src()
        expect = 'program Main0;\n   var x1 : real0;\n   ' + \
                 'var y1 : real0;\n   var z1 : integer0;\n   ' + \
                 'var b1 : integer0;\n   var c1 : real0;\n   ' + \
                 'var d1 : integer0;\n   procedure Alpha1(' + \
                 'a2 : integer0);\n      ' + \
                 'var y2 : integer0;\n      begin\n      ' + \
                 '<x1:REAL0> := <a2:INTEGER0> + ' + \
                 '<x1:REAL0> * <y2:INTEGER0>\n      ' + \
                 '<x1:REAL0> := - <a2:INTEGER0>\n      ' + \
                 '<x1:REAL0> := + <a2:INTEGER0>\n      ' + \
                 '<x1:REAL0> := - + <a2:INTEGER0>\n      end;    ' + \
                 '{END OF Alpha}\n   begin\n   ' + \
                 '<z1:INTEGER0> := <z1:INTEGER0> - ' + \
                 '<d1:INTEGER0> / <b1:INTEGER0> DIV ' + \
                 '<c1:REAL0>\n   end.    ' + \
                 '{END OF Main}'
        self.assertEqual(actual, expect)

    def test_deco_src_part14_2(self):
        ast = make_prog_ast_from_file("part14_s2s_2.pas")
        lyz = SemanticAnalyzer(s2s=True)
        lyz.analyze(ast)
        actual = lyz.deco_src()
        expect = 'program Main0;\n   ' + \
                 'var x1 : real0;\n   ' + \
                 'var y1 : real0;\n   ' + \
                 'var z1 : integer0;\n   ' + \
                 'procedure AlphaA1(a2 : integer0);\n      ' + \
                 'var y2 : integer0;\n      ' + \
                 'begin\n      ' + \
                 '<x1:REAL0> := <a2:INTEGER0> + <x1:REAL0> ' + \
                 '+ <y2:INTEGER0>\n      ' + \
                 'end;    {END OF AlphaA}\n   ' + \
                 'procedure AlphaB1(a2 : integer0);\n      ' + \
                 'var b2 : integer0;\n      begin\n      ' + \
                 'end;    {END OF AlphaB}\n   begin\n   ' + \
                 'end.    {END OF Main}'
        self.assertEqual(actual, expect)

    def test_deco_src_part14_3(self):
        ast = make_prog_ast_from_file("part14_s2s_3.pas")
        lyz = SemanticAnalyzer(s2s=True)
        lyz.analyze(ast)
        actual = lyz.deco_src()
        expect = 'program Main0;\n   var b1 : real0;\n   ' + \
                 'var x1 : real0;\n   var y1 : real0;\n   ' + \
                 'var z1 : integer0;\n   procedure ' + \
                 'AlphaA1(a2 : integer0);\n      ' + \
                 'var b2 : integer0;\n      ' + \
                 'procedure Beta2(c3 : integer0);\n         ' + \
                 'var y3 : integer0;\n         ' + \
                 'procedure Gamma3(c4 : integer0);\n            ' + \
                 'var x4 : integer0;\n            begin\n            ' + \
                 '<x4:INTEGER0> := <a2:INTEGER0> + <b2:INTEGER0> + ' + \
                 '<c4:INTEGER0> + <x4:INTEGER0> + <y3:INTEGER0> + ' + \
                 '<z1:INTEGER0>\n            end;    ' + \
                 '{END OF Gamma}\n         begin\n         end;    ' + \
                 '{END OF Beta}\n      begin\n      end;    ' + \
                 '{END OF AlphaA}\n   procedure ' + \
                 'AlphaB1(a2 : integer0);\n      var c2 : real0;\n      ' + \
                 'begin\n      <c2:REAL0> := <a2:INTEGER0> + ' + \
                 '<b1:REAL0>\n      ' + \
                 'end;    {END OF AlphaB}\n   begin\n   end.    {END OF Main}'
        self.assertEqual(actual, expect)

    def test_deco_src_part10(self):
        ast = make_prog_ast_from_file("part10.pas")
        lyz = SemanticAnalyzer(s2s=True)
        lyz.analyze(ast)
        actual = lyz.deco_src()
        expect = 'program Part10AST0;\n' + \
                 '   var a1 : INTEGER0;\n' + \
                 '   var b1 : INTEGER0;\n' + \
                 '   var y1 : REAL0;\n' + \
                 '   begin\n' + \
                 '   <a1:INTEGER0> := 2\n' + \
                 '   <b1:INTEGER0> := 10 * <a1:INTEGER0> + ' + \
                 '10 * <a1:INTEGER0> DIV 4\n' + \
                 '   <y1:REAL0> := 20 / 7 + 3.14\n' + \
                 '   end.    {END OF Part10AST}'
        self.assertEqual(actual, expect)


class FrameTest(unittest.TestCase):
    def test_frame(self):
        f = Frame("main", FrameType.PROGRAM, 1)
        f["foo"] = 1
        f["bar"] = 2

        self.assertEqual(f["foo"], 1)
        self.assertEqual(f["bar"], 2)

        with self.assertRaises(KeyError):
            f["baz"]

        self.assertEqual(f.get("baz"), None)

        self.assertEqual(f.name, "main")
        self.assertEqual(f.nesting_lv, 1)

        self.assertEqual(
            'Frame(\n  name: main,\n  ty: FrameType.PROGRAM,\n  '
            'nesting_lv: 1,\n  members: {\n    foo: 1,\n    '
            'bar: 2\n  }\n)',
            str(f)
        )

        self.assertEqual(
            'Frame(name: main, ty: FrameType.PROGRAM, '
            'nesting_lv: 1, members: {foo: 1, bar: 2})',
            repr(f)
        )

    def test_frame_ctor(self):
        f = Frame("main", FrameType.PROGRAM, 1, {"foo": 1, "bar": 2})
        self.assertEqual(f["foo"], 1)
        self.assertEqual(f["bar"], 2)


class RuntimeStackTest(unittest.TestCase):
    def test_callstack(self):
        main_frame = Frame("main", FrameType.PROGRAM, 1)
        main_frame["foo"] = 1
        main_frame["bar"] = 2
        func_frame = Frame("func", FrameType.PROCEDURE, 2)
        func_frame["foo"] = 100
        func_frame["baz"] = 3
        stack = RuntimeStack()
        stack.push(main_frame)
        stack.push(func_frame)
        self.assertEqual(
            'RuntimeStack(\n  frames: [\n    Frame(\n      '
            'name: func,\n      ty: FrameType.PROCEDURE,\n      '
            'nesting_lv: 2,\n      members: {\n        '
            'foo: 100,\n        baz: 3\n      }\n    ),\n    '
            'Frame(\n      name: main,\n      ty: FrameType.PROGRAM,'
            '\n      nesting_lv: 1,\n      members: {\n        '
            'foo: 1,\n        bar: 2\n      }\n    )\n  ]\n)',
            str(stack)
        )
        self.assertEqual(
            'RuntimeStack(frames: [Frame(name: func, '
            'ty: FrameType.PROCEDURE, nesting_lv: 2, '
            'members: {foo: 100, baz: 3}), '
            'Frame(name: main, ty: FrameType.PROGRAM, '
            'nesting_lv: 1, members: {foo: 1, bar: 2})])',
            repr(stack)
        )

    def test_callstack_2(self):
        stack = RuntimeStack()
        stack.push(Frame("main", FrameType.PROGRAM, 1, {"foo": 1}))
        stack.push(Frame("foo", FrameType.PROCEDURE, 2, {"foo": 100}))
        self.assertEqual(stack.peek()["foo"], 100)
        self.assertEqual(stack.pop()["foo"], 100)
        self.assertEqual(stack.peek()["foo"], 1)
        self.assertEqual(stack.pop()["foo"], 1)
        with self.assertRaises(IndexError):
            stack.pop()

    def test_callstack_emplace(self):
        stack = RuntimeStack()

        empty_blk = Block([], Compound())

        stack.emplace_frame(
            ProcSymbol("main", empty_blk, 1),
            members={"foo": 1}
        )
        stack.emplace_frame(
            ProcSymbol("foo", empty_blk, 2),
            members={"foo": 2}
        )

        self.assertEqual(
            'RuntimeStack(frames: [Frame(name: foo, '
            'ty: FrameType.PROCEDURE, nesting_lv: 2, '
            'members: {foo: 2}), Frame(name: main, '
            'ty: FrameType.PROGRAM, nesting_lv: 1, members: {foo: 1})])',
            repr(stack)
        )

    def test_callstack_broken_invariant(self):
        stack = RuntimeStack()

        f = Frame("main", FrameType.PROCEDURE, 1)
        with self.assertRaises(AssertionError):
            stack.push(f)

        with self.assertRaises(AssertionError):
            Frame("main", FrameType.PROGRAM, -1)

        f = Frame("main", FrameType.PROGRAM, 2)
        with self.assertRaises(AssertionError):
            stack.push(f)

        stack.push(Frame("main", FrameType.PROGRAM, 1))

        f = Frame("main", FrameType.PROCEDURE, 3)
        with self.assertRaises(AssertionError):
            stack.push(f)

        f = Frame("main", FrameType.PROGRAM, 2)
        with self.assertRaises(AssertionError):
            stack.push(f)

    def test_part17_pas(self):
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join("..", "spi.py"),
                "--verbose-stack",
                "part17.pas"
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        exp = [
            'ENTER: PROGRAM Main',
            'RuntimeStack(',
            '  frames: [',
            '    Frame(',
            '      name: Main,',
            '      ty: FrameType.PROGRAM,',
            '      nesting_lv: 1,',
            '      members: {',
            '      }',
            '    )',
            '  ]',
            ')',
            'LEAVE: PROGRAM Main',
            'RuntimeStack(',
            '  frames: [',
            '    Frame(',
            '      name: Main,',
            '      ty: FrameType.PROGRAM,',
            '      nesting_lv: 1,',
            '      members: {',
            '        y: 7,',
            '        x: 30',
            '      }',
            '    )',
            '  ]',
            ')'
        ]
        act = proc.stdout.splitlines()
        self.assertEqual(exp, act)


class MypyTest(unittest.TestCase):
    def test_typing(self):
        proc = subprocess.run(
            [sys.executable, "-m", "mypy", os.path.join("..", "spi.py")],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        self.assertEqual(0, proc.returncode, proc.stdout)


class Foo(unittest.TestCase):
    def test_foo(self):
        with open("part12.pas") as f:
            txt = f.read()

        lexer = Lexer(txt)
        for tok in lexer:
            print(tok)

        print()

        parser = Parser(lexer)
        ast = parser.parse()
        print(RenderTree(ast))

        print()

        with StringBufferLoggers() as sbs:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)
        print(sbs.scope.getvalue())


if __name__ == '__main__':
    unittest.main()
