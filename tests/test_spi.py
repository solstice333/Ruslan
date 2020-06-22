import sys
import os
import os.path
import unittest
import logging
import io
import re

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


class Float:
    def __init__(self, val):
       self.val = val

    def eq(self, eps, other):
        if isinstance(other, float):
            other = Float(other)
        return abs(self.val - other.val) < eps  


class LoggingToStrBuf():
    _initd: bool = False
    _sb = io.StringIO()
    _sh = logging.StreamHandler(_sb)

    @classmethod
    def _update_handler(cls):
        root_logger = logging.getLogger()
        root_logger.removeHandler(cls._sh)
        cls._sb = io.StringIO()
        cls._sh = logging.StreamHandler(cls._sb)
        root_logger.addHandler(cls._sh)

    @classmethod
    def basic_config(cls):
        logging.basicConfig(
            format="{message}", 
            style="{", 
            level=logging.INFO, 
            handlers=[cls._sh]
        )
        cls._initd = True

    def __enter__(self):
        cls = type(self)
        logging.disable(logging.NOTSET)

        if cls._initd:
            cls._update_handler()
        else:
            cls.basic_config()

        return cls._sb

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
        with self.assertRaises(RuntimeError) as e:
            p.parse_compound()

        exp_msg = \
            "Invalid syntax: found IdTok(_a). " + \
            "Expected semi-colon at 6:17"
        act_msg = e.exception.args[0]
        self.assertEqual(act_msg, exp_msg)

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


class InterpreterTestCase(unittest.TestCase):
    def setUp(self):
        self.interpreter = Interpreter()

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
        with self.assertRaises(RuntimeError) as e:
            ast = make_expr_ast('   ')
        msg = e.exception.args[0]
        self.assertEqual(
            msg, 
            "Invalid syntax: was expecting TokenType.ID, " + \
            "got TokenType.EOF at 1:4"
        )

    def test_expression_invalid_syntax1(self):
        with self.assertRaises(RuntimeError):
            ast = make_expr_ast('10 *')

    def test_expression_invalid_syntax2(self):
        with self.assertRaises(RuntimeError):
            ast = make_expr_ast('1 (1 + 2)')

    def test_expression_compound(self):
        with open("foo.pas") as foo_pas:
            txt = foo_pas.read()

        ast = self.make_compound_ast(txt)
        self.interpreter.interpret(ast)
        self.assertEqual(
            self.interpreter.GLOBAL_SCOPE,
            {
                'number': 2,
                '_a': 2,
                'b': 25,
                'c': 27,
                'x': 11
            }
        )

    def test_part10_program(self):
        with open("part10.pas") as pas:
            txt = pas.read()

        ast = make_prog_ast(txt)
        self.interpreter.interpret(ast)
        self.assertEqual(len(self.interpreter.GLOBAL_SCOPE), 3)
        self.assertEqual(self.interpreter.GLOBAL_SCOPE['a'], 2)
        self.assertEqual(self.interpreter.GLOBAL_SCOPE['b'], 25)

        self.assertTrue(
            Float(self.interpreter.GLOBAL_SCOPE['y']).eq(0.01, Float(5.99))
        )

    def test_part12_program(self):
        ast = make_prog_ast_from_file("part12.pas")
        self.interpreter.interpret(ast)
        self.assertEqual(len(self.interpreter.GLOBAL_SCOPE), 1)
        self.assertEqual(self.interpreter.GLOBAL_SCOPE['a'], 10)


class SemanticAnalyzerTestCase(unittest.TestCase):
    def _get_scopes_from_str(self, s):
        return [el for el in s.splitlines() if re.match(r"\s*\(name", el)]

    def test_builder(self):
        ast = make_prog_ast_from_file("part11.pas")

        with LoggingToStrBuf() as sb:
            with SemanticAnalyzer() as lyz:
                lyz.visit(ast)

        self.assertNotEqual(sb.getvalue(), "")
        scopes = self._get_scopes_from_str(sb.getvalue())
        self.assertEqual(len(scopes), 2)
        self.assertEqual(
            scopes[0], 
            "(" + \
                "name: global, " + \
                "level: 1, " + \
                "encl_scope: builtins, " + \
                "symbols: ['<x:INTEGER>', '<y:REAL>']" + \
            ")",
            f"got {scopes[1]}"
        )
        self.assertEqual(
            scopes[1], 
            "(" + \
                "name: builtins, " + \
                "level: 0, " + \
                "encl_scope: None, " + \
                "symbols: ['INTEGER', 'REAL', " + \
                    "'ProcSymbol(name=Part11, params=[])']" + \
            ")",
            f"got {scopes[0]}"
        )

    def test_builder_name_error(self):
        ast = make_prog_ast_from_file("name_err.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(NameError) as e:
            lyz.visit(ast)
        self.assertEqual(e.exception.args[0], "b is not declared at 6:13")

    def test_builder_name_error2(self):
        ast = make_prog_ast_from_file("name_err2.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(NameError) as e:
            lyz.visit(ast)
        self.assertEqual(e.exception.args[0], "a is not declared at 7:4")

    def test_builder_part12(self):
        ast = make_prog_ast_from_file("part12.pas")

        with LoggingToStrBuf() as sb:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        scopes = self._get_scopes_from_str(sb.getvalue())

        self.assertEqual(len(scopes), 4)
        self.assertEqual(
            scopes[0], 
            "(" + \
                "name: P2, " + \
                "level: 3, " + \
                "encl_scope: P1, " + \
                "symbols: ['<a:INTEGER>', '<z:INTEGER>']" + \
            ")"
        )
        self.assertEqual(
            scopes[1],
            "(" + \
                "name: P1, " + \
                "level: 2, " + \
                "encl_scope: global, " + \
                "symbols: [" + \
                    "'<a:REAL>', " + \
                    "'<k:INTEGER>', " + \
                    "'ProcSymbol(name=P2, params=[])'" + \
                "]" + \
            ")"
        )
        self.assertEqual(
            scopes[2],
            "(" + \
                "name: global, " + \
                "level: 1, " + \
                "encl_scope: builtins, " + \
                "symbols: [" + \
                    "'<a:INTEGER>', " + \
                    "'ProcSymbol(name=P1, params=[])'" + \
                "]" + \
            ")"
        )
        self.assertEqual(
            scopes[3], 
            "(" + \
                "name: builtins, " + \
                "level: 0, " + \
                "encl_scope: None, " + \
                "symbols: ['INTEGER', 'REAL', " + \
                    "'ProcSymbol(name=Part12, params=[])']" + \
            ")",
        )

    def test_dup_var(self):
        ast = make_prog_ast_from_file("dup_var_err.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(NameError) as e:
            lyz.analyze(ast)
        self.assertEqual(
            e.exception.args[0], "duplicate identifier y found at 3:8")

    def test_part14_decl_only_chained_scope(self):
        ast = make_prog_ast_from_file("part14_decl_only.pas")

        with LoggingToStrBuf() as sb:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        actual = listify_str(sb.getvalue())

        expected = [
            "ENTER scope builtins",
            "Insert: INTEGER",
            "Insert: REAL",
            "Insert: Main",
            "ENTER scope global",
            "Lookup: real. (Scope name: global)",
            "Lookup: real. (Scope name: builtins)",
            "Lookup: x. (Scope name: global)",
            "Insert: x",
            "Lookup: real. (Scope name: global)",
            "Lookup: real. (Scope name: builtins)",
            "Lookup: y. (Scope name: global)",
            "Insert: y",
            "Insert: Alpha",
            "ENTER scope Alpha",
            "Lookup: integer. (Scope name: Alpha)",
            "Lookup: integer. (Scope name: global)",
            "Lookup: integer. (Scope name: builtins)",
            "Lookup: a. (Scope name: Alpha)",
            "Insert: a",
            "Lookup: integer. (Scope name: Alpha)",
            "Lookup: integer. (Scope name: global)",
            "Lookup: integer. (Scope name: builtins)",
            "Lookup: y. (Scope name: Alpha)",
            "Insert: y",
            "(name: Alpha, level: 2, " + \
                "encl_scope: global, symbols: ['<a:INTEGER>', '<y:INTEGER>'])",
            "LEAVE scope Alpha",
            "(name: global, level: 1, encl_scope: builtins, " + \
                "symbols: ['<x:REAL>', '<y:REAL>', " + \
                "\"ProcSymbol(name=Alpha, " + \
                    "params=[VarSymbol(name='a', type='INTEGER')])\"])",
            "LEAVE scope global",
            "(name: builtins, level: 0, encl_scope: None, " + \
                "symbols: ['INTEGER', 'REAL', " + \
                "'ProcSymbol(name=Main, params=[])'])",
            "LEAVE scope builtins"
        ]

        self.assertEqual(actual, expected)


    def test_part14_dup_param(self):
        ast = make_prog_ast_from_file("part14_dup_param.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(NameError) as e:
            lyz.analyze(ast)
        self.assertEqual(
            e.exception.args[0], "duplicate identifier a found 4:33")

    def test_part14_sibling_scopes(self):
        ast = make_prog_ast_from_file(
            "part14_decl_only_sibling_scopes.pas")

        with LoggingToStrBuf() as sb:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        scopes = self._get_scopes_from_str(sb.getvalue())

        self.assertEqual(len(scopes), 4)
        self.assertEqual(
            scopes[0], 
            "(" + \
                "name: AlphaA, " + \
                "level: 2, " + \
                "encl_scope: global, " + \
                "symbols: ['<a:INTEGER>', '<y:INTEGER>']" + \
            ")"
        )
        self.assertEqual(
            scopes[1], 
            "(" + \
                "name: AlphaB, " + \
                "level: 2, " + \
                "encl_scope: global, " + \
                "symbols: ['<a:INTEGER>', '<b:INTEGER>']" + \
            ")"
        )
        self.assertEqual(
            scopes[2], 
            "(" + \
                "name: global, " + \
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
                "symbols: ['INTEGER', 'REAL', " + \
                    "'ProcSymbol(name=Main, params=[])']" + \
            ")",
        )

    def test_part14_var_ref(self):
        ast = make_prog_ast_from_file("part14_var_ref.pas")

        with LoggingToStrBuf() as sb:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        actual = listify_str(sb.getvalue())

        expect = [
            "ENTER scope builtins",
            "Insert: INTEGER",
            "Insert: REAL",
            "Insert: Main",
            "ENTER scope global",
            "Lookup: real. (Scope name: global)",
            "Lookup: real. (Scope name: builtins)",
            "Lookup: x. (Scope name: global)",
            "Insert: x",
            "Lookup: real. (Scope name: global)",
            "Lookup: real. (Scope name: builtins)",
            "Lookup: y. (Scope name: global)",
            "Insert: y",
            "Insert: Alpha",
            "ENTER scope Alpha",
            "Lookup: integer. (Scope name: Alpha)",
            "Lookup: integer. (Scope name: global)",
            "Lookup: integer. (Scope name: builtins)",
            "Lookup: a. (Scope name: Alpha)",
            "Insert: a",
            "Lookup: integer. (Scope name: Alpha)",
            "Lookup: integer. (Scope name: global)",
            "Lookup: integer. (Scope name: builtins)",
            "Lookup: y. (Scope name: Alpha)",
            "Insert: y",
            "Lookup: a. (Scope name: Alpha)",
            "Lookup: x. (Scope name: Alpha)",
            "Lookup: x. (Scope name: global)",
            "Lookup: y. (Scope name: Alpha)",
            "Lookup: x. (Scope name: Alpha)",
            "Lookup: x. (Scope name: global)",
            "(name: Alpha, level: 2, encl_scope: global, " + \
                "symbols: ['<a:INTEGER>', '<y:INTEGER>'])",
            "LEAVE scope Alpha",
            "(name: global, level: 1, encl_scope: builtins, " + \
                "symbols: ['<x:REAL>', '<y:REAL>', " + \
                "\"ProcSymbol(name=Alpha, " + \
                "params=[VarSymbol(name='a', type='INTEGER')])\"])",
            "LEAVE scope global",
            "(name: builtins, level: 0, encl_scope: None, " + \
                "symbols: ['INTEGER', 'REAL', " + \
                "'ProcSymbol(name=Main, params=[])'])",
            "LEAVE scope builtins"
        ]

        errmsg = prettify_strlist(sb.getvalue())
        self.assertEqual(actual, expect, f"got \n{errmsg}")


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
            'var d1 : integer0;\n   procedure Alpha1(a2 : integer0);\n      ' + \
            'var y2 : integer0;\n      begin\n      ' + \
            '<x1:REAL0> := <a2:INTEGER0> + <x1:REAL0> * <y2:INTEGER0>\n      ' + \
            '<x1:REAL0> := - <a2:INTEGER0>\n      ' + \
            '<x1:REAL0> := + <a2:INTEGER0>\n      ' + \
            '<x1:REAL0> := - + <a2:INTEGER0>\n      end;    ' + \
            '{END OF Alpha}\n   begin\n   ' + \
            '<z1:INTEGER0> := <z1:INTEGER0> - ' + \
            '<d1:INTEGER0> / <b1:INTEGER0> DIV <c1:REAL0>\n   end.    ' + \
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
            '<x1:REAL0> := <a2:INTEGER0> + <x1:REAL0> + <y2:INTEGER0>\n      ' + \
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
            'begin\n      <c2:REAL0> := <a2:INTEGER0> + <b1:REAL0>\n      ' + \
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

        with LoggingToStrBuf() as sb:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)
        print(sb.getvalue())

def listify_str(s):
    return s.strip().splitlines()

def prettify_strlist(s):
    pretty_s = ""
    ls = s.strip().split("\n")
    pretty_s += "[\n"
    for i, elem in enumerate(ls):
        comma = "" if i >= len(ls) - 1 else ","
        elem = re.sub(r"\"", "\\\"", elem)
        pretty_s += f"    \"{elem}\"{comma}\n"
    pretty_s += "]\n"
    return pretty_s

def print_str_as_list_of_str(s):
    print(prettify_strlist(s))

def make_parser(text):
    lexer = Lexer(text)
    return Parser(lexer)

def make_expr_ast(txt):
    p = make_parser(txt)
    return p.parse_expr()

def make_prog_ast(txt):
    parser = make_parser(txt)
    return parser.parse()

def make_prog_ast_from_file(path):
    with open(path) as f:
        txt = f.read()
    return make_prog_ast(txt)

LoggingToStrBuf.basic_config()
if __name__ == '__main__':
    unittest.main()
