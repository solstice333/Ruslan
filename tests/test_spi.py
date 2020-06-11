import sys
import os
import os.path
import unittest
import logging
import io
import re

from anytree import PostOrderIter, RenderTree

sys.path.append(os.path.realpath(".."))
from spi import TokenType, Token, Lexer, Parser, Interpreter, SemanticAnalyzer


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
        self.assertEqual(token.value, '[dD][iI][vV]')

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
        res_kw_dict = Lexer._RES_KW_TO_TID_INFO()

        self.assertEqual(len(tokens), 9)

        self.assertEqual(
            tokens[0], Token(TokenType.BEGIN, "[bB][eE][gG][iI][nN]"))
        self.assertEqual(tokens[1], Token(TokenType.ID, "a"))
        self.assertEqual(tokens[2], Token(TokenType.ASSIGN, ":="))
        self.assertEqual(tokens[3], Token(TokenType.INT_CONST, 0))
        self.assertEqual(tokens[4], Token(TokenType.SEMI, ";"))
        self.assertEqual(tokens[5], Token(TokenType.END, "[eE][nN][dD]"))
        self.assertEqual(tokens[6], Token(TokenType.DOT, "."))
        self.assertEqual(tokens[7], Token(TokenType.EOF, ""))
        self.assertEqual(tokens[8], Token(TokenType.EOF, None))

        self.assertIs(tokens[0], res_kw_dict['BEGIN'].token)
        self.assertIs(tokens[5], res_kw_dict['END'].token)

    def test_lexer_res_kw_multi_char_var(self):
        lexer = self.makeLexer("BEGIN foo_bar123 := 0; END.")
        tokens = list(iter(lexer))
        res_kw_dict = Lexer._RES_KW_TO_TID_INFO()

        self.assertEqual(len(tokens), 9)

        self.assertEqual(
            tokens[0], Token(TokenType.BEGIN, "[bB][eE][gG][iI][nN]"))
        self.assertEqual(tokens[1], Token(TokenType.ID, "foo_bar123"))
        self.assertEqual(tokens[2], Token(TokenType.ASSIGN, ":="))
        self.assertEqual(tokens[3], Token(TokenType.INT_CONST, 0))
        self.assertEqual(tokens[4], Token(TokenType.SEMI, ";"))
        self.assertEqual(tokens[5], Token(TokenType.END, "[eE][nN][dD]"))
        self.assertEqual(tokens[6], Token(TokenType.DOT, "."))
        self.assertEqual(tokens[7], Token(TokenType.EOF, ""))
        self.assertEqual(tokens[8], Token(TokenType.EOF, None))

        self.assertIs(tokens[0], res_kw_dict['BEGIN'].token)
        self.assertIs(tokens[5], res_kw_dict['END'].token)


class ParserTestCase(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_parser1(self):
        p = make_parser("BEGIN x := 11; y := 2 + x END.")
        ast = p.parse_compound()
        act = str(list(PostOrderIter(ast)))
        exp = "[" + \
            "Var(Token(TokenType.ID, x)), " + \
            "Num(Token(TokenType.INT_CONST, 11)), " + \
            "Assign(Token(TokenType.ASSIGN, :=)), " +\
            "Var(Token(TokenType.ID, y)), " + \
            "Num(Token(TokenType.INT_CONST, 2)), " + \
            "Var(Token(TokenType.ID, x)), " + \
            "Add(Token(TokenType.ADD, +)), " +\
            "Assign(Token(TokenType.ASSIGN, :=)), " +\
            "Compound(Token(TokenType.EOF, None))" +\
        "]"
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
        act = str(list(PostOrderIter(ast)))
        exp = "[" + \
            "Var(Token(TokenType.ID, number)), " + \
            "Num(Token(TokenType.INT_CONST, 2)), " + \
            "Assign(Token(TokenType.ASSIGN, :=)), " + \
            "Var(Token(TokenType.ID, a)), " + \
            "Var(Token(TokenType.ID, number)), " + \
            "Assign(Token(TokenType.ASSIGN, :=)), " + \
            "Var(Token(TokenType.ID, b)), " + \
            "Num(Token(TokenType.INT_CONST, 10)), " + \
            "Var(Token(TokenType.ID, a)), " + \
            "Mul(Token(TokenType.MUL, *)), " + \
            "Num(Token(TokenType.INT_CONST, 10)), " + \
            "Var(Token(TokenType.ID, number)), " + \
            "Mul(Token(TokenType.MUL, *)), " + \
            "Num(Token(TokenType.INT_CONST, 4)), " + \
            "IntDiv(Token(TokenType.INT_DIV, [dD][iI][vV])), " + \
            "Add(Token(TokenType.ADD, +)), " + \
            "Assign(Token(TokenType.ASSIGN, :=)), " + \
            "Var(Token(TokenType.ID, c)), " + \
            "Var(Token(TokenType.ID, a)), " + \
            "Var(Token(TokenType.ID, b)), " + \
            "Neg(Token(TokenType.SUB, -)), " + \
            "Sub(Token(TokenType.SUB, -)), " + \
            "Assign(Token(TokenType.ASSIGN, :=)), " + \
            "Compound(Token(TokenType.EOF, None)), " + \
            "Var(Token(TokenType.ID, x)), " + \
            "Num(Token(TokenType.INT_CONST, 11)), " + \
            "Assign(Token(TokenType.ASSIGN, :=)), " + \
            "NoOp(Token(TokenType.EOF, None)), " + \
            "Compound(Token(TokenType.EOF, None))" + \
        "]"

        self.assertEqual(act, exp)

    def test_parser3(self):
        with open("tests/part10.pas") as f:
            p = make_parser(f.read())
        ast = p.parse()
        act = str(list(PostOrderIter(ast)))
        exp = "[" + \
            "Var(Token(TokenType.ID, a)), " + \
            "Type(Token(TokenType.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TokenType.EOF, None)), " + \
            "Var(Token(TokenType.ID, b)), " + \
            "Type(Token(TokenType.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TokenType.EOF, None)), " + \
            "Var(Token(TokenType.ID, y)), " + \
            "Type(Token(TokenType.REAL, [rR][eE][aA][lL])), " + \
            "VarDecl(Token(TokenType.EOF, None)), " + \
            "Var(Token(TokenType.ID, a)), " + \
            "Num(Token(TokenType.INT_CONST, 2)), " + \
            "Assign(Token(TokenType.ASSIGN, :=)), " + \
            "Var(Token(TokenType.ID, b)), " + \
            "Num(Token(TokenType.INT_CONST, 10)), " + \
            "Var(Token(TokenType.ID, a)), " + \
            "Mul(Token(TokenType.MUL, *)), " + \
            "Num(Token(TokenType.INT_CONST, 10)), " + \
            "Var(Token(TokenType.ID, a)), " + \
            "Mul(Token(TokenType.MUL, *)), " + \
            "Num(Token(TokenType.INT_CONST, 4)), " + \
            "IntDiv(Token(TokenType.INT_DIV, [dD][iI][vV])), " + \
            "Add(Token(TokenType.ADD, +)), " + \
            "Assign(Token(TokenType.ASSIGN, :=)), " + \
            "Var(Token(TokenType.ID, y)), " + \
            "Num(Token(TokenType.INT_CONST, 20)), " + \
            "Num(Token(TokenType.INT_CONST, 7)), " + \
            "FloatDiv(Token(TokenType.FLOAT_DIV, /)), " + \
            "Num(Token(TokenType.REAL_CONST, 3.14)), " + \
            "Add(Token(TokenType.ADD, +)), " + \
            "Assign(Token(TokenType.ASSIGN, :=)), " + \
            "NoOp(Token(TokenType.EOF, None)), " + \
            "Compound(Token(TokenType.EOF, None)), " + \
            "Block(Token(TokenType.EOF, None)), " + \
            "Program(Token(TokenType.EOF, None))" + \
        "]"

        self.assertEqual(act, exp)

    def test_parser_proc(self):
        with open("tests/part12.pas") as f:
            p = make_parser(f.read())

        ast = p.parse()
        act = str(list(PostOrderIter(ast)))
        exp = "[" + \
            "Var(Token(TokenType.ID, a)), " + \
            "Type(Token(TokenType.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TokenType.EOF, None)), " + \
            "Var(Token(TokenType.ID, a)), " + \
            "Type(Token(TokenType.REAL, [rR][eE][aA][lL])), " + \
            "VarDecl(Token(TokenType.EOF, None)), " + \
            "Var(Token(TokenType.ID, k)), " + \
            "Type(Token(TokenType.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TokenType.EOF, None)), " + \
            "Var(Token(TokenType.ID, a)), " + \
            "Type(Token(TokenType.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TokenType.EOF, None)), " + \
            "Var(Token(TokenType.ID, z)), " + \
            "Type(Token(TokenType.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TokenType.EOF, None)), " + \
            "Var(Token(TokenType.ID, z)), " + \
            "Num(Token(TokenType.INT_CONST, 777)), " + \
            "Assign(Token(TokenType.ASSIGN, :=)), "  + \
            "NoOp(Token(TokenType.EOF, None)), " + \
            "Compound(Token(TokenType.EOF, None)), " + \
            "Block(Token(TokenType.EOF, None)), " + \
            "ProcDecl(Token(TokenType.EOF, p2)), " + \
            "NoOp(Token(TokenType.EOF, None)), " + \
            "Compound(Token(TokenType.EOF, None)), " + \
            "Block(Token(TokenType.EOF, None)), " + \
            "ProcDecl(Token(TokenType.EOF, p1)), " + \
            "Var(Token(TokenType.ID, a)), " + \
            "Num(Token(TokenType.INT_CONST, 10)), " + \
            "Assign(Token(TokenType.ASSIGN, :=)), " + \
            "NoOp(Token(TokenType.EOF, None)), " + \
            "Compound(Token(TokenType.EOF, None)), " + \
            "Block(Token(TokenType.EOF, None)), " + \
            "Program(Token(TokenType.EOF, None))" + \
        "]"
        self.assertEqual(act, exp)

    def test_fail_parse(self):
        with open("tests/bar.pas") as f:
            p = make_parser(f.read())
        with self.assertRaises(RuntimeError) as e:
            p.parse_compound()

        exp_msg = \
            "Invalid syntax: found Token(TokenType.ID, _a). " + \
            "Expected semi-colon, line 6"
        act_msg = e.exception.args[0]
        self.assertEqual(act_msg, exp_msg)

    def test_parser_proc_sig(self):
        ast = make_prog_ast_from_file("tests/part14.pas")
        actual = str(list(PostOrderIter(ast)))

        self.assertEqual(
            actual,
            "[" + \
                "Var(Token(TokenType.ID, x)), " + \
                "Type(Token(TokenType.REAL, [rR][eE][aA][lL])), " + \
                "VarDecl(Token(TokenType.EOF, None)), " + \
                "Var(Token(TokenType.ID, y)), " + \
                "Type(Token(TokenType.REAL, [rR][eE][aA][lL])), " + \
                "VarDecl(Token(TokenType.EOF, None)), " + \
                "Var(Token(TokenType.ID, a)), " + \
                "Type(Token(TokenType.INTEGER, " + \
                    "[iI][nN][tT][eE][gG][eE][rR])), " +\
                "Param(Token(TokenType.EOF, None)), " + \
                "Var(Token(TokenType.ID, y)), " + \
                "Type(Token(TokenType.INTEGER, " + \
                    "[iI][nN][tT][eE][gG][eE][rR])), " +\
                "VarDecl(Token(TokenType.EOF, None)), " + \
                "Var(Token(TokenType.ID, x)), " + \
                "Var(Token(TokenType.ID, a)), " + \
                "Var(Token(TokenType.ID, x)), " + \
                "Add(Token(TokenType.ADD, +)), " + \
                "Var(Token(TokenType.ID, y)), " + \
                "Add(Token(TokenType.ADD, +)), " + \
                "Assign(Token(TokenType.ASSIGN, :=)), " + \
                "NoOp(Token(TokenType.EOF, None)), " + \
                "Compound(Token(TokenType.EOF, None)), " + \
                "Block(Token(TokenType.EOF, None)), " + \
                "ProcDecl(Token(TokenType.EOF, alpha)), " + \
                "NoOp(Token(TokenType.EOF, None)), " + \
                "Compound(Token(TokenType.EOF, None)), " + \
                "Block(Token(TokenType.EOF, None)), " + \
                "Program(Token(TokenType.EOF, None))" + \
            "]"
        )

    def test_parser_proc_sig2(self):
        ast = make_prog_ast_from_file("tests/part14_2.pas")
        actual = str(list(PostOrderIter(ast)))

        self.assertEqual(
            actual,
            "[" + \
                "Var(Token(TokenType.ID, x)), " + \
                "Type(Token(TokenType.REAL, [rR][eE][aA][lL])), " + \
                "VarDecl(Token(TokenType.EOF, None)), " + \
                "Var(Token(TokenType.ID, y)), " + \
                "Type(Token(TokenType.REAL, [rR][eE][aA][lL])), " + \
                "VarDecl(Token(TokenType.EOF, None)), " + \
                "Var(Token(TokenType.ID, a)), " + \
                "Type(Token(TokenType.INTEGER, " + \
                    "[iI][nN][tT][eE][gG][eE][rR])), " +\
                "Param(Token(TokenType.EOF, None)), " + \
                "Var(Token(TokenType.ID, b)), " + \
                "Type(Token(TokenType.INTEGER, " + 
                    "[iI][nN][tT][eE][gG][eE][rR])), " +\
                "Param(Token(TokenType.EOF, None)), " + \
                "Var(Token(TokenType.ID, c)), " + \
                "Type(Token(TokenType.REAL, [rR][eE][aA][lL])), " + \
                "Param(Token(TokenType.EOF, None)), " + \
                "Var(Token(TokenType.ID, y)), " + \
                "Type(Token(TokenType.INTEGER, " + \
                    "[iI][nN][tT][eE][gG][eE][rR])), " +\
                "VarDecl(Token(TokenType.EOF, None)), " + \
                "Var(Token(TokenType.ID, x)), " + \
                "Var(Token(TokenType.ID, a)), " + \
                "Var(Token(TokenType.ID, x)), " + \
                "Add(Token(TokenType.ADD, +)), " + \
                "Var(Token(TokenType.ID, y)), " + \
                "Add(Token(TokenType.ADD, +)), " + \
                "Assign(Token(TokenType.ASSIGN, :=)), " + \
                "NoOp(Token(TokenType.EOF, None)), " + \
                "Compound(Token(TokenType.EOF, None)), " + \
                "Block(Token(TokenType.EOF, None)), " + \
                "ProcDecl(Token(TokenType.EOF, alpha)), " + \
                "NoOp(Token(TokenType.EOF, None)), " + \
                "Compound(Token(TokenType.EOF, None)), " + \
                "Block(Token(TokenType.EOF, None)), " + \
                "Program(Token(TokenType.EOF, None))" + \
            "]"
        )


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
            "got TokenType.EOF, line 1"
        )

    def test_expression_invalid_syntax1(self):
        with self.assertRaises(RuntimeError):
            ast = make_expr_ast('10 *')

    def test_expression_invalid_syntax2(self):
        with self.assertRaises(RuntimeError):
            ast = make_expr_ast('1 (1 + 2)')

    def test_expression_compound(self):
        with open("tests/foo.pas") as foo_pas:
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
        with open("tests/part10.pas") as pas:
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
        ast = make_prog_ast_from_file("tests/part12.pas")
        self.interpreter.interpret(ast)
        self.assertEqual(len(self.interpreter.GLOBAL_SCOPE), 1)
        self.assertEqual(self.interpreter.GLOBAL_SCOPE['a'], 10)


class SemanticAnalyzerTestCase(unittest.TestCase):
    def _get_scopes_from_str(self, s):
        return [el for el in s.splitlines() if re.match(r"\s*\(name", el)]

    def test_builder(self):
        ast = make_prog_ast_from_file("tests/part11.pas")

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
                    "'ProcSymbol(name=part11, params=[])']" + \
            ")",
            f"got {scopes[0]}"
        )

    def test_builder_name_error(self):
        ast = make_prog_ast_from_file("tests/name_err.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(NameError) as e:
            lyz.visit(ast)
        self.assertEqual(e.exception.args[0], "b at line 6")

    def test_builder_name_error2(self):
        ast = make_prog_ast_from_file("tests/name_err2.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(NameError) as e:
            lyz.visit(ast)
        self.assertEqual(e.exception.args[0], "a at line 7")

    def test_builder_part12(self):
        ast = make_prog_ast_from_file("tests/part12.pas")

        with LoggingToStrBuf() as sb:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        scopes = self._get_scopes_from_str(sb.getvalue())

        self.assertEqual(len(scopes), 4)
        self.assertEqual(
            scopes[0], 
            "(" + \
                "name: p2, " + \
                "level: 3, " + \
                "encl_scope: p1, " + \
                "symbols: ['<a:INTEGER>', '<z:INTEGER>']" + \
            ")"
        )
        self.assertEqual(
            scopes[1],
            "(" + \
                "name: p1, " + \
                "level: 2, " + \
                "encl_scope: global, " + \
                "symbols: [" + \
                    "'<a:REAL>', " + \
                    "'<k:INTEGER>', " + \
                    "'ProcSymbol(name=p2, params=[])'" + \
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
                    "'ProcSymbol(name=p1, params=[])'" + \
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
                    "'ProcSymbol(name=part12, params=[])']" + \
            ")",
        )

    def test_dup_var(self):
        ast = make_prog_ast_from_file("tests/dup_var_err.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(NameError) as e:
            lyz.analyze(ast)
        self.assertEqual(
            e.exception.args[0], "duplicate identifier y found at line 3")

    def test_part14_decl_only_chained_scope(self):
        ast = make_prog_ast_from_file("tests/part14_decl_only.pas")

        with LoggingToStrBuf() as sb:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        actual = listify_str(sb.getvalue())

        expected = [
            "ENTER scope builtins",
            "Insert: INTEGER",
            "Insert: REAL",
            "Insert: main",
            "ENTER scope global",
            "Lookup: REAL. (Scope name: global)",
            "Lookup: REAL. (Scope name: builtins)",
            "Lookup: x. (Scope name: global)",
            "Insert: x",
            "Lookup: REAL. (Scope name: global)",
            "Lookup: REAL. (Scope name: builtins)",
            "Lookup: y. (Scope name: global)",
            "Insert: y",
            "Insert: alpha",
            "ENTER scope alpha",
            "Lookup: INTEGER. (Scope name: alpha)",
            "Lookup: INTEGER. (Scope name: global)",
            "Lookup: INTEGER. (Scope name: builtins)",
            "Lookup: a. (Scope name: alpha)",
            "Insert: a",
            "Lookup: INTEGER. (Scope name: alpha)",
            "Lookup: INTEGER. (Scope name: global)",
            "Lookup: INTEGER. (Scope name: builtins)",
            "Lookup: y. (Scope name: alpha)",
            "Insert: y",
            "(name: alpha, level: 2, " + \
                "encl_scope: global, symbols: ['<a:INTEGER>', '<y:INTEGER>'])",
            "LEAVE scope alpha",
            "(name: global, level: 1, encl_scope: builtins, " + \
                "symbols: ['<x:REAL>', '<y:REAL>', " + \
                "\"ProcSymbol(name=alpha, " + \
                    "params=[VarSymbol(name='a', type='INTEGER')])\"])",
            "LEAVE scope global",
            "(name: builtins, level: 0, encl_scope: None, " + \
                "symbols: ['INTEGER', 'REAL', " + \
                "'ProcSymbol(name=main, params=[])'])",
            "LEAVE scope builtins"
        ]

        self.assertEqual(actual, expected)


    def test_part14_dup_param(self):
        ast = make_prog_ast_from_file("tests/part14_dup_param.pas")
        lyz = SemanticAnalyzer()

        with self.assertRaises(NameError) as e:
            lyz.analyze(ast)
        self.assertEqual(
            e.exception.args[0], "duplicate identifier a found at line 4")

    def test_part14_sibling_scopes(self):
        ast = make_prog_ast_from_file(
            "tests/part14_decl_only_sibling_scopes.pas")

        with LoggingToStrBuf() as sb:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        scopes = self._get_scopes_from_str(sb.getvalue())

        self.assertEqual(len(scopes), 4)
        self.assertEqual(
            scopes[0], 
            "(" + \
                "name: alphaa, " + \
                "level: 2, " + \
                "encl_scope: global, " + \
                "symbols: ['<a:INTEGER>', '<y:INTEGER>']" + \
            ")"
        )
        self.assertEqual(
            scopes[1], 
            "(" + \
                "name: alphab, " + \
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
                        "name=alphaa, " + \
                        "params=[VarSymbol(name='a', type='INTEGER')]" + \
                    ")\", " + \
                    "\"ProcSymbol(" + \
                        "name=alphab, " + \
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
                    "'ProcSymbol(name=main, params=[])']" + \
            ")",
        )

    def test_part14_var_ref(self):
        ast = make_prog_ast_from_file("tests/part14_var_ref.pas")

        with LoggingToStrBuf() as sb:
            with SemanticAnalyzer() as lyz:
                lyz.analyze(ast)

        actual = listify_str(sb.getvalue())

        expect = [
            "ENTER scope builtins",
            "Insert: INTEGER",
            "Insert: REAL",
            "Insert: main",
            "ENTER scope global",
            "Lookup: REAL. (Scope name: global)",
            "Lookup: REAL. (Scope name: builtins)",
            "Lookup: x. (Scope name: global)",
            "Insert: x",
            "Lookup: REAL. (Scope name: global)",
            "Lookup: REAL. (Scope name: builtins)",
            "Lookup: y. (Scope name: global)",
            "Insert: y",
            "Insert: alpha",
            "ENTER scope alpha",
            "Lookup: INTEGER. (Scope name: alpha)",
            "Lookup: INTEGER. (Scope name: global)",
            "Lookup: INTEGER. (Scope name: builtins)",
            "Lookup: a. (Scope name: alpha)",
            "Insert: a",
            "Lookup: INTEGER. (Scope name: alpha)",
            "Lookup: INTEGER. (Scope name: global)",
            "Lookup: INTEGER. (Scope name: builtins)",
            "Lookup: y. (Scope name: alpha)",
            "Insert: y",
            "Lookup: a. (Scope name: alpha)",
            "Lookup: x. (Scope name: alpha)",
            "Lookup: x. (Scope name: global)",
            "Lookup: y. (Scope name: alpha)",
            "Lookup: x. (Scope name: alpha)",
            "Lookup: x. (Scope name: global)",
            "(name: alpha, level: 2, encl_scope: global, " + \
                "symbols: ['<a:INTEGER>', '<y:INTEGER>'])",
            "LEAVE scope alpha",
            "(name: global, level: 1, encl_scope: builtins, " + \
                "symbols: ['<x:REAL>', '<y:REAL>', " + \
                "\"ProcSymbol(name=alpha, " + \
                "params=[VarSymbol(name='a', type='INTEGER')])\"])",
            "LEAVE scope global",
            "(name: builtins, level: 0, encl_scope: None, " + \
                "symbols: ['INTEGER', 'REAL', " + \
                "'ProcSymbol(name=main, params=[])'])",
            "LEAVE scope builtins"
        ]


        errmsg = prettify_strlist(sb.getvalue())
        self.assertEqual(actual, expect, f"got \n{errmsg}")


class DecoSrcBuilderTestCase(unittest.TestCase):
    def test_deco_src_part11(self):
        ast = make_prog_ast_from_file("tests/part11.pas")
        lyz = SemanticAnalyzer(s2s=True)
        lyz.analyze(ast)
        actual = lyz.deco_src()
        expect = \
            "program part110;\n   var x1 : INTEGER;\n   " + \
            "var y1 : REAL;\n   begin\n   end.    {END OF part11}"
        self.assertEqual(actual, expect)

    def test_deco_src_part14(self):
        ast = make_prog_ast_from_file("tests/part14_s2s.pas")
        lyz = SemanticAnalyzer(s2s=True)
        lyz.analyze(ast)
        expect = 'program main0;\n   var x1 : REAL;\n   ' + \
            'var y1 : REAL;\n   var z1 : INTEGER;\n   ' + \
            'var b1 : INTEGER;\n   var c1 : REAL;\n   ' + \
            'var d1 : INTEGER;\n   procedure alpha1(a2 : INTEGER);\n      ' + \
            'var y2 : INTEGER;\n      begin\n      ' + \
            '<x1:REAL> := <a2:INTEGER> + <x1:REAL> * <y2:INTEGER>\n      ' + \
            '<x1:REAL> := - <a2:INTEGER>\n      ' + \
            '<x1:REAL> := + <a2:INTEGER>\n      ' + \
            '<x1:REAL> := - + <a2:INTEGER>\n      end;    ' + \
            '{END OF alpha}\n   begin\n   ' + \
            '<z1:INTEGER> := <z1:INTEGER> - ' + \
            '<d1:INTEGER> / <b1:INTEGER> DIV <c1:REAL>\n   end.    ' + \
            '{END OF main}'
        self.assertEqual(lyz.deco_src(), expect)

    def test_deco_src_part14_2(self):
        ast = make_prog_ast_from_file("tests/part14_s2s_2.pas")
        lyz = SemanticAnalyzer(s2s=True)
        lyz.analyze(ast)
        expect = 'program main0;\n   ' + \
            'var x1 : REAL;\n   ' + \
            'var y1 : REAL;\n   ' + \
            'var z1 : INTEGER;\n   ' + \
            'procedure alphaa1(a2 : INTEGER);\n      ' + \
            'var y2 : INTEGER;\n      ' + \
            'begin\n      ' + \
            '<x1:REAL> := <a2:INTEGER> + <x1:REAL> + <y2:INTEGER>\n      ' + \
            'end;    {END OF alphaa}\n   ' + \
            'procedure alphab1(a2 : INTEGER);\n      ' + \
            'var b2 : INTEGER;\n      begin\n      ' + \
            'end;    {END OF alphab}\n   begin\n   ' + \
            'end.    {END OF main}'
        self.assertEqual(lyz.deco_src(), expect)

    def test_deco_src_part14_3(self):
        ast = make_prog_ast_from_file("tests/part14_s2s_3.pas")
        lyz = SemanticAnalyzer(s2s=True)
        lyz.analyze(ast)
        expect = 'program main0;\n   var b1 : REAL;\n   ' + \
            'var x1 : REAL;\n   var y1 : REAL;\n   ' + \
            'var z1 : INTEGER;\n   procedure ' + \
            'alphaa1(a2 : INTEGER);\n      ' + \
            'var b2 : INTEGER;\n      ' + \
            'procedure beta2(c3 : INTEGER);\n         ' + \
            'var y3 : INTEGER;\n         ' + \
            'procedure gamma3(c4 : INTEGER);\n            ' + \
            'var x4 : INTEGER;\n            begin\n            ' + \
            '<x4:INTEGER> := <a2:INTEGER> + <b2:INTEGER> + ' + \
            '<c4:INTEGER> + <x4:INTEGER> + <y3:INTEGER> + ' + \
            '<z1:INTEGER>\n            end;    ' + \
            '{END OF gamma}\n         begin\n         end;    ' + \
            '{END OF beta}\n      begin\n      end;    ' + \
            '{END OF alphaa}\n   procedure ' + \
            'alphab1(a2 : INTEGER);\n      var c2 : REAL;\n      ' + \
            'begin\n      <c2:REAL> := <a2:INTEGER> + <b1:REAL>\n      ' + \
            'end;    {END OF alphab}\n   begin\n   end.    {END OF main}'
        self.assertEqual(lyz.deco_src(), expect)


class Foo(unittest.TestCase):
    def test_foo(self):
        with open("tests/part12.pas") as f:
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
