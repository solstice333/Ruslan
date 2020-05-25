import sys
import os
import os.path
import unittest
from anytree import PostOrderIter, RenderTree

sys.path.append(os.path.realpath(".."))
from spi import TypeId, Token, Lexer, Parser, Interpreter, SemanticAnalyzer


class Float:
    def __init__(self, val):
       self.val = val

    def eq(self, eps, other):
        if isinstance(other, float):
            other = Float(other)
        return abs(self.val - other.val) < eps  


class LexerTestCase(unittest.TestCase):
    def makeLexer(self, text):
        lexer = Lexer(text)
        return lexer

    def test_lexer_integer(self):
        lexer = self.makeLexer('234')
        token = next(iter(lexer))
        self.assertEqual(token.type, TypeId.INT_CONST)
        self.assertEqual(token.value, 234)

    def test_lexer_mul(self):
        lexer = self.makeLexer('*')
        token = next(iter(lexer))
        self.assertEqual(token.type, TypeId.MUL)
        self.assertEqual(token.value, '*')

    def test_lexer_div(self):
        lexer = self.makeLexer(' dIv ')
        token = next(iter(lexer))
        self.assertEqual(token.type, TypeId.INT_DIV)
        self.assertEqual(token.value, '[dD][iI][vV]')

    def test_lexer_plus(self):
        lexer = self.makeLexer('+')
        token = next(iter(lexer))
        self.assertEqual(token.type, TypeId.ADD)
        self.assertEqual(token.value, '+')

    def test_lexer_minus(self):
        lexer = self.makeLexer('-')
        token = next(iter(lexer))
        self.assertEqual(token.type, TypeId.SUB)
        self.assertEqual(token.value, '-')

    def test_lexer_lparen(self):
        lexer = self.makeLexer('(')
        token = next(iter(lexer))
        self.assertEqual(token.type, TypeId.LPAR)
        self.assertEqual(token.value, '(')

    def test_lexer_rparen(self):
        lexer = self.makeLexer(')')
        token = next(iter(lexer))
        self.assertEqual(token.type, TypeId.RPAR)
        self.assertEqual(token.value, ')')

    def test_lexer_ident(self):
        pass

    def test_lexer_res_kw_single_char_var(self):
        lexer = self.makeLexer("BEGIN a := 0; END.")
        tokens = list(iter(lexer))
        res_kw_dict = Lexer._RES_KW_TO_TID_INFO()

        self.assertEqual(len(tokens), 9)

        self.assertEqual(tokens[0], Token(TypeId.BEGIN, "[bB][eE][gG][iI][nN]"))
        self.assertEqual(tokens[1], Token(TypeId.ID, "a"))
        self.assertEqual(tokens[2], Token(TypeId.ASSIGN, ":="))
        self.assertEqual(tokens[3], Token(TypeId.INT_CONST, 0))
        self.assertEqual(tokens[4], Token(TypeId.SEMI, ";"))
        self.assertEqual(tokens[5], Token(TypeId.END, "[eE][nN][dD]"))
        self.assertEqual(tokens[6], Token(TypeId.DOT, "."))
        self.assertEqual(tokens[7], Token(TypeId.EOF, ""))
        self.assertEqual(tokens[8], Token(TypeId.EOF, None))

        self.assertIs(tokens[0], res_kw_dict['BEGIN'].token)
        self.assertIs(tokens[5], res_kw_dict['END'].token)

    def test_lexer_res_kw_multi_char_var(self):
        lexer = self.makeLexer("BEGIN foo_bar123 := 0; END.")
        tokens = list(iter(lexer))
        res_kw_dict = Lexer._RES_KW_TO_TID_INFO()

        self.assertEqual(len(tokens), 9)

        self.assertEqual(tokens[0], Token(TypeId.BEGIN, "[bB][eE][gG][iI][nN]"))
        self.assertEqual(tokens[1], Token(TypeId.ID, "foo_bar123"))
        self.assertEqual(tokens[2], Token(TypeId.ASSIGN, ":="))
        self.assertEqual(tokens[3], Token(TypeId.INT_CONST, 0))
        self.assertEqual(tokens[4], Token(TypeId.SEMI, ";"))
        self.assertEqual(tokens[5], Token(TypeId.END, "[eE][nN][dD]"))
        self.assertEqual(tokens[6], Token(TypeId.DOT, "."))
        self.assertEqual(tokens[7], Token(TypeId.EOF, ""))
        self.assertEqual(tokens[8], Token(TypeId.EOF, None))

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
            "Var(Token(TypeId.ID, x)), " + \
            "Num(Token(TypeId.INT_CONST, 11)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), " +\
            "Var(Token(TypeId.ID, y)), " + \
            "Num(Token(TypeId.INT_CONST, 2)), " + \
            "Var(Token(TypeId.ID, x)), " + \
            "Add(Token(TypeId.ADD, +)), " +\
            "Assign(Token(TypeId.ASSIGN, :=)), " +\
            "Compound(Token(TypeId.EOF, None))" +\
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
            "Var(Token(TypeId.ID, number)), " + \
            "Num(Token(TypeId.INT_CONST, 2)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), " + \
            "Var(Token(TypeId.ID, a)), " + \
            "Var(Token(TypeId.ID, number)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), " + \
            "Var(Token(TypeId.ID, b)), " + \
            "Num(Token(TypeId.INT_CONST, 10)), " + \
            "Var(Token(TypeId.ID, a)), " + \
            "Mul(Token(TypeId.MUL, *)), " + \
            "Num(Token(TypeId.INT_CONST, 10)), " + \
            "Var(Token(TypeId.ID, number)), " + \
            "Mul(Token(TypeId.MUL, *)), " + \
            "Num(Token(TypeId.INT_CONST, 4)), " + \
            "IntDiv(Token(TypeId.INT_DIV, [dD][iI][vV])), " + \
            "Add(Token(TypeId.ADD, +)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), " + \
            "Var(Token(TypeId.ID, c)), " + \
            "Var(Token(TypeId.ID, a)), " + \
            "Var(Token(TypeId.ID, b)), " + \
            "Neg(Token(TypeId.SUB, -)), " + \
            "Sub(Token(TypeId.SUB, -)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), " + \
            "Compound(Token(TypeId.EOF, None)), " + \
            "Var(Token(TypeId.ID, x)), " + \
            "Num(Token(TypeId.INT_CONST, 11)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), " + \
            "NoOp(Token(TypeId.EOF, None)), " + \
            "Compound(Token(TypeId.EOF, None))" + \
        "]"

        self.assertEqual(act, exp)

    def test_parser3(self):
        with open("tests/part10.pas") as f:
            p = make_parser(f.read())
        ast = p.parse()
        act = str(list(PostOrderIter(ast)))
        exp = "[" + \
            "Var(Token(TypeId.ID, a)), " + \
            "Type(Token(TypeId.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TypeId.EOF, None)), " + \
            "Var(Token(TypeId.ID, b)), " + \
            "Type(Token(TypeId.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TypeId.EOF, None)), " + \
            "Var(Token(TypeId.ID, y)), " + \
            "Type(Token(TypeId.REAL, [rR][eE][aA][lL])), " + \
            "VarDecl(Token(TypeId.EOF, None)), " + \
            "Var(Token(TypeId.ID, a)), " + \
            "Num(Token(TypeId.INT_CONST, 2)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), " + \
            "Var(Token(TypeId.ID, b)), " + \
            "Num(Token(TypeId.INT_CONST, 10)), " + \
            "Var(Token(TypeId.ID, a)), " + \
            "Mul(Token(TypeId.MUL, *)), " + \
            "Num(Token(TypeId.INT_CONST, 10)), " + \
            "Var(Token(TypeId.ID, a)), " + \
            "Mul(Token(TypeId.MUL, *)), " + \
            "Num(Token(TypeId.INT_CONST, 4)), " + \
            "IntDiv(Token(TypeId.INT_DIV, [dD][iI][vV])), " + \
            "Add(Token(TypeId.ADD, +)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), " + \
            "Var(Token(TypeId.ID, y)), " + \
            "Num(Token(TypeId.INT_CONST, 20)), " + \
            "Num(Token(TypeId.INT_CONST, 7)), " + \
            "FloatDiv(Token(TypeId.FLOAT_DIV, /)), " + \
            "Num(Token(TypeId.REAL_CONST, 3.14)), " + \
            "Add(Token(TypeId.ADD, +)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), " + \
            "NoOp(Token(TypeId.EOF, None)), " + \
            "Compound(Token(TypeId.EOF, None)), " + \
            "Block(Token(TypeId.EOF, None)), " + \
            "Program(Token(TypeId.EOF, None))" + \
        "]"

        self.assertEqual(act, exp)

    def test_parser_proc(self):
        with open("tests/part12.pas") as f:
            p = make_parser(f.read())

        ast = p.parse()
        act = str(list(PostOrderIter(ast)))
        exp = "[" + \
            "Var(Token(TypeId.ID, a)), " + \
            "Type(Token(TypeId.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TypeId.EOF, None)), " + \
            "Var(Token(TypeId.ID, a)), " + \
            "Type(Token(TypeId.REAL, [rR][eE][aA][lL])), " + \
            "VarDecl(Token(TypeId.EOF, None)), " + \
            "Var(Token(TypeId.ID, k)), " + \
            "Type(Token(TypeId.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TypeId.EOF, None)), " + \
            "Var(Token(TypeId.ID, a)), " + \
            "Type(Token(TypeId.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TypeId.EOF, None)), " + \
            "Var(Token(TypeId.ID, z)), " + \
            "Type(Token(TypeId.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TypeId.EOF, None)), " + \
            "Var(Token(TypeId.ID, z)), " + \
            "Num(Token(TypeId.INT_CONST, 777)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), "  + \
            "NoOp(Token(TypeId.EOF, None)), " + \
            "Compound(Token(TypeId.EOF, None)), " + \
            "Block(Token(TypeId.EOF, None)), " + \
            "ProcDecl(Token(TypeId.EOF, p2)), " + \
            "NoOp(Token(TypeId.EOF, None)), " + \
            "Compound(Token(TypeId.EOF, None)), " + \
            "Block(Token(TypeId.EOF, None)), " + \
            "ProcDecl(Token(TypeId.EOF, p1)), " + \
            "Var(Token(TypeId.ID, a)), " + \
            "Num(Token(TypeId.INT_CONST, 10)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), " + \
            "NoOp(Token(TypeId.EOF, None)), " + \
            "Compound(Token(TypeId.EOF, None)), " + \
            "Block(Token(TypeId.EOF, None)), " + \
            "Program(Token(TypeId.EOF, None))" + \
        "]"
        self.assertEqual(act, exp)

    def test_fail_parse(self):
        with open("tests/bar.pas") as f:
            p = make_parser(f.read())
        with self.assertRaises(RuntimeError) as e:
            p.parse_compound()

        exp_msg = \
            "Invalid syntax: found Token(TypeId.ID, _a). " + \
            "Expected semi-colon, line 6"
        act_msg = e.exception.args[0]
        self.assertEqual(act_msg, exp_msg)


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
            "Invalid syntax: was expecting TypeId.ID, got TypeId.EOF, line 1"
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
    def test_builder(self):
        ast = make_prog_ast_from_file("tests/part11.pas")
        stb = SemanticAnalyzer()
        stb.visit(ast)

        self.assertEqual(
            str(stb.table), 
            "Symbols: ['INTEGER', 'REAL', '<x:INTEGER>', '<y:REAL>']"
        )

    def test_builder_name_error(self):
        ast = make_prog_ast_from_file("tests/name_err.pas")
        stb = SemanticAnalyzer()

        with self.assertRaises(NameError) as e:
            stb.visit(ast)
        self.assertEqual(e.exception.args[0], "b at line 6")

    def test_builder_name_error2(self):
        ast = make_prog_ast_from_file("tests/name_err2.pas")
        stb = SemanticAnalyzer()

        with self.assertRaises(NameError) as e:
            stb.visit(ast)
        self.assertEqual(e.exception.args[0], "a at line 7")

    def test_builder_part12(self):
        ast = make_prog_ast_from_file("tests/part12.pas")
        stb = SemanticAnalyzer()
        stb.build(ast)
        self.assertEqual(
            "Symbols: ['INTEGER', 'REAL', '<a:INTEGER>']", 
            str(stb.table)
        )


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

if __name__ == '__main__':
    unittest.main()
