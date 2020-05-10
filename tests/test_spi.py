import sys
import os
import os.path
import unittest
from anytree import PostOrderIter, RenderTree

sys.path.append(os.path.realpath(".."))
from spi import TypeId, Token, Lexer, Parser, Interpreter, SymbolTableBuilder


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

    def makeParser(self, text):
        lexer = Lexer(text)
        return Parser(lexer)

    def test_parser1(self):
        p = self.makeParser("BEGIN x := 11; y := 2 + x END.")
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
            "Compound(Token(TypeId.EOF, Compound))" +\
        "]"
        self.assertEqual(act, exp)

    def test_parser2(self):
        p = self.makeParser(
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
            "Compound(Token(TypeId.EOF, Compound)), " + \
            "Var(Token(TypeId.ID, x)), " + \
            "Num(Token(TypeId.INT_CONST, 11)), " + \
            "Assign(Token(TypeId.ASSIGN, :=)), " + \
            "NoOp(Token(TypeId.EOF, NoOp)), " + \
            "Compound(Token(TypeId.EOF, Compound))" + \
        "]"

        self.assertEqual(act, exp)

    def test_parser3(self):
        with open("tests/part10.pas") as f:
            p = self.makeParser(f.read())
        ast = p.parse()
        act = str(list(PostOrderIter(ast)))
        exp = "[" + \
            "Var(Token(TypeId.ID, a)), " + \
            "Type(Token(TypeId.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TypeId.EOF, VarDecl)), " + \
            "Var(Token(TypeId.ID, b)), " + \
            "Type(Token(TypeId.INTEGER, [iI][nN][tT][eE][gG][eE][rR])), " + \
            "VarDecl(Token(TypeId.EOF, VarDecl)), " + \
            "Var(Token(TypeId.ID, y)), " + \
            "Type(Token(TypeId.REAL, [rR][eE][aA][lL])), " + \
            "VarDecl(Token(TypeId.EOF, VarDecl)), " + \
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
            "NoOp(Token(TypeId.EOF, NoOp)), " + \
            "Compound(Token(TypeId.EOF, Compound)), " + \
            "Block(Token(TypeId.EOF, Block)), " + \
            "Program(Token(TypeId.EOF, Program))" + \
        "]"

        self.assertEqual(act, exp)

    def test_fail_parse(self):
        with open("tests/bar.pas") as f:
            p = self.makeParser(f.read())
        with self.assertRaises(RuntimeError) as e:
            p.parse_compound()

        exp_msg = \
            "Invalid syntax: found Token(TypeId.ID, _a). " + \
            "Expected semi-colon, line 6"
        act_msg = e.exception.args[0]
        self.assertEqual(act_msg, exp_msg)


class InterpreterTestCase(unittest.TestCase):
    def makeInterpreter(self, text):
        interpreter = Interpreter(text)
        return interpreter

    def test_expression1(self):
        interpreter = self.makeInterpreter('3')
        result = interpreter.interpret_expr()
        self.assertEqual(result, 3)

    def test_expression2(self):
        interpreter = self.makeInterpreter('2 + 7 * 4')
        result = interpreter.interpret_expr()
        self.assertEqual(result, 30)

    def test_expression3(self):
        interpreter = self.makeInterpreter('7 - 8 div 4')
        result = interpreter.interpret_expr()
        self.assertEqual(result, 5)

    def test_expression4(self):
        interpreter = self.makeInterpreter('14 + 2 * 3 - 6 div 2')
        result = interpreter.interpret_expr()
        self.assertEqual(result, 17)

    def test_expression5(self):
        interpreter = self.makeInterpreter('7 + 3 * (10 div (12 div (3 + 1) - 1))')
        result = interpreter.interpret_expr()
        self.assertEqual(result, 22)

    def test_expression6(self):
        interpreter = self.makeInterpreter(
            '7 + 3 * (10 div (12 div (3 + 1) - 1)) div (2 + 3) - 5 - 3 + (8)'
        )
        result = interpreter.interpret_expr()
        self.assertEqual(result, 10)

    def test_expression7(self):
        interpreter = self.makeInterpreter('7 + (((3 + 2)))')
        result = interpreter.interpret_expr()
        self.assertEqual(result, 12)

    def test_expression8(self):
        interpreter = self.makeInterpreter('- 3')
        result = interpreter.interpret_expr()
        self.assertEqual(result, -3)

    def test_expression9(self):
        interpreter = self.makeInterpreter('+ 3')
        result = interpreter.interpret_expr()
        self.assertEqual(result, 3)

    def test_expression10(self):
        interpreter = self.makeInterpreter('5 - - - + - 3')
        result = interpreter.interpret_expr()
        self.assertEqual(result, 8)

    def test_expression11(self):
        interpreter = self.makeInterpreter('5 - - - + - (3 + 4) - +2')
        result = interpreter.interpret_expr()
        self.assertEqual(result, 10)

    def test_no_expression(self):
        interpreter = self.makeInterpreter('   ')
        result = interpreter.interpret_expr()
        self.assertEqual(result, '')

    def test_expression_invalid_syntax1(self):
        interpreter = self.makeInterpreter('10 *')
        with self.assertRaises(RuntimeError):
            interpreter.interpret_expr()

    def test_expression_invalid_syntax2(self):
        interpreter = self.makeInterpreter('1 (1 + 2)')
        with self.assertRaises(RuntimeError):
            interpreter.interpret_expr()

    def test_expression_program(self):
        with open("tests/foo.pas") as foo_pas:
            txt = foo_pas.read()

        interpreter = self.makeInterpreter(txt)
        interpreter.interpret_compound()
        self.assertEqual(
            interpreter.GLOBAL_SCOPE,
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

        interpreter = self.makeInterpreter(txt)
        interpreter.interpret()
        self.assertEqual(len(interpreter.GLOBAL_SCOPE), 3)
        self.assertEqual(interpreter.GLOBAL_SCOPE['a'], 2)
        self.assertEqual(interpreter.GLOBAL_SCOPE['b'], 25)

        self.assertTrue(
            Float(interpreter.GLOBAL_SCOPE['y']).eq(0.01, Float(5.99))
        )


class SymbolTableBuilderTestCase(unittest.TestCase):
    def test_symbol_builder(self):
        with open("tests/part11.pas") as f:
            txt = f.read()
        parser = Parser(Lexer(txt))
        ast = parser.parse()
        stb = SymbolTableBuilder()
        stb.visit(ast)
        self.assertEqual(
            str(stb.table), 
            "Symbols: [INTEGER, REAL, <x:INTEGER>, <y:REAL>]"
        )

    def test_symbol_builder_name_error(self):
        with open("tests/name_err.pas") as f:
            txt = f.read()
        parser = Parser(Lexer(txt))
        ast = parser.parse()
        stb = SymbolTableBuilder()

        with self.assertRaises(NameError) as e:
            stb.visit(ast)

        self.assertEqual(e.exception.args[0], "b at line 6")

    def test_symbol_builder_name_error2(self):
        with open("tests/name_err2.pas") as f:
            txt = f.read()
        parser = Parser(Lexer(txt))
        ast = parser.parse()
        stb = SymbolTableBuilder()

        with self.assertRaises(NameError) as e:
            stb.visit(ast)

        self.assertEqual(e.exception.args[0], "a at line 7")

if __name__ == '__main__':
    unittest.main()
