# run with `python3 -m unittest discover`
import unittest
from spi import TypeId, Lexer, Parser, Interpreter


class LexerTestCase(unittest.TestCase):
    def makeLexer(self, text):
        lexer = Lexer(text)
        return lexer

    def test_lexer_integer(self):
        lexer = self.makeLexer('234')
        token = lexer.get_next_token()
        self.assertEqual(token.type, TypeId.INT)
        self.assertEqual(token.value, 234)

    def test_lexer_mul(self):
        lexer = self.makeLexer('*')
        token = lexer.get_next_token()
        self.assertEqual(token.type, TypeId.MUL)
        self.assertEqual(token.value, '*')

    def test_lexer_div(self):
        lexer = self.makeLexer(' / ')
        token = lexer.get_next_token()
        self.assertEqual(token.type, TypeId.DIV)
        self.assertEqual(token.value, '/')

    def test_lexer_plus(self):
        lexer = self.makeLexer('+')
        token = lexer.get_next_token()
        self.assertEqual(token.type, TypeId.ADD)
        self.assertEqual(token.value, '+')

    def test_lexer_minus(self):
        lexer = self.makeLexer('-')
        token = lexer.get_next_token()
        self.assertEqual(token.type, TypeId.SUB)
        self.assertEqual(token.value, '-')

    def test_lexer_lparen(self):
        lexer = self.makeLexer('(')
        token = lexer.get_next_token()
        self.assertEqual(token.type, TypeId.LPAR)
        self.assertEqual(token.value, '(')

    def test_lexer_rparen(self):
        lexer = self.makeLexer(')')
        token = lexer.get_next_token()
        self.assertEqual(token.type, TypeId.RPAR)
        self.assertEqual(token.value, ')')


class InterpreterTestCase(unittest.TestCase):
    def makeInterpreter(self, text):
        interpreter = Interpreter(text)
        return interpreter

    def test_expression1(self):
        interpreter = self.makeInterpreter('3')
        result = interpreter.interpret()
        self.assertEqual(result, 3)

    def test_expression2(self):
        interpreter = self.makeInterpreter('2 + 7 * 4')
        result = interpreter.interpret()
        self.assertEqual(result, 30)

    def test_expression3(self):
        interpreter = self.makeInterpreter('7 - 8 / 4')
        result = interpreter.interpret()
        self.assertEqual(result, 5)

    def test_expression4(self):
        interpreter = self.makeInterpreter('14 + 2 * 3 - 6 / 2')
        result = interpreter.interpret()
        self.assertEqual(result, 17)

    def test_expression5(self):
        interpreter = self.makeInterpreter('7 + 3 * (10 / (12 / (3 + 1) - 1))')
        result = interpreter.interpret()
        self.assertEqual(result, 22)

    def test_expression6(self):
        interpreter = self.makeInterpreter(
            '7 + 3 * (10 / (12 / (3 + 1) - 1)) / (2 + 3) - 5 - 3 + (8)'
        )
        result = interpreter.interpret()
        self.assertEqual(result, 10)

    def test_expression7(self):
        interpreter = self.makeInterpreter('7 + (((3 + 2)))')
        result = interpreter.interpret()
        self.assertEqual(result, 12)

    def test_expression8(self):
        interpreter = self.makeInterpreter('- 3')
        result = interpreter.interpret()
        self.assertEqual(result, -3)

    def test_expression9(self):
        interpreter = self.makeInterpreter('+ 3')
        result = interpreter.interpret()
        self.assertEqual(result, 3)

    def test_expression10(self):
        interpreter = self.makeInterpreter('5 - - - + - 3')
        result = interpreter.interpret()
        self.assertEqual(result, 8)

    def test_expression11(self):
        interpreter = self.makeInterpreter('5 - - - + - (3 + 4) - +2')
        result = interpreter.interpret()
        self.assertEqual(result, 10)

    def test_no_expression(self):
        interpreter = self.makeInterpreter('   ')
        result = interpreter.interpret()
        self.assertEqual(result, '')

    def test_expression_invalid_syntax1(self):
        interpreter = self.makeInterpreter('10 *')
        with self.assertRaises(RuntimeError):
            interpreter.interpret()

    def test_expression_invalid_syntax2(self):
        interpreter = self.makeInterpreter('1 (1 + 2)')
        with self.assertRaises(RuntimeError):
            interpreter.interpret()


if __name__ == '__main__':
    unittest.main()
