# Ruslan Interpreter

## Description

This is a modified version of the interpreter described here:

Tutorial here: https://ruslanspivak.com/lsbasi-part1/

1. Type annotations have been added (typing and mypy)

2. re module is used for the lexer

3. anytree is used for the AST

## Type Checking

```
$ python3 -m mypy spi.py
```

## Usage

```
$ python3 spi.py
```

## Tests

```
$ python3 -m unittest discover -v [-k <test method 1> [-k <test method 2> ...]]
```

where -k arguments are optional
