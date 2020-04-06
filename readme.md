# Ruslan Interpreter

## Description

This is a modified version of the interpreter described here:

Tutorial here: https://ruslanspivak.com/lsbasi-part1/

1. Type annotations have been added (typing and mypy)

2. re module is used for the lexer

3. anytree is used for the AST

## Install Dependencies

From the top-level:

```
$ python3 -m pip install -r requirements.txt
```

## Type Checking

From the top-level:

```
$ python3 -m mypy spi.py
```

## Usage

From the top-level:

```
$ python3 spi.py -h
usage: spi.py [-h] FILE

simple pascal interpreter

positional arguments:
  FILE        pascal file

optional arguments:
  -h, --help  show this help message and exit
```

For example:

```
$ python3 spi.py tests/assignments.txt
```

## Tests

From the top-level:

```
$ python3 -m unittest discover -s tests -v
```
