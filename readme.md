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
 python3 spi.py -h
usage: spi.py [-h] [--verbose-stack] [--verbose-scope] [-v] [-s] FILE

simple pascal interpreter

positional arguments:
  FILE              pascal source file

optional arguments:
  -h, --help        show this help message and exit
  --verbose-stack   verbose output related to runtime stack events
  --verbose-scope   verbose output related to semantic analysis
  -v, --verbose     enables all verbose flags (--stack-verbose, --scope-
                    verbose)
  -s, --src-to-src  translate source in FILE to decorated source and print

```

For example:

```
$ python3 spi.py tests/foo.pas
```

## Tests

From the top-level:

```
$ cd tests
$ python3 -m unittest -v -f
```

## PyCharm

To set up default .idea configuration:

```
$ git merge --no-ff origin/idea_config
$ git reset HEAD~1
```

Tests can be ran by right clicking on `tests` folder in the project window and clicking on `Run 'Unittests in tests'`.
