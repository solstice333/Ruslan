from anytree import NodeMixin # type:ignore
from enum import Enum, auto
from typing import Any, List, overload

import enum
import anytree

@enum.unique
class TypeId(Enum):
   INT: int = auto()
   ADD: int = auto()
   SUB: int = auto()
   MUL: int = auto()
   DIV: int = auto()

   def __repr__(self) -> str:
      return str(self)

   @classmethod
   def operators(cls) -> List['TypeId']:
      return [
         cls.ADD, 
         cls.SUB, 
         cls.MUL, 
         cls.DIV
      ]

class Token:
   def __init__(self, ty: TypeId, value: Any) -> None:
      self.type = ty
      self.value = value

   def __str__(self) -> str:
      return f"Token({self.type}, {self.value})"

   def __repr__(self) -> str:
      return str(self)

class AST(NodeMixin):
   def __init__(
      self, 
      tok: Token,
      parent: 'AST'=None, 
      children: List['AST']=None 
   ) -> None:
      super().__init__()
      self.tok = tok
      self.parent = parent
      if children:
         self.children = children

   def __str__(self) -> str:
      return str(self.tok) 

   def __repr__(self) -> str:
      return str(self)

class Num(AST):
   def __init__(
      self, 
      numtok: Token, 
      parent: 'BinOp'=None
   ) -> None:
      assert numtok.type == TypeId.INT, \
         f"expecting {TypeId.INT}, got {numtok.type}"
      super().__init__(numtok, parent, None)

class BinOp(AST):
   def __init__(
      self, 
      optok: Token, 
      left: Num, right: Num, 
      parent: 'BinOp'=None
   ) -> None:
      optypes = [
         TypeId.ADD,
         TypeId.SUB
      ]
      assert optok.type in TypeId.operators(), \
         f"expecting {[TypeId.ADD, TypeId.SUB]}, got {optok.type}"
      super().__init__(optok, parent, [left, right])
      self.left = self.children[0]
      self.right = self.children[1]

def term() -> AST:
   two_tok = Token(TypeId.INT, 2)
   mul_tok = Token(TypeId.MUL, '*')
   three_tok = Token(TypeId.INT, 3)
   return BinOp(mul_tok, Num(two_tok), Num(three_tok))

def expr() -> AST:
   one_tok = Token(TypeId.INT, 1)
   add_tok = Token(TypeId.ADD, '+')
   return BinOp(add_tok, Num(one_tok), term())

def main() -> None:
   print(anytree.RenderTree(expr()))
   print(list(anytree.PostOrderIter(expr())))

if __name__ == '__main__':
   main()
