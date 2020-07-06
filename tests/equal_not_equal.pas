PROGRAM equal_not_equal;
VAR
  res: integer;
  res2: integer;
BEGIN
  res := 0x3 ^ 0x1 == 0x1; { 2 }
  res2 := 0x3 ^ 0x1 != 0x2; { 2 }
END.