PROGRAM left_right_shift;
VAR
  res: integer;
  res2: integer;
  res3: integer;
  res4: integer;
BEGIN
  res := 3 + 1 << 2; { 16 }
  res2 := 3 + 1 >> 2; { 1 }
  res3 := 4 & 7 << 2; { 4 }
  res4 := 4 | 7 >> 2; { 5 }
END.