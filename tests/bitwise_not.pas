PROGRAM bitwise_not;
VAR
  res: integer;
  res2: integer;
  res3: integer;
  res4: integer;
  res5: integer;
  res6: integer;
BEGIN
  res := ~-3 + 4; { 6  }
  res2 := ~(3 + 4); { -8 }
  res3 := ~+-0; { -1 }
  res4 := ~(-3 + 3); { -1 }
  res5 := ~-3 + 1; { 3 }
  res6 := ~3 + 1; { -3 }
END.