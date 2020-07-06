PROGRAM greater_greater_equal_less_less_equal;
VAR
  res: integer;
  res2: integer;
  res3: integer;
  res4: integer;
  res5: integer;
  res6: integer;
  res7: integer;
  res8: integer;
  res9: integer;
  res10: integer;
  res11: integer;
  res12: integer;
  res13: integer;
  res14: integer;
  res15: integer;
  res16: integer;
BEGIN
  res := 2 + 1 > 3; { False }
  res2 := 2 | 1 > 3; { 2 }
  res3 := 2 + 2 > 3; { True }
  res4 := 2 | 4 > 3; { 3 }

  res5 := 2 + 1 >= 3; { True }
  res6 := 2 | 1 >= 3; { 2 }
  res7 := 2 + 2 >= 3; { True }
  res8 := 2 | 4 >= 3; { 3 }

  res9 := 2 + 1 < 4; { True }
  res10 := 2 | 1 < 4; { 3 }
  res11 := 2 + 2 < 4; { False }
  res12 := 2 | 4 < 4; { 2 }

  res13 := 2 + 1 <= 4; { True }
  res14 := 2 | 1 <= 4; { 3 }
  res15 := 2 + 2 <= 4; { True }
  res16 := 2 | 4 <= 4; { 3 }
END.