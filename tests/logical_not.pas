PROGRAM logical_not;
VAR
  res: integer;
  res2: integer;
  res3: integer;
  res4: integer;
BEGIN
  res := !-3 + 4; { 4 }
  res2 := !(3 + 4); { False }
  res3 := !+-0; { True }
  res4 := !(-3 + 3); { True }
END.