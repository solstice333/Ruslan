PROGRAM bool_test2;
VAR
  foo: boolean;
  res: boolean;
  res2: boolean;
  res3: boolean;
  res4: boolean;

BEGIN
  foo := true;
  res := 1 * 2 && 3 + 4 || falsE; {true}
  res2 := (foo && false) || True; {true}
  res3 := foo && 1 * 2 && fALse; {false}
  res4 := True || False && False; {true}
END.