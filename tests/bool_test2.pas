PROGRAM bool_test2;
VAR
  foo: boolean;
  res: boolean;
  res2: boolean;
  res3: boolean;
  res4: boolean;

BEGIN
  foo := true;
  res := 1 * 2 and 3 + 4 or falsE; {true}
  res2 := (foo and false) or True; {true}
  res3 := foo and 1 * 2 and fALse; {false}
  res4 := True or False and False; {true}
END.