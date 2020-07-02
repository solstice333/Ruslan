PROGRAM bool_test;
VAR
  foo: boolean;
  res: boolean;
  res2: boolean;
  res3: boolean;

BEGIN
  foo := true;
  res := 1 * 2 and 3 + 4 or false; {true}
  res2 := (foo and false) or true; {true}
  res3 := foo and 1 * 2 and false; {false}
END.