PROGRAM bitwise_or;
VAR
  foo: integer;
  bar: integer;
  baz: integer;
  res: integer;

BEGIN
  foo := 0x5;
  bar := 0x1;
  baz := 0x3;
  res := (foo + baz)*2 - 8 | bar;
  writeln(res);
END.