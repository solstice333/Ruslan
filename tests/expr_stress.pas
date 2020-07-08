PROGRAM expr_stress;
VAR
  res: integer;

BEGIN
  res := ((((((3 > 4) + -7 + +9 - 1*5 + 10 >> 1 << 2 == 0) || true) +
    1 << 3 && true ^ 1) + 1 << 4 + (~-1 + 1)) - (~-32 + 1)*2 % 10 -
    (~-28 + 1)*2) % 10;
END.