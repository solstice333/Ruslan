program cond_test;
var
  foo_res: integer;
  foo_res2: integer;

  foo_t: boolean;
  foo_f: boolean;

begin
  foo_res := 0;
  foo_res2 := 0;
  foo_t := true;
  foo_f := false;

  if (false) then
    foo_res := 0
  else if (1 * 2 and 3 + 4 or falsE) then
    foo_res := 1
  else
    foo_res2 := 1;

  if (True or False and False) then begin
    foo_res := foo_res + 2;
    foo_res := foo_res + 4
  end
  else begin
    foo_res2 := foo_res2 + 2;
    foo_res2 := foo_res2 + 4
  end;

  if (foo_f) then
    foo_res := foo_res + 8
  else
    foo_res2 := foo_res2 + 8;

  if (foo_t and 1 * 2 and fALse) then
    foo_res := foo_res + 16
  else
    foo_res2 := foo_res2 + 16;
end.