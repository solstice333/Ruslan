program cond_test;
var
  foo_res: integer;
  foo: boolean;

{  foo2: boolean;}
{  foo3: boolean;}
{  foo4: boolean;}
{  foo5: boolean;}

begin
  foo_res := 0;
  foo := true;

  if (foo) then
    foo_res := 1;

  if (foo) then
  begin
    foo_res := foo_res + 2;
    foo_res := foo_res + 4
  end;

{  foo2 := false;}
{  foo3 := 1 * 2 and 3 + 4 or falsE; (*true*)}
{  foo4 := (foo and false) or true; (*true*)}
{  foo5 := foo and 1 * 2 and fALse; (*false*)}
end.