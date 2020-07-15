program Main;
var y: integer;
   procedure Alpha(a : integer; b : integer);
   var x : integer;
   begin
      x := (a + b ) * 2;
   end;
begin { Main }
   Alpha(3 + 5, 7);  { procedure call }
   y := 0x20;
end.  { Main }
