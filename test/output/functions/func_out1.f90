! RUN: %fc %s -o %t && %t | FileCheck %s
function add(x, y)
  real :: x
  real :: y
  real :: add
  add = x + y 
end function add

program t

  real :: r
  r = add(10.55, 5.1010)
  print *,  r

end program t

!CHECK: 15.65100002
