! RUN: %fc %s -o %t && %t | FileCheck %s
program test 
  real(kind=8), parameter :: g = 2.345

  print *, g, log(g)

end program test

!CHECK: 2.34500003  0.85228539
