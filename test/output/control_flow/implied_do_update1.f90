! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer  :: a(10) ,I = 0,j

  a = (/ (I, I = 1, 10) /)
  print *,I
  print *,a
end program test

!CHECK: 0

!CHECK: 1            2            3            4            5            6            7           
