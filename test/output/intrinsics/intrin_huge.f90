! RUN: %fc %s -o %t && %t | FileCheck %s
program test_huge_tiny
  print *, huge(0)
end program test_huge_tiny

!CHECK: 2147483647
