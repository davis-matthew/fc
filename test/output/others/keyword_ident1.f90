! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer         :: status(3)

  status(1) = 99
  status(2) = 88
  status(3) = 77

  print *, status
end program test

!CHECK: 99           88           77
