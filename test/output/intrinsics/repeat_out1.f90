! RUN: %fc %s -o %t && %t | FileCheck %s
program test
      print *, repeat('a', 10)
end program test

!CHECK: aaaaaaaaaa
