! RUN: %fc %s -o %t && %t | FileCheck %s
program test
      character, dimension(10) :: arr
      character(len=100) :: msg
      msg = "compiler " // " tree " //  "technologies" // " fc"
      print *, msg
end program test

!CHECK: compiler  tree technologies fc
