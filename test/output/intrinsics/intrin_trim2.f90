! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  
  character(len=80) :: filename
  character(len=80) :: ext

  filename = "file       "
  ext = ".dat"

  print *, trim(filename)

  ! to make sure trim realy works
  filename = trim(filename) // ext

  print *, filename

end program test

!CHECK: file

!CHECK: file.dat
