program foo
 integer :: a
 read(5, *) a
 write(6, *) a
end program foo
! RUN: %fc %s -o %t && %t < ../input/stdio_rw1.in | FileCheck %s
!CHECK:            1
