program readtest
  integer, dimension(10) :: x

  open  (1, file = '../input/2.dat', status = 'old')
  read (1, *) x
  print *, x
  close(1)

end program readtest
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK: 0
!CHECK: 1
!CHECK: 2
!CHECK: 3
!CHECK: 4
!CHECK: 5
!CHECK: 6
!CHECK: 7
!CHECK: 8
!CHECK: 9
