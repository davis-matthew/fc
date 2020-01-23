! RUN: %fc %s -o %t && %t | FileCheck %s
program a
  integer:: ar= 1

  do 
    ar = ar + 1
    print *, ar
    if (ar == 4) exit
  end do
end program a

!CHECK: 2

!CHECK: 3

!CHECK: 4
