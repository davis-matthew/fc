! RUN: %fc %s -o %t && %t | FileCheck %s
PROGRAM SUBDEM 

real A
integer B
B = 2
A = 3.23

print *, (A ** A)
print *, (A ** B)
print *, (B ** B)
print *, (B ** A)

end program SUBDEM

!CHECK: 44.12900925

!CHECK: 10.43290043

!CHECK: 4

!CHECK: 9.38267994
