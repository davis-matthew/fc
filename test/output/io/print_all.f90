program v

logical::boolT = .TRUE.
logical::boolF = .false.

integer:: a= 10
integer(kind=8) ::d = 9999999999_8

real :: b = 3.45600000
double precision ::c=4.556_8

print *,boolT
print *,boolF

print *,a
print *,d

print *,b
print *,c

end program v
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            T
!CHECK:            F
!CHECK:           10
!CHECK:   9999999999
!CHECK:   3.45600009
!CHECK:   4.55600000
