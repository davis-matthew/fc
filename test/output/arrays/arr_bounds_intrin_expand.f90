! RUN: %fc %s -o %t && %t | FileCheck %s
subroutine test(array)
  integer, dimension(1:10, -1:20),intent(inout) :: array
  array = -3
  print *,array
end subroutine test


program vin
  integer ::array(10, -1:20)
  call test(array)
end program vin

!CHECK: -3           -3           -3           -3           -3           -3           -3          
