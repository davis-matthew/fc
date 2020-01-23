! RUN: %fc %s -o %t && %t | FileCheck %s
subroutine outer()
      print *, "Hello from outter"
      call inner()
      contains
        subroutine inner()
          print *, "Hello from inner"
        end subroutine inner
end subroutine outer

program foo
        call outer()
end program foo

!CHECK: Hello from outter

!CHECK: Hello from inner
