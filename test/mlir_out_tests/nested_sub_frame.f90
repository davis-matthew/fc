subroutine outer(a)
  integer :: a
  print *, "Hello from outter", a
  call inner(10)
  contains
    subroutine inner(b)
      integer :: b
      print *, "Hello from inner", a, b, a+b
    end subroutine inner
end subroutine outer

program foo
        call outer(5)
end program foo
