subroutine pgm(n,m)
integer:: n,m, i, j, c
integer :: a(n,m)

c = 0
do i = 1,m
do j = 1,n
  a(j,i) = c
  c = c  + 1
enddo
enddo

do i = 1,m
do j = 1,n
  print *, a(j,i)
enddo
enddo

end

program p

integer :: n = 5 , m = 5
call pgm(n,m)

end
