program pgm
integer :: a(2:3,2:3,2:3), i , j, k, c

c = 1
do i = 2, 3
do j = 2, 3
do k = 2, 3
 a(j,i,k) = c
 c = c + 1
enddo
enddo
enddo

do k = 2, 3
do j = 2, 3
do i = 2, 3
 if (a(k,j,i) <= a(i,j,k)) then
  print *, a(k,j,i)
 endif
enddo
enddo
enddo
end
