program pgm
integer :: a(3,3,3), i , j, k, c

c = 1
do i = 1, 3
do j = 1, 3
do k = 1, 3
 a(j,i,k) = c
 c = c + 1
enddo
enddo
enddo

do k = 1, 3
do j = 1, 3
do i = 1, 3
 if (a(k,j,i) <= a(i,j,k)) then
  print *, a(k,j,i)
 endif
enddo
enddo
enddo
end
