program pgm
integer :: a(5:10),i
a(10) = 3

do i = 5,10
 a(i) = i
enddo

do i = 5,10
  print *, a(i)
enddo
end
