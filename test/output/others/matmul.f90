! RUN: %fc %s -o %t && %t | FileCheck %s
subroutine kernel(a,b,c, n, m, l, lb)
integer, intent(in) :: n,m,l,lb
integer :: a(lb:n,lb:m), b(lb:m, lb:l) , c(lb:n,lb:l)
integer :: example(2:10, 3:30)
integer :: i,j,k

do i = 1,n
  do j = 1,l  
    c(i,j) = 0
    do k = 1,m
      c(i,j) = c(i,j) + a(i,k) * b(k,j)
    enddo
  enddo
enddo

end

subroutine matmul(n,m,l, lb)
integer, intent(in) :: n,m,l,lb
integer :: a(lb:n,lb:m), b(lb:m, lb:l) , c(lb:n,lb:l)
integer :: i,j,k, sumVal=0

do i = 1,m
  do j = 1,n
   a(i,j)= 1
  enddo
  do j = 1,l
   b(i,j)= 1
  enddo
enddo

call kernel(a,b,c,n,m,l,lb)

do i = 1,n
  do j = 1,l
   sumVal = sumVal + mod(c(i,j),10)
  enddo
enddo 

print *, sumVal
end

program pgm
integer ::n,m,l, lb

lb = 1
n = 11
m = n 
l = n
call matmul(n,m,l, lb)

end

!CHECK: 121
