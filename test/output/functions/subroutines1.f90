! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: a(10,10)
  integer :: n =10,i,j

  do i = 1,n,1
    do j = 1,n,1
      a(i,j) =  i * j + 3
  end do
    end do
    i = 10
    j = 10
  call dummy(a,n,i,j)
end program t

subroutine dummy(vin, n,i,j)
  integer,intent(inout) :: vin(10,10),n,i,j
 
  IF (n == 10) THEN
    do i = 1,n,1
      do j = 1,n,1
        print *,vin(j,i)
      end do
    end do
  END IF
  call printOthers(n,i,j)
end subroutine dummy

subroutine printOthers(n,i,j)
  integer,intent(in) :: n,i,j
 
  PRINT *, n,i,j
end subroutine printOthers

!CHECK: 4

!CHECK: 5

!CHECK: 6

!CHECK: 7

!CHECK: 8

!CHECK: 9

!CHECK: 10

!CHECK: 11

!CHECK: 12

!CHECK: 13

!CHECK: 5

!CHECK: 7

!CHECK: 9

!CHECK: 11

!CHECK: 13

!CHECK: 15

!CHECK: 17

!CHECK: 19

!CHECK: 21

!CHECK: 23

!CHECK: 6

!CHECK: 9

!CHECK: 12

!CHECK: 15

!CHECK: 18

!CHECK: 21

!CHECK: 24

!CHECK: 27

!CHECK: 30

!CHECK: 33

!CHECK: 7

!CHECK: 11

!CHECK: 15

!CHECK: 19

!CHECK: 23

!CHECK: 27

!CHECK: 31

!CHECK: 35

!CHECK: 39

!CHECK: 43

!CHECK: 8

!CHECK: 13

!CHECK: 18

!CHECK: 23

!CHECK: 28

!CHECK: 33

!CHECK: 38

!CHECK: 43

!CHECK: 48

!CHECK: 53

!CHECK: 9

!CHECK: 15

!CHECK: 21

!CHECK: 27

!CHECK: 33

!CHECK: 39

!CHECK: 45

!CHECK: 51

!CHECK: 57

!CHECK: 63

!CHECK: 10

!CHECK: 17

!CHECK: 24

!CHECK: 31

!CHECK: 38

!CHECK: 45

!CHECK: 52

!CHECK: 59

!CHECK: 66

!CHECK: 73

!CHECK: 11

!CHECK: 19

!CHECK: 27

!CHECK: 35

!CHECK: 43

!CHECK: 51

!CHECK: 59

!CHECK: 67

!CHECK: 75

!CHECK: 83

!CHECK: 12

!CHECK: 21

!CHECK: 30

!CHECK: 39

!CHECK: 48

!CHECK: 57

!CHECK: 66

!CHECK: 75

!CHECK: 84

!CHECK: 93

!CHECK: 13

!CHECK: 23

!CHECK: 33

!CHECK: 43

!CHECK: 53

!CHECK: 63

!CHECK: 73

!CHECK: 83

!CHECK: 93

!CHECK: 103

!CHECK: 10          11          11
