! RUN: %fc %s -o %t && %t | FileCheck %s
module m1

contains
subroutine sub1(val)
  integer,intent(inout) :: val(:, :)
  integer :: i,j
  do i = 1,10
    do j = 1,20
      print *,val(i,j)
    end do
  end do

end subroutine sub1

end module m1

module m2
use m1

contains

  subroutine pgm

    integer :: val1(10,20),i,j
    do i = 1,10
      do j = 1,20
        val1(i,j) = (i * 10) + j
      end do
    end do
  call sub1(val1)
  end subroutine pgm

end module m2

program test
use m2
  call pgm()
end program test

!CHECK: 11

!CHECK: 12

!CHECK: 13

!CHECK: 14

!CHECK: 15

!CHECK: 16

!CHECK: 17

!CHECK: 18

!CHECK: 19

!CHECK: 20

!CHECK: 21

!CHECK: 22

!CHECK: 23

!CHECK: 24

!CHECK: 25

!CHECK: 26

!CHECK: 27

!CHECK: 28

!CHECK: 29

!CHECK: 30

!CHECK: 21

!CHECK: 22

!CHECK: 23

!CHECK: 24

!CHECK: 25

!CHECK: 26

!CHECK: 27

!CHECK: 28

!CHECK: 29

!CHECK: 30

!CHECK: 31

!CHECK: 32

!CHECK: 33

!CHECK: 34

!CHECK: 35

!CHECK: 36

!CHECK: 37

!CHECK: 38

!CHECK: 39

!CHECK: 40

!CHECK: 31

!CHECK: 32

!CHECK: 33

!CHECK: 34

!CHECK: 35

!CHECK: 36

!CHECK: 37

!CHECK: 38

!CHECK: 39

!CHECK: 40

!CHECK: 41

!CHECK: 42

!CHECK: 43

!CHECK: 44

!CHECK: 45

!CHECK: 46

!CHECK: 47

!CHECK: 48

!CHECK: 49

!CHECK: 50

!CHECK: 41

!CHECK: 42

!CHECK: 43

!CHECK: 44

!CHECK: 45

!CHECK: 46

!CHECK: 47

!CHECK: 48

!CHECK: 49

!CHECK: 50

!CHECK: 51

!CHECK: 52

!CHECK: 53

!CHECK: 54

!CHECK: 55

!CHECK: 56

!CHECK: 57

!CHECK: 58

!CHECK: 59

!CHECK: 60

!CHECK: 51

!CHECK: 52

!CHECK: 53

!CHECK: 54

!CHECK: 55

!CHECK: 56

!CHECK: 57

!CHECK: 58

!CHECK: 59

!CHECK: 60

!CHECK: 61

!CHECK: 62

!CHECK: 63

!CHECK: 64

!CHECK: 65

!CHECK: 66

!CHECK: 67

!CHECK: 68

!CHECK: 69

!CHECK: 70

!CHECK: 61

!CHECK: 62

!CHECK: 63

!CHECK: 64

!CHECK: 65

!CHECK: 66

!CHECK: 67

!CHECK: 68

!CHECK: 69

!CHECK: 70

!CHECK: 71

!CHECK: 72

!CHECK: 73

!CHECK: 74

!CHECK: 75

!CHECK: 76

!CHECK: 77

!CHECK: 78

!CHECK: 79

!CHECK: 80

!CHECK: 71

!CHECK: 72

!CHECK: 73

!CHECK: 74

!CHECK: 75

!CHECK: 76

!CHECK: 77

!CHECK: 78

!CHECK: 79

!CHECK: 80

!CHECK: 81

!CHECK: 82

!CHECK: 83

!CHECK: 84

!CHECK: 85

!CHECK: 86

!CHECK: 87

!CHECK: 88

!CHECK: 89

!CHECK: 90

!CHECK: 81

!CHECK: 82

!CHECK: 83

!CHECK: 84

!CHECK: 85

!CHECK: 86

!CHECK: 87

!CHECK: 88

!CHECK: 89

!CHECK: 90

!CHECK: 91

!CHECK: 92

!CHECK: 93

!CHECK: 94

!CHECK: 95

!CHECK: 96

!CHECK: 97

!CHECK: 98

!CHECK: 99

!CHECK: 100

!CHECK: 91

!CHECK: 92

!CHECK: 93

!CHECK: 94

!CHECK: 95

!CHECK: 96

!CHECK: 97

!CHECK: 98

!CHECK: 99

!CHECK: 100

!CHECK: 101

!CHECK: 102

!CHECK: 103

!CHECK: 104

!CHECK: 105

!CHECK: 106

!CHECK: 107

!CHECK: 108

!CHECK: 109

!CHECK: 110

!CHECK: 101

!CHECK: 102

!CHECK: 103

!CHECK: 104

!CHECK: 105

!CHECK: 106

!CHECK: 107

!CHECK: 108

!CHECK: 109

!CHECK: 110

!CHECK: 111

!CHECK: 112

!CHECK: 113

!CHECK: 114

!CHECK: 115

!CHECK: 116

!CHECK: 117

!CHECK: 118

!CHECK: 119

!CHECK: 120
