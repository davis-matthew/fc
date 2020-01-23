! RUN: %fc %s -o %t && %t | FileCheck %s
program test
      character(len=10) :: msg
      integer :: i

      msg = "msg2"

      print *, "Message " , msg

      
      if (msg /= 'a') then
        print *, "message not a"
      end if

      
end program test

!CHECK: Message             msg2

!CHECK: message not a
