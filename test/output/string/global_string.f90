! RUN: %fc %s -o %t && %t | FileCheck %s
module dummy
  character(len=65) :: message = "This is a random message"
end module dummy

program test
  use dummy 
  print *, "message ", message
end program test

!CHECK: message             This is a random message
