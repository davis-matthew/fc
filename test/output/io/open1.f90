! RUN: %fc %s -o %t && %t | FileCheck %s
program test
      !Fc by default creates new file
      open(10, file='1.txt')
      close(10)
      print *,"openclose" !CHECK: openclose
end program test
