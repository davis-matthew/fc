program readtest
  character(len=80) ::msg

  open  (1, file = '../input/3.dat', status = 'old')
  read (1, *) msg
  print *, msg
  close(1)

end program readtest
!RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK: ccompilertestcase
