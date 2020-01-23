! RUN: %fc %s -o %t && %t | FileCheck %s
program vin
  integer:: val(10,5),val2(5)
  val = 5

  val2 = sum(val, dim=1)
  print *,sum(val, dim=2)
  print *,sum(val, dim=2,mask=val /=5)
  print *,sum(val, dim=1,mask=val ==5)
  print *,val2
end program vin

!CHECK: 25           25           25           25           25           25           25          

!CHECK: 0            0            0            0            0            0            0           

!CHECK: 50           50           50           50           50

!CHECK: 50           50           50           50           50
