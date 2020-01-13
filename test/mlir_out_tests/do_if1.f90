program pgm
  integer c
  integer i
  c = 10


  do i = 1,10
    if(i .gt. 4) then
      c = c + 1
    else if ( i .gt. 3) then
      c = c + 2
    else if ( i .gt. 2) then
      c = c + 2
    else
      c = c -1 
    endif
  end do
  
  print *, c
  end
