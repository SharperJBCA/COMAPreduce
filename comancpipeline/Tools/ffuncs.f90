subroutine scanEdges(x, rms, edges, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: x(len_bn)
  real*8, intent(in) :: rms
  real*8, intent(out) :: edges(len_bn)


  integer :: i, k
  integer :: STATUS
  logical :: STARTING
  ! STATUS = 0 :: slowing down
  ! STATUS = 1 :: speeding up
  k = 1
  
  STARTING = .TRUE.

  do i=1, len_bn-1
     IF (ABS(x(i)-x(i+1)) > rms) THEN
        IF (STARTING) THEN
           IF (x(i) > x(i+1)) THEN
              STATUS = 0
              edges(k) = i
              k = k + 1
           ELSE IF (x(i) < x(i+1)) THEN
              STATUS = 1
              edges(k) = i
              k = k+1
           END IF
           STARTING = .FALSE.
        END IF 

        IF (x(i) > x(i+1) .and. STATUS == 1) THEN ! slowing down
           STATUS = 0
           edges(k) = i
           k = k + 1
        ELSE IF (x(i) < x(i+1) .and. STATUS == 0) THEN
           STATUS = 1
           edges(k) = i
           k = k+1
        END IF
      END IF
   enddo
   edges(k) = len_bn

end subroutine scanEdges
