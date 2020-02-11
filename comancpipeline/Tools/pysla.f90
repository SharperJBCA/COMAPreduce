!!Quick routine to wrap the SLALIB moon position function.
!!It is inaccurate but fine for QUIJOTE accuracy. It downsamples
!! the data by 100x!

subroutine refro(zd, hm, tdk, pmb, rh, wl, phi, tlr, eps, ref, len_zd)
  implicit none
  integer, intent(in) :: len_zd
  real*8, intent(in) :: zd(len_zd)
  real*8, intent(out) :: ref(len_zd)
  real*8, intent(in) :: hm ! height
  real*8, intent(in) :: tdk ! ambient temp (K)
  real*8, intent(in) :: pmb ! pressure (mb)
  real*8, intent(in) :: rh ! relative humidity (0-1)
  real*8, intent(in) :: wl ! effective wavelength (um)
  real*8, intent(in) :: tlr ! latitude of observer
  real*8, intent(in) :: phi ! temperature lapse rate (K/m)
  real*8, intent(in) :: eps ! precision required (radian)

  !f2py real*8 zd, hm, tdk, pmb, rh, wl, phi, tlr, eps, ref
  !f2py integer len_zd

  integer :: i

  do i=1, len_zd
     call sla_refro(zd(i), hm, tdk, pmb, rh, wl, phi, tlr, eps, ref(i))
  enddo


end subroutine refro

subroutine rdplan(jd, np, lon, lat, ra, dec, diam, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  integer, intent(in) :: np
  real*8, intent(in) :: jd(len_bn)
  real*8, intent(out) :: diam(len_bn)
  real*8, intent(out) :: ra(len_bn)
  real*8, intent(out) :: dec(len_bn)

  !f2py integer len_bn
  !f2py real*8 lon, lat
  !f2py real*8 ra,dec,jd, diam

  !integer :: i

  real*8 :: pi = 3.14159265359

  integer :: step = 100
  integer :: kup,k
  kup = len_bn/step


  do k=1, len_bn
     call sla_rdplan(jd(k),np,lon*pi/180.0,lat*pi/180.0,ra(k),dec(k),diam(k))
  enddo
  
end subroutine rdplan




subroutine planet(jd, np, dist, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  integer, intent(in) :: np
  real*8, intent(in) :: jd(len_bn)
  real*8, intent(out) :: dist(6,len_bn)

  !f2py integer len_bn
  !f2py real*8 jd, dist

  integer :: jstat

  !real*8 :: pi = 3.14159265359

  integer :: step = 100
  integer :: kup,k
  kup = len_bn/step

  !mask = 1.0d0

  do k=1, len_bn
     call sla_planet(jd(k),np,dist(:,k),jstat)
  enddo
  
end subroutine planet


subroutine h2e(az, el, mjd, lon, lat, ra, dec, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: az(len_bn)
  real*8, intent(in) :: el(len_bn)
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(out) :: ra(len_bn)
  real*8, intent(out) :: dec(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface


  interface
     real*8 FUNCTION sla_dranrm(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dranrm
  end interface

  !f2py integer len_bn
  !f2py real*8 lon, lat
  !f2py real*8 ra,dec,mjd

  integer :: i

  real*8 :: gmst

  do i=1, len_bn
     call sla_dh2e(az(i), el(i), lat, ra(i), dec(i))
     gmst = sla_gmst(mjd(i))
     ra(i) = gmst + lon - ra(i)
     ra(i) = sla_dranrm(ra(i))
  enddo    

  
end subroutine h2e

subroutine e2h(ra, dec, mjd, lon, lat, az, el,lha, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: ra(len_bn)
  real*8, intent(in) :: dec(len_bn)
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(out) :: az(len_bn)
  real*8, intent(out) :: el(len_bn)
  real*8, intent(out) :: lha(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface


  interface
     real*8 FUNCTION sla_dranrm(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dranrm
  end interface


  !f2py integer len_bn
  !f2py real*8 lon, lat
  !f2py real*8 az,el,mjd

  integer :: i
  real*8 :: gmst

  do i=1, len_bn
     gmst = sla_gmst(mjd(i))
     lha(i) = lon + gmst - ra(i) ! CONVERT TO LHA

     call sla_de2h(lha(i), dec(i), lat, az(i), el(i))
  enddo    

  
end subroutine e2h

subroutine precess(ra, dec,mjd, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(inout) :: ra(len_bn)
  real*8, intent(inout) :: dec(len_bn)

  interface
     real*8 FUNCTION sla_epb(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_epb
  end interface

  !f2py integer len_bn
  !f2py real*8 mjd
  !f2py real*8 ra,dec

  integer :: i
  real*8 :: epoch

  do i=1, len_bn
     epoch = sla_epb(mjd(i))
     call sla_preces('FK5', epoch, 2000D0, ra(i), dec(i))
  enddo    
  
end subroutine precess


subroutine precess_year(ra, dec,mjd, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(inout) :: ra(len_bn)
  real*8, intent(inout) :: dec(len_bn)

  interface
     real*8 FUNCTION sla_epb(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_epb
  end interface

  !f2py integer len_bn
  !f2py real*8 mjd
  !f2py real*8 ra,dec

  integer :: i
  real*8 :: epoch

  do i=1, len_bn
     epoch = sla_epb(mjd(i))
     call sla_preces('FK5', 2000D0, epoch, ra(i), dec(i))
  enddo    
  
end subroutine precess_year





subroutine pa(ra, dec,mjd, lon,lat,pang, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(in) :: ra(len_bn)
  real*8, intent(in) :: dec(len_bn)
  real*8, intent(out) :: pang(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface

  interface
     real*8 FUNCTION sla_pa(radummy, decdummy, phidummy)
     real*8 :: radummy, decdummy, phidummy
     END FUNCTION sla_pa
  end interface

  !f2py integer len_bn
  !f2py real*8 mjd, lon, lat
  !f2py real*8 ra,dec, pang

  integer :: i
  real*8 :: ha, gmst

  do i=1, len_bn
     gmst = sla_gmst(mjd(i))
     ha = gmst + lon - ra(i)
     pang(i) = sla_pa(ha, dec(i), lat)
  enddo    

  
end subroutine pa


subroutine e2g(ra, dec, gl, gb, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: ra(len_bn)
  real*8, intent(in) :: dec(len_bn)
  real*8, intent(out) :: gl(len_bn)
  real*8, intent(out) :: gb(len_bn)

  !f2py integer len_bn
  !f2py real*8 ra,dec,gl,gb

  integer :: i


  do i=1, len_bn
     call sla_eqgal(ra(i), dec(i), gl(i), gb(i))
  enddo    

  
end subroutine e2g
