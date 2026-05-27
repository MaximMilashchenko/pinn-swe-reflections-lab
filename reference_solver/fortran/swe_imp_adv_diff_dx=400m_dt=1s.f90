!***************************************************************************
!*                                                                         *
!*             1-D Shallow-Water equations semi-implicit solver            *
!*                                                                         *
!*                           Kai Logemann, HZG, October 2020               *
!*                                                                         *
!***************************************************************************

program shallow_water
implicit none

integer i,nx,ip,im
parameter(nx=5002)

real dt,dx,u(nx),un(nx),zeta(nx),zetan(nx),h(nx),cd,g,uint(nx),hm,us(nx),uints(nx),div(nx)
real time,rhour,umin,umax,zmin,zmax,wimp,ce(nx),cw(nx),zetnew,zs(nx),dif,maxdif,d(nx),rminute,minute
real x, pi, sigma, mu, uu, adv, AH, adv_coefficient
integer hour,it,ho


! set constants ==================================

dt = 1. ! time step [s]
dx = 400. ! grid spacing [m]

do i = 2,nx-1,1
 d(i) = 100. ! undisturbed water depth [m], topography
enddo
d(1) = -10. ! dry boundary cell
d(nx) = -10. ! dry boundary cell


adv_coefficient = 1. ! coefficient for in-/excluding advection
cd = 0.0 ! 2.e-3 ! bottom drag coefficient
AH = 5.e+4 ! 1e-6 ! turbulent diffusion coefficient [m**2/s]
g = 9.81 ! acceleration due to gravity [m/s**2]

uint = 0. ! vertically integrated velocity [m**2/s]

wimp = 0.5 ! weight between time levels

PI  = 4. * ATAN(1.)

! initalisation ====================================

u = 0. ! velocity [m/s]
un = 0. ! velocity at new time step [m/s]
us = 0. ! interim solution of velocity [m/s]

zeta = 0. ! sea surface elevation [m]
zetan = 0. ! sea surface elevation at new time level [m]

do i = 2,nx-1,1

  x = (REAL(i)-0.5) * dx ! start sine wave from first wet cell

  mu = 1000000.0
  sigma = 100000.0
  zeta(i) = 2.5 * sigma * (1. / (sigma * sqrt(2*pi))) * exp(-0.5 * ( (x-mu) / sigma )**2. )

  h(i) = d(i) + zeta(i)

enddo

time = 0. ! run time [s]

! ======================================================

ho = 0
call output(ho) ! write output to directory "dat"
write(*,'(A,F8.1,A,I5)') 'INITIAL OUTPUT: time=', time, ' ho=', ho

! ======================================================

open(66, file = '../data/raw/AH=5e+4/pegel/pegel_07.dat') ! output file for gauge data
                                    ! for the eastern coast (i = nx-1)

! begin of time loop

100 time = time + dt
rhour = time/3600.
hour = int(rhour)
rminute = time/60.
minute = int(rminute)

! equation of motion => us (interim solution)
do i = 1,nx,1
  im = max(1,i-1)
  ip = min(nx,i+1)

  ! momentum advection (upstream) => adv -----------------
    uu = 0.5*u(i)+0.25*u(im)+0.25*u(ip)

	if(uu.ge.0.) then
	  uu = max(0.,u(im))
	  adv = -dt*uu*(u(i)-u(im))/dx
	else
      uu = min(0.,u(ip))
	  adv = -dt*uu*(u(ip)-u(i))/dx
	endif

  if((d(i).gt.0.1).and.(d(ip).gt.0.1)) then
    hm = 0.5*(h(i)+h(ip))
    us(i) = u(i)- dt*cd/hm*u(i)*abs(u(i)) - (1.-wimp)*dt*g*(zeta(ip)-zeta(i))/dx &
            + dt * AH * (u(i+1) - 2. * u(i) + u(i-1)) / (dx**2.) + adv * adv_coefficient
  else
    us(i) = 0.
  endif
enddo

! vertical integrals of velocity => uint, and interim solution uints

do i = 1,nx,1
  ip = min(nx,i+1)
  if(d(i).gt.0.0.and.(d(ip).gt.0.0)) then
    hm = 0.5*(h(i)+h(ip))
    uint(i) = hm*u(i)
	uints(i) = hm*us(i)
  else
    uint(i) = 0.
	uints(i) = 0.
  endif
enddo

! equation of continuity, first compute div

div = 0.
do i = 1,nx,1
  if(d(i).gt.0.0) then
    im = max(1,i-1)
    div(i) = -dt*(1.-wimp)*(uint(i)-uint(im))/dx  - dt*wimp*(uints(i)-uints(im))/dx
  endif
enddo

! now compute coefficients for zetan system of equations: ce and cw

ce = 0.
cw = 0.

do i = 1,nx,1
  im = max(1,i-1)
  ip = min(nx,i+1)
  
  if((d(i).gt.0.0).and.(d(ip).gt.0.0)) then
    ce(i) = dt*dt*wimp*wimp*g*0.5*(h(i)+h(ip))/(dx*dx)
  endif
  
  if((d(i).gt.0.0).and.(d(im).gt.0.0)) then
    cw(i) = dt*dt*wimp*wimp*g*0.5*(h(i)+h(im))/(dx*dx)
  endif
  
enddo

! iterative solution of the system of eqations

it = 0

zs = zeta

1 continue
maxdif = 0.
it = it+1

do i = 1,nx,1
  im = max(1,i-1)
  ip = min(nx,i+1)
  if(d(i).gt.0.1) then
    zetnew = 1./(1. + ce(i) + cw(i))*(zeta(i) + div(i) + ce(i)*zetan(ip) + cw(i)*zetan(im))
	
	dif = abs(zetnew-zs(i))
	maxdif = max(maxdif,dif)
	zs(i) = zetnew
  else
    zs(i) = 0.
  endif
enddo

zetan = zs ! update iterative solution (could be also done in the loop above, but would lead to problems when parallelisation is applied.)

!write(*,*) 'it,maxdif',it,maxdif
!read(*,*)

if(maxdif.gt.1.e-6) goto 1 ! convergence check

write(*,*) 'iterations: ',it

! ===============================================================
! finally compute compute velocity at new time step un

do i = 1,nx,1
  ip = min(nx,i+1)
  if((d(i).gt.0.1).and.(d(ip).gt.0.1)) then
    un(i) = us(i) - wimp*dt*g*(zetan(ip)-zetan(i))/dx 
  else
    un(i) = 0.
  endif
enddo

! ================================================================
! swap time level

u = un
zeta = zetan

do i = 1,nx,1
  if(d(i).gt.0.1) then
    h(i) = d(i) + zeta(i)
  endif
enddo
 
! ================================================
! compute min,max for screen output

umin = 1.e6
umax = -1.e6
zmin = 1.e6
zmax = -1.e6

do i = 1,nx,1
  if(d(i).gt.0.1) then
    umin = min(umin,u(i))
	umax = max(umax,u(i))
	zmin = min(zmin,zeta(i))
	zmax = max(zmax,zeta(i))
  endif
enddo

write(*,*) 'rminute,umin,umax,zmin,zmax',rminute,umin,umax,zmin,zmax


! =========================================================
! write hourly output

if(minute.ne.ho) then
  ho = minute
  call output(ho)
endif

! write gauge data output

write(66,'(2f12.4)') rminute,zeta(nx-1)

! ================================================

if(rminute.lt.9000) goto 100 ! continuation criterium of time loop

stop

contains

! ========================================================================

  subroutine output(ho)
  integer ho
  character*50 ufile,zfile
  
  ufile = '../data/raw/AH=5e+4/udata/uHHHH.dat'
  zfile = '../data/raw/AH=5e+4/zdata/zHHHH.dat'

  write(ufile(15:18),'(i4.4)') ho
  write(zfile(15:18),'(i4.4)') ho
  
  open(1, file = ufile)
  do i = 1,nx,1
    write(1,'(f12.4)') u(i)
  enddo
  close(1)
  
  open(1, file = zfile)
  do i = 1,nx,1
    write(1,'(f12.4)') zeta(i)
  enddo
  close(1)
  
  return
  end subroutine output
  

end program shallow_water
  





