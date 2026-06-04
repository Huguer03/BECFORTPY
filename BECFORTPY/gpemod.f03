! para compilar: python -m numpy.f2py -c gpemod.f03 -m gpe_solver -L$CONDA_PREFIX/lib -lfftw3 --f90flags="-I$CONDA_PREFIX/include"

module gpe_solver
    use iso_c_binding
    implicit none

    include 'fftw3.f03'

    complex(8), parameter :: zi = (0.0d0, 1.0d0)
    type(C_PTR), private  :: plan_foreward_1d_x, plan_backward_1d_x
    type(C_PTR), private  :: plan_foreward_1d_y, plan_backward_1d_y
    type(C_PTR), private  :: plan_foreward_2d, plan_backward_2d

contains

    subroutine fftw_create_plans_1d(nx, ny)
        integer, intent(in)       :: nx,ny
        complex(8), allocatable   :: phi(:,:)
        integer :: n(1)

        allocate(phi(nx,ny))
        n(1) = nx
        plan_foreward_1d_x = fftw_plan_many_dft(1, n, ny, &
            phi, n, 1, nx, phi, n, 1, nx, FFTW_FORWARD, FFTW_PATIENT)
        plan_backward_1d_x = fftw_plan_many_dft(1, n, ny, &
            phi, n, 1, nx, phi, n, 1, nx, FFTW_BACKWARD, FFTW_PATIENT)

        n(1) = ny
        plan_foreward_1d_y = fftw_plan_many_dft(1, n, nx, &
            phi, n, nx, 1, phi, n, nx, 1, FFTW_FORWARD, FFTW_PATIENT)
        plan_backward_1d_y = fftw_plan_many_dft(1, n, nx, &
            phi, n, nx, 1, phi, n, nx, 1, FFTW_BACKWARD, FFTW_PATIENT)
        deallocate(phi)
    end subroutine fftw_create_plans_1d

    subroutine fftw_create_plans_2d(nx, ny)
        integer, intent(in)     :: nx,ny
        complex(8), allocatable :: phi(:,:)

        allocate(phi(nx,ny))
        plan_foreward_2d  = fftw_plan_dft_2d(nx, ny, phi, phi, FFTW_FORWARD, FFTW_PATIENT)
        plan_backward_2d = fftw_plan_dft_2d(nx, ny, phi, phi, FFTW_BACKWARD, FFTW_PATIENT)
        deallocate(phi)
    end subroutine fftw_create_plans_2d

    subroutine fftw_create_plans(nx, ny, dim)
        integer, intent(in) :: nx,ny,dim

        if (dim == 1) then
            call fftw_create_plans_1d(nx, ny)
        elseif (dim == 2) then
            call fftw_create_plans_2d(nx, ny)
        else 
            call fftw_create_plans_1d(nx, ny)
            call fftw_create_plans_2d(nx, ny)
        end if
    end subroutine fftw_create_plans

    subroutine fftw_destroy_plans(dim)
        integer, intent(in) :: dim

        if ( dim == 1 ) then
            call fftw_destroy_plan(plan_foreward_1d_x)
            call fftw_destroy_plan(plan_backward_1d_x)
            call fftw_destroy_plan(plan_foreward_1d_y)
            call fftw_destroy_plan(plan_backward_1d_y)
        elseif ( dim == 2) then
            call fftw_destroy_plan(plan_foreward_2d)
            call fftw_destroy_plan(plan_backward_2d)
        else 
            call fftw_destroy_plan(plan_foreward_1d_x)
            call fftw_destroy_plan(plan_backward_1d_x)
            call fftw_destroy_plan(plan_foreward_1d_y)
            call fftw_destroy_plan(plan_backward_1d_y)
            call fftw_destroy_plan(plan_foreward_2d)
            call fftw_destroy_plan(plan_backward_2d)
        end if
    end subroutine fftw_destroy_plans

    subroutine fft(phi, nx, ny, axis, direction)
        complex(8), intent(inout) :: phi(nx, ny)
        integer, intent(in)       :: nx, ny, direction, axis

        if (axis == 1) then 
            if (direction == 1) then
                call fftw_execute_dft(plan_foreward_1d_x, phi, phi)
                phi = phi / sqrt(real(nx, 8))
            else
                call fftw_execute_dft(plan_backward_1d_x, phi, phi)
                phi = phi / sqrt(real(nx, 8))
            end if
        else if (axis == 2) then
            if (direction == 1) then
                call fftw_execute_dft(plan_foreward_1d_y, phi, phi)
                phi = phi / sqrt(real(ny, 8))
            else
                call fftw_execute_dft(plan_backward_1d_y, phi, phi)
                phi = phi / sqrt(real(ny, 8))
            end if
        end if
    end subroutine fft

    subroutine fft2(phi, nx, ny, direction)
        complex(8), intent(inout) :: phi(nx, ny)
        integer, intent(in)       :: nx, ny, direction

        if (direction == 1) then
            call fftw_execute_dft(plan_foreward_2d, phi, phi)
            phi = phi / sqrt(real(nx * ny, 8))
        else
            call fftw_execute_dft(plan_backward_2d, phi, phi)
            phi = phi / sqrt(real(nx * ny, 8))
        end if
    end subroutine fft2

    subroutine gradient_descent_step(phi, v, kx, ky, k2, x, y, nx, ny, dx, dy, &
                                    beta, omega, dt)
        complex(8), intent(inout) :: phi(nx, ny)
        real(8), intent(in) :: v(nx, ny), kx(nx, ny), ky(nx, ny), k2(nx, ny)
        real(8), intent(in) :: x(nx, ny), y(nx, ny)
        integer, intent(in) :: nx, ny
        real(8), intent(in) :: dx, dy, beta, omega, dt
        
        complex(8) :: phi_k(nx, ny), grad(nx, ny)
        complex(8) :: dx_phi(nx, ny), dy_phi(nx, ny)
        complex(8) :: temp(nx, ny)
        real(8) :: norm

        phi_k = phi
        call fft2(phi_k, nx, ny, 1)
        dx_phi = zi * kx * phi_k
        dy_phi = zi * ky * phi_k

        temp = -k2 * phi_k
        call fft2(temp, nx, ny, -1)

        grad = -0.5d0 * temp + v * phi + beta * abs(phi)**2 * phi

        if (omega /= 0.0d0) then
            call fft2(dx_phi, nx, ny, -1)
            call fft2(dy_phi, nx, ny, -1)

            grad = grad - zi * omega * (x * dy_phi - y * dx_phi)
        end if

        phi = phi - dt * grad 

        norm = sum(abs(phi)**2) * dx * dy
        phi = phi / sqrt(norm)
    end subroutine gradient_descent_step

    function energy(phi, v, kx, ky, x, y, nx, ny, dx, dy,&
                                     beta, omega) result(E)
        complex(8), intent(in) :: phi(nx, ny)
        real(8), intent(in) :: v(nx, ny), kx(nx, ny), ky(nx, ny)
        real(8), intent(in) :: x(nx, ny), y(nx, ny)
        integer, intent(in) :: nx, ny
        real(8), intent(in) :: dx, dy, beta, omega
        
        complex(8) :: phi_k(nx, ny)
        complex(8) :: dx_phi(nx, ny), dy_phi(nx, ny)
        complex(8) :: Lz_phi(nx,ny)
        complex(8) :: expextation
        real(8) :: laplacian(nx,ny)
        real(8) :: E_kin, E_pot, E_beta, E_rot, E

        phi_k = phi
        call fft2(phi_k, nx, ny, 1)

        dx_phi = zi * kx * phi_k
        dy_phi = zi * ky * phi_k
        call fft2(dx_phi, nx, ny, -1)
        call fft2(dy_phi, nx, ny, -1)

        laplacian = (abs(dx_phi)**2 + abs(dy_phi)**2)
        E_kin     = 0.5d0 * sum(laplacian) * dx * dy

        E_pot = sum(abs(phi)**2 * v) * dx * dy

        E_beta = 0.5d0 * beta * sum(abs(phi)**4) * dx * dy

        E = E_kin + E_pot + E_beta

        if (omega /= 0.0d0) then
            Lz_phi = -zi * (x * dy_phi - y * dx_phi)
            expextation = sum(conjg(phi) * Lz_phi) * dx * dy
            E_rot       = -omega * real(expextation)
        else 
            E_rot = 0.0d0
        end if

        E = E + E_rot
    end function energy

    subroutine gradient_descent_evol(phi, v, kx, ky, k2, x, y, nx, ny, dx, dy, &
                                    beta, omega, dt, max_iter, tol, converge)
        complex(8), intent(inout) :: phi(nx, ny)
        real(8), intent(in)  :: v(nx, ny), kx(nx, ny), ky(nx, ny), k2(nx, ny)
        real(8), intent(in)  :: x(nx, ny), y(nx, ny)
        integer, intent(in)  :: nx, ny
        real(8), intent(in)  :: dx, dy, beta, omega, dt
        logical, intent(out) :: converge
        integer, optional    :: max_iter
        real(8), optional    :: tol

        real(8)    :: E_rel, E_new, E_old, norm
        integer    :: i

        if (.not. present(max_iter)) max_iter = 100000
        if (.not. present(tol)) tol = 1e-6

        call fftw_create_plans(nx, ny, 2)

        E_old = energy(phi, v, kx, ky, x, y, nx, ny, dx, dy, beta, omega)

        do i = 1, max_iter
            call gradient_descent_step(phi, v, kx, ky, k2, x, y, nx, ny,&
                                         dx, dy, beta, omega, dt)
            if ( modulo(i,10) == 0 ) then
                norm  = sum(abs(phi)**2) * dx * dy
                phi   = phi / sqrt(norm)
                E_new = energy(phi, v, kx, ky, x, y, nx, ny, dx, dy, beta, omega)
                E_rel = abs(E_new - E_old) / E_old
                E_old = E_new
                if ( E_rel < tol ) then
                    print*, "Iterations needed for convergence: ", i
                    converge = .true.
                    return
                end if
            end if
        end do

        if ( i == max_iter ) then
            converge = .false.
        end if

        call fftw_destroy_plans(2)
    end subroutine gradient_descent_evol 

    subroutine rot(phi, angle, kx, ky, x, y, nx, ny)
        complex(8), intent(inout) :: phi(nx,ny)
        real(8), intent(in)       :: kx(nx,ny), ky(nx,ny)
        real(8), intent(in)       :: x(nx,ny), y(nx,ny)
        real(8), intent(in)       :: angle
        integer, intent(in)       :: nx, ny

        real(8) :: alpha, beta
        integer :: j, i
        real(8) :: kx_vec(nx), ky_vec(ny)
        real(8) :: x_vec(nx), y_vec(ny)

        kx_vec = kx(:, 1)
        ky_vec = ky(1, :)
        x_vec  = x(:, 1)
        y_vec  = y(1, :)

        alpha = -tan(angle / 2.0d0)
        beta  = sin(angle)

        call fft(phi, nx, ny, 1, 1)
        do j = 1, ny
            phi(:, j) = phi(:, j) * exp(-zi * kx_vec * (alpha * y_vec(j)))
        end do
        call fft(phi, nx, ny, 1, -1)

        call fft(phi, nx, ny, 2, 1)
        do i = 1, nx
            phi(i, :) = phi(i, :) * exp(-zi * ky_vec * (beta * x_vec(i)))
        end do
        call fft(phi, nx, ny, 2, -1)

        call fft(phi, nx, ny, 1, 1)
        do j = 1, ny
            phi(:, j) = phi(:, j) * exp(-zi * kx_vec * (alpha * y_vec(j)))
        end do
        call fft(phi, nx, ny, 1, -1)
    end subroutine rot

    subroutine ssfm_step(phi, v, kx, ky, k2, x, y, nx, ny, &
                                    beta, angle, dt)
        complex(8), intent(inout) :: phi(nx,ny)
        real(8), intent(in)       :: v(nx,ny)
        real(8), intent(in)       :: x(nx,ny), y(nx,ny)
        real(8), intent(in)       :: kx(nx,ny), ky(nx,ny)
        real(8), intent(in)       :: k2(nx,ny)
        real(8), intent(in)       :: dt, beta, angle
        integer, intent(in)       :: nx, ny

        phi   = exp(-zi * (v + beta * abs(phi)**2) * dt * 0.5d0) * phi

        if (angle /= 0.0d0) then
            call fft2(phi, nx, ny, 1)
            phi = exp(-0.25d0 * zi * k2 * dt) * phi
            call fft2(phi, nx, ny, -1)

            call rot(phi, angle, kx, ky, x, y, nx, ny)

            call fft2(phi, nx, ny, 1)
            phi = exp(-0.25d0 * zi * k2 * dt) * phi
            call fft2(phi, nx, ny, -1)

        else
            call fft2(phi, nx, ny, 1)
            phi = exp(-0.5d0 * zi * k2 * dt) * phi
            call fft2(phi, nx, ny, -1)
        end if

        phi   = exp(-zi * (v + beta * abs(phi)**2) * dt * 0.5d0) * phi
    end subroutine ssfm_step

    subroutine ssfm_evol(phi, v, kx, ky, k2, x, y, nx, ny,&
                                    beta, omega, final_time, dt)
        complex(8), intent(inout) :: phi(nx,ny)
        real(8), intent(in)       :: v(nx,ny)
        real(8), intent(in)       :: x(nx,ny), y(nx,ny)
        real(8), intent(in)       :: kx(nx,ny), ky(nx,ny)
        real(8), intent(in)       :: k2(nx,ny)
        real(8), intent(in)       :: dt, beta, omega, final_time
        integer, intent(in)       :: nx, ny

        real(8) :: angle
        integer :: i, steps

        call fftw_create_plans(nx, ny, 0)
        
        steps = int(final_time / dt)
        angle = omega * dt

        do i = 1,steps
            call ssfm_step(phi, v, kx, ky, k2, x, y, nx, ny, &
                                    beta, angle, dt)
        enddo
        
        call fftw_destroy_plans(0)
    end subroutine ssfm_evol

end module gpe_solver