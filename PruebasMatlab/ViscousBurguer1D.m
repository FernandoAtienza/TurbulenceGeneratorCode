% 1D Viscous Burgers: Compact 8th order FD + RK3 (single step)
clc; clear; close all;

% Numerical scheme:
% 0 -> 8th order compact FD
% 1 -> 7th order WENO FV
scheme = 0;

% Spatial input parameters
L_min = -1; L_max = 1;
dx = 1/15;                      % Grid spacing                          
Nx = (L_max - L_min)/dx + 1;    % Number of grid points       
x = linspace(L_min, L_max, Nx);     

% Time input parameters
t_final = 0.5;                  % Final time
dt = 0.001;                     % Time step
Nt = ceil(t_final/dt);          % Number of time steps

% Viscosity input
nu = 0.01/pi;   

% Initial condition
u0 = -sin(pi*x);
u = u0(:);  % column vector for matrix operations

% First derivative coefficients
alpha2 = 4/9;
beta2 = 1/36;
a2 = 20/27;
b2 = 25/216;

% Second derivative coefficients
alpha3 = 344/1179;
beta3  = (38*alpha3 - 9)/214;
a3     = (696 - 1191*alpha3)/438;
b3     = (1227*alpha3 - 147)/1070;

% Get matrices
A_x  = zeros(Nx,Nx);
A_xx = zeros(Nx,Nx);
for i = 1:Nx
    [im2, im1, ip1, ip2] = GetIndices(i, Nx);
    
    % First derivative
    A_x(i, im2) = beta2;  A_x(i, im1) = alpha2;  A_x(i, i) = 1;
    A_x(i, ip1) = alpha2; A_x(i, ip2) = beta2;
    
    % Second derivative
    A_xx(i, im2) = beta3;  A_xx(i, im1) = alpha3;  A_xx(i, i) = 1;
    A_xx(i, ip1) = alpha3; A_xx(i, ip2) = beta3;
end

% Solution update in time
time = 0;
for n = 1:Nt
    % Numerical solution
    u = RK3_update(u, Nx, dx, dt, nu, A_x, A_xx, a2, a3, b2, b3, scheme);

    time = n*dt;

    % Analytical solution
    if n == 0
        u_an = u0;
    else 
        u_an = Burgers_Analytical(x, time, nu);
    end

    % Plot every 25 steps
    if mod(n,25) == 0
        plot(x, u0, 'b', x, u, 'ro--', x, u_an, 'k');
        xlabel('x'); ylabel('u'); grid minor;
        legend('Initial u0','u at time t', 'Analytical sol.');
        title(sprintf('t = %.3f s', time));
        if scheme == 0
            title(sprintf('8th order FD at t = %.3f s', time));
        elseif scheme == 1
            title(sprintf('WENO at t = %.3f s', time));
        end
        subtitle(sprintf('$\\Delta x = %.4f \\quad \\nu = %.4f$', dx, nu));
        drawnow
    end
    if n == Nt && scheme == 0
        data = Get_8FD_WangData;
        hold on
        plot (data(:,1), data(:,2), "g*");
        legend('Initial u0','u at time t', 'Analytical sol.', 'Wang et. al. 2010');
        hold off
    elseif n == Nt && scheme == 1
        data = GetWENO_WangData;
        hold on
        plot (data(:,1), data(:,2), "g*");
        legend('Initial u0','u at time t', 'Analytical sol.', 'Wang et. al. 2010');
        hold off
    end
end

%% Get stencil indices since function is periodic
function [im2, im1, ip1, ip2] = GetIndices(i, Nx)
    % Periodic BCs (wrap-around)
    if i == 1
        im1 = Nx;
    else 
        im1 = i-1; 
    end

    if i <= 2
        im2 = i-2+Nx;
    else 
        im2 = i-2;
    end

    if i == Nx
        ip1 = 1;
    else 
        ip1 = i+1;
    end

    if i >= Nx-1
        ip2 = i+2-Nx;
    else
        ip2 = i+2;
    end
end

%% Right-hand-side of 1D Burguers Eqn.
function u_t = RHS_Burgers(u, Nx, dx, nu, A_x, A_xx, a2, a3, b2, b3, scheme)
    % Compute compact FD derivatives and RHS
    RHS_x  = zeros(Nx,1);
    RHS_xx = zeros(Nx,1);
    
    for i = 1:Nx
        [im2, im1, ip1, ip2] = GetIndices(i, Nx);
        RHS_x(i)  = (a2/dx)*(u(ip1) - u(im1)) + (b2/dx)*(u(ip2) - u(im2));
        RHS_xx(i) = (a3/dx^2)*(u(ip1) - 2*u(i) + u(im1)) + (b3/dx^2)*(u(ip2) - 2*u(i) + u(im2));
    end
    
    u_xx = A_xx \ RHS_xx;
    
    if scheme == 0
        % 8th order FD 
        u_x  = A_x  \ RHS_x;
        u_t = -u .* u_x + nu * u_xx;
    elseif scheme == 1
        % 7th order WENO
        f = 0.5 * u.^2;
        f_x = WENO7_FluxDerivative(f, dx);
        u_t = -f_x + nu * u_xx;
    end
end

%% Runge-kutta 3 solution update
function u = RK3_update(u, Nx, dx, dt, nu, A_x, A_xx, a2, a3, b2, b3, scheme)
    % Third-order TVD Runge-Kutta single step
    k1 = RHS_Burgers(u, Nx, dx, nu, A_x, A_xx, a2, a3, b2, b3, scheme);
    u1 = u + dt*k1;
    
    k2 = RHS_Burgers(u1, Nx, dx, nu, A_x, A_xx, a2, a3, b2, b3, scheme);
    u2 = (3/4)*u + (1/4)*(u1 + dt*k2);
    
    k3 = RHS_Burgers(u2, Nx, dx, nu, A_x, A_xx, a2, a3, b2, b3, scheme);
    u = (1/3)*u + (2/3)*(u2 + dt*k3);
end

%% First spatial derivative 7-th order WENO
function f_x = WENO7_FluxDerivative(u, dx)
% WENO7 (r=4) flux derivative for Burgers equation
    % Input:
    %   u  : Nx x 1 column vector
    %   dx : grid spacing
    % Output:
    %   f_x : Nx x 1 vector approximating d/dx (u^2/2)

    Nx = length(u);
    f_x = zeros(Nx,1);
    
    epsilon = 1e-6;
    
    % Flux
    f = 0.5*u.^2;
    
    % Lax-Friedrichs flux splitting
    alpha = max(abs(u));
    f_plus  = 0.5*(f + alpha*u);
    f_minus = 0.5*(f - alpha*u);
    
    % Optimal weights
    C = [1/35, 12/35, 18/35, 4/35];
    
    % Preallocate interface fluxes
    fph = zeros(Nx,1); % f_{i+1/2}
    fmh = zeros(Nx,1); % f_{i-1/2}
    
    % ===== LEFT-BIASED RECONSTRUCTION (f_plus) =====
    for i = 1:Nx
        % Indices
        im3 = mod(i-4,Nx)+1;
        im2 = mod(i-3,Nx)+1;
        im1 = mod(i-2,Nx)+1;
        ip0 = i;
        ip1 = mod(i,Nx)+1;
        ip2 = mod(i+1,Nx)+1;
        ip3 = mod(i+2,Nx)+1;
    
        v = f_plus;
    
        % Candidate reconstructions
        q0 = -(1/4)*v(im3) + 13/12*v(im2) - 23/12*v(im1) + 25/12*v(ip0);
        q1 =  1/12*v(im2) - 5/12*v(im1) + 13/12*v(ip0) + 1/4*v(ip1);
        q2 = -1/12*v(im1) + 7/12*v(ip0) + 7/12*v(ip1) - 1/12*v(ip2);
        q3 =  1/4*v(ip0) + 13/12*v(ip1) - 5/12*v(ip2) + 1/12*v(ip3);
    
        % Smoothness indicators
        IS0 = 547*v(im3)^2 - 3882*v(im3)*v(im2) + 4642*v(im3)*v(im1) - 1854*v(im3)*v(ip0) + ...
              7043*v(im2)^2 - 17246*v(im2)*v(im1) + 7042*v(im2)*v(ip0) + ...
              11003*v(im1)^2 - 9402*v(im1)*v(ip0) + 2107*v(ip0)^2;
    
        IS1 = 267*v(im2)^2 - 1642*v(im2)*v(im1) + 1602*v(im2)*v(ip0) - 494*v(im2)*v(ip1) + ...
              2843*v(im1)^2 - 5966*v(im1)*v(ip0) + 1922*v(im1)*v(ip1) + ...
              3443*v(ip0)^2 - 2522*v(ip0)*v(ip1) + 547*v(ip1)^2;
    
        IS2 = 547*v(im1)^2 - 2522*v(im1)*v(ip0) + 1922*v(im1)*v(ip1) - 494*v(im1)*v(ip2) + ...
              3443*v(ip0)^2 - 5966*v(ip0)*v(ip1) + 1602*v(ip0)*v(ip2) + ...
              2843*v(ip1)^2 - 1642*v(ip1)*v(ip2) + 267*v(ip2)^2;
    
        IS3 = 2107*v(ip0)^2 - 9402*v(ip0)*v(ip1) + 7042*v(ip0)*v(ip2) - 1854*v(ip0)*v(ip3) + ...
              11003*v(ip1)^2 - 17246*v(ip1)*v(ip2) + 4642*v(ip1)*v(ip3) + ...
              7043*v(ip2)^2 - 3882*v(ip2)*v(ip3) + 547*v(ip3)^2;
    
        IS = [IS0, IS1, IS2, IS3];
    
        alpha_w = C ./ (epsilon + IS).^2;
        omega = alpha_w / sum(alpha_w);
    
        fph(i) = omega(1)*q0 + omega(2)*q1 + omega(3)*q2 + omega(4)*q3;
    end
    
    % RIGHT-BIASED RECONSTRUCTION (f_minus)
    for i = 1:Nx
        % Indices (mirror stencil)
        ip3 = mod(i+3,Nx)+1;
        ip2 = mod(i+2,Nx)+1;
        ip1 = mod(i+1,Nx)+1;
        ip0 = i;
        im1 = mod(i-2,Nx)+1;
        im2 = mod(i-3,Nx)+1;
        im3 = mod(i-4,Nx)+1;
    
        v = f_minus;
    
        % Mirror reconstruction (reverse roles)
        q0 = -(1/4)*v(ip3) + 13/12*v(ip2) - 23/12*v(ip1) + 25/12*v(ip0);
        q1 =  1/12*v(ip2) - 5/12*v(ip1) + 13/12*v(ip0) + 1/4*v(im1);
        q2 = -1/12*v(ip1) + 7/12*v(ip0) + 7/12*v(im1) - 1/12*v(im2);
        q3 =  1/4*v(ip0) + 13/12*v(im1) - 5/12*v(im2) + 1/12*v(im3);
    
        % Smoothness indicators (reuse same form)
        IS0 = 547*v(ip3)^2 - 3882*v(ip3)*v(ip2) + 4642*v(ip3)*v(ip1) - 1854*v(ip3)*v(ip0) + ...
              7043*v(ip2)^2 - 17246*v(ip2)*v(ip1) + 7042*v(ip2)*v(ip0) + ...
              11003*v(ip1)^2 - 9402*v(ip1)*v(ip0) + 2107*v(ip0)^2;
    
        IS1 = 267*v(ip2)^2 - 1642*v(ip2)*v(ip1) + 1602*v(ip2)*v(ip0) - 494*v(ip2)*v(im1) + ...
              2843*v(ip1)^2 - 5966*v(ip1)*v(ip0) + 1922*v(ip1)*v(im1) + ...
              3443*v(ip0)^2 - 2522*v(ip0)*v(im1) + 547*v(im1)^2;
    
        IS2 = 547*v(ip1)^2 - 2522*v(ip1)*v(ip0) + 1922*v(ip1)*v(im1) - 494*v(ip1)*v(im2) + ...
              3443*v(ip0)^2 - 5966*v(ip0)*v(im1) + 1602*v(ip0)*v(im2) + ...
              2843*v(im1)^2 - 1642*v(im1)*v(im2) + 267*v(im2)^2;
    
        IS3 = 2107*v(ip0)^2 - 9402*v(ip0)*v(im1) + 7042*v(ip0)*v(im2) - 1854*v(ip0)*v(im3) + ...
              11003*v(im1)^2 - 17246*v(im1)*v(im2) + 4642*v(im1)*v(im3) + ...
              7043*v(im2)^2 - 3882*v(im2)*v(im3) + 547*v(im3)^2;
    
        IS = [IS0, IS1, IS2, IS3];
    
        alpha_w = C ./ (epsilon + IS).^2;
        omega = alpha_w / sum(alpha_w);
    
        fmh(i) = omega(1)*q0 + omega(2)*q1 + omega(3)*q2 + omega(4)*q3;
    end
    
    % Final derivative
    for i = 1:Nx
        im1 = mod(i-2,Nx)+1;
        f_x(i) = (fph(i) - fph(im1) + fmh(i) - fmh(im1)) / dx;
    end

end
%% 1D Viscous Burgers Analyticalsolution
function u_exact = Burgers_Analytical(x, t, nu)
    % Analytical solution of 1D viscous Burgers equation
    % u_t + u u_x = nu u_xx
    % Initial condition: u(x,0) = -sin(pi*x)
    % Domain: periodic [-1,1]
    
    Nx = length(x);
    u_exact = zeros(size(x));
    
    % Integration grid (for convolution integral)
    Ng = 200;                         % increase for more accuracy
    g = linspace(-1,1,Ng);
    dg = g(2) - g(1);
    
    for i = 1:Nx
        xi = x(i);
        
        num = 0;
        den = 0;
        
        for k = 1:Ng
            y = xi - g(k);
            
            % Periodicity (wrap into [-1,1])
            y = mod(y+1,2) - 1;
            
            % f(y) from Cole-Hopf
            f = exp(-cos(pi*y)/(2*pi*nu));
            
            G = exp(-(g(k)^2)/(4*nu*t));   % Gaussian kernel
            
            num = num + sin(pi*y)*f*G;
            den = den + f*G;
        end
        
        u_exact(i) = -num/den;
    end

end

%% WENO data from Wang 2010
function data = GetWENO_WangData()
    data = [
    -0.9997075704692436, -0.012355914499926568;
    -0.9327070643573699,  0.07660619479062292;
    -0.7987060521336227,  0.2397033951566303;
    -0.6681409895142429,  0.3929159544196337;
    -0.4654215356974607,  0.6252058687490112;
    -0.33313838031282916, 0.768533786909011;
    -0.26613787420095547, 0.8327841237602933;
    -0.19741951124996604, 0.8871499326297432;
    -0.13041900513809235, 0.9316309872750179;
    -0.06685436998854088, 0.9019769508448348;
     0.00014613612333280734, 0.012355857939340575;
     0.0671466422352065, -0.901977007405421;
     0.1341471483470802, -0.9464580620506957;
     0.19427575525021856, -0.8871499891903294;
     0.26814816057082735, -0.8278419163299631;
     0.33343065255949456, -0.7734761074605129;
     0.40214917279457474, -0.694398752394139;
     0.5344321708951156, -0.5510708342341393;
     0.6649973121565407, -0.39785838809230745;
     0.8007163385034946, -0.2397034517172163;
     0.9329994938881261, -0.07660625135120891
    ];
end

function data2 = Get_8FD_WangData ()
    data2 = [
    -1,                    0.0062937062937062915;
    -0.9330909090909092,   0.04825174825174816;
    -0.8661818181818182,   0.24125874125874125;
    -0.8007272727272727,   0.11118881118881108;
    -0.7338181818181819,   0.5139860139860142;
    -0.6669090909090909,   0.1447552447552447;
    -0.6000000000000001,   0.790909090909091;
    -0.5330909090909091,   0.17832167832167833;
    -0.46763636363636374,  1.0342657342657344;
    -0.3992727272727272,   0.25804195804195795;
    -0.3338181818181818,   1.2188811188811188;
    -0.266909090909091,    0.3713286713286714;
    -0.19854545454545458,  1.348951048951049;
    -0.13163636363636355,  0.4636363636363636;
    -0.06618181818181823,  1.4328671328671327;
    -0.0007272727272726875,0.0062937062937062915;
     0.06618181818181812, -1.4202797202797204;
     0.13454545454545452, -0.45524475524475516;
     0.19854545454545436, -1.3363636363636364;
     0.266909090909091,   -0.36293706293706296;
     0.33527272727272717, -1.2146853146853147;
     0.4007272727272726,  -0.24125874125874125;
     0.46618181818181825, -1.0258741258741257;
     0.5345454545454542,  -0.17832167832167833;
     0.5985454545454547,  -0.7825174825174825;
     0.6669090909090907,  -0.13216783216783212;
     0.7323636363636365,  -0.5055944055944056;
     0.8007272727272723,  -0.10699300699300696;
     0.8676363636363635,  -0.23706293706293713;
     0.9330909090909087,  -0.03986013986013992
    ];
end 