%% ==================== MAIN SCRIPT =========================
clc; clear; close all;

% --- Parameters, grids, initial condition ---
L_min = -1.0; L_max = 1.0;
nx = 31; dx = 1/15; dt = 0.001; nu = 0.01/pi;
T_frames = 50; steps_per_frame = 10;

x = linspace(L_min,L_max,nx)';
u_initial = -sin(pi*x);
u = u_initial;

% --- Tridiagonal matrix for compact scheme ---
alpha1 = 3/8;
e = ones(nx,1);
A = spdiags([alpha1*e, e, alpha1*e], [-1 0 1], nx, nx);
A(1,end) = alpha1; A(end,1) = alpha1;
[L_fac,U_fac] = lu(A);
solve_A = @(R) U_fac\(L_fac\R);


% --- Animation setup ---
figure('Color','w'); hold on;
line_ana = plot(x,u,'b-','LineWidth',2);
scatter_num = scatter(x,u,50,'g','filled');
scatter([],[],'g','filled'); scatter([],[],'r','filled'); % Legend proxies
xlim([L_min L_max]); ylim([-1.2 1.2]);
xlabel('x'); ylabel('u');
legend({'Analytical (Cole-Hopf)','Smooth (Compact FD)','Shock (WENO-5)'},'Location','northeast');
grid on;

current_shock_region = false(size(u));

% --- Time-stepping + animation ---
for frame = 1:T_frames
    for step = 1:steps_per_frame
        u1 = u + dt*RHS_hybrid(u,dx,nu,solve_A,alpha1);
        u2 = 0.75*u + 0.25*(u1 + dt*RHS_hybrid(u1,dx,nu,solve_A,alpha1));
        u  = (1/3)*u + (2/3)*(u2 + dt*RHS_hybrid(u2,dx,nu,solve_A,alpha1));
    end
    
    t_current = frame*steps_per_frame*dt;
    
    % Update scatter colors
    colors = repmat([0 1 0], length(u),1);  % green
    shock_idx = current_shock_region;
    colors(shock_idx,:) = repmat([1 0 0], sum(shock_idx),1); % red
   
    delete(scatter_num);
    scatter_num = scatter(x,u,50,colors,'filled');
    
    title(sprintf('Hybrid Scheme (Compact FD + WENO) — t = %.3f', t_current));
    drawnow;
end

%% ==================== FUNCTIONS ============================
function dudt = RHS_hybrid(u, dx, nu, solve_A, alpha1)
    nx = length(u);
    theta = (circshift(u,-1)-circshift(u,1))/(2*dx);
    theta_rms = sqrt(mean(theta.^2));
    is_shock_node = theta < -3*theta_rms;
    shock_region = is_shock_node;
    for k=1:3
        shock_region = shock_region | circshift(is_shock_node,k) | circshift(is_shock_node,-k);
    end
    assignin('base','current_shock_region',shock_region);
    
    f = 0.5*u.^2;
    a1_c = 25/32; b1_c = 1/20; c1_c = -1/480;
    F_comp = c1_c*(circshift(f,-3)+circshift(f,2)) + ...
             (b1_c+c1_c)*(circshift(f,-2)+circshift(f,1)) + ...
             (a1_c+b1_c+c1_c)*(circshift(f,-1)+f);
    F_weno_raw = get_weno_interface_flux(u);
    F_weno_hat = alpha1*circshift(F_weno_raw,1) + F_weno_raw + alpha1*circshift(F_weno_raw,-1);
    
    is_shock_edge = shock_region & circshift(shock_region,-1);
    is_smooth_edge = (~shock_region) & (~circshift(shock_region,-1));
    is_joint_edge = ~(is_shock_edge | is_smooth_edge);
    
    F_hybrid = zeros(nx,1);
    F_hybrid(is_smooth_edge) = F_comp(is_smooth_edge);
    F_hybrid(is_shock_edge) = F_weno_hat(is_shock_edge);
    F_hybrid(is_joint_edge) = 0.5*(F_comp(is_joint_edge)+F_weno_hat(is_joint_edge));
    
    R = (F_hybrid - circshift(F_hybrid,1))/dx;
    advection = solve_A(R);
    
    diffusion = nu * ( -(1/560)*circshift(u,-4) + (8/315)*circshift(u,-3) ...
                       - (1/5)*circshift(u,-2) + (8/5)*circshift(u,-1) ...
                       - (205/72)*u + (8/5)*circshift(u,1) - (1/5)*circshift(u,2) ...
                       + (8/315)*circshift(u,3) - (1/560)*circshift(u,4) ) / dx^2;
    dudt = -advection + diffusion;
end

function F_h = get_weno_interface_flux(u)
    nx = length(u);
    f = 0.5*u.^2;
    alpha = max(abs(u));
    fp = 0.5*(f+alpha*u);
    fm = 0.5*(f-alpha*u);
    v1_p = circshift(fp,2); v2_p = circshift(fp,1); v3_p = fp; v4_p = circshift(fp,-1); v5_p = circshift(fp,-2);
    fp_half = weno5_flux(v1_p,v2_p,v3_p,v4_p,v5_p);
    v1_m = circshift(fm,-3); v2_m = circshift(fm,-2); v3_m = circshift(fm,-1); v4_m = fm; v5_m = circshift(fm,1);
    fm_half = weno5_flux(v1_m,v2_m,v3_m,v4_m,v5_m);
    F_h = fp_half + fm_half;
end

function F = weno5_flux(v1,v2,v3,v4,v5)
    eps = 1e-6;
    q0 = (1/3)*v1 - (7/6)*v2 + (11/6)*v3;
    q1 = -(1/6)*v2 + (5/6)*v3 + (1/3)*v4;
    q2 = (1/3)*v3 + (5/6)*v4 - (1/6)*v5;
    IS0 = (13/12)*(v1-2*v2+v3).^2 + (1/4)*(v1-4*v2+3*v3).^2;
    IS1 = (13/12)*(v2-2*v3+v4).^2 + (1/4)*(v2-v4).^2;
    IS2 = (13/12)*(v3-2*v4+v5).^2 + (1/4)*(3*v3-4*v4+v5).^2;
    alpha0 = 0.1./(eps+IS0).^2; alpha1 = 0.6./(eps+IS1).^2; alpha2 = 0.3./(eps+IS2).^2;
    sum_alpha = alpha0 + alpha1 + alpha2;
    F = (alpha0.*q0 + alpha1.*q1 + alpha2.*q2)./sum_alpha;
end

function u_ex = exact_solution(x_arr,t,nu,roots,weights)
    if t==0
        u_ex = -sin(pi*x_arr);
        return;
    end
    nx = length(x_arr);
    u_ex = zeros(nx,1);
    for i=1:nx
        xi = x_arr(i);
        g = sqrt(4*nu*t)*roots;
        y = xi - g;
        f_y = exp(-cos(pi*y)/(2*pi*nu));
        u_ex(i) = sum(weights.*(-sin(pi*y)).*f_y)/sum(weights.*f_y);
    end
end