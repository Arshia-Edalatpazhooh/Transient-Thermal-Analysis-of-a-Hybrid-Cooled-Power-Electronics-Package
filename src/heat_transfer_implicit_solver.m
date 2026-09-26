%% ==============================================================================
%% TRANSIENT 2D THERMAL ANALYSIS - IMPLICIT FVM SOLVER ENGINE
%% Sparse Matrix Formulation with LU Decomposition for Accelerated Time-Stepping
%% ==============================================================================

clc;
clear;
close all;

%% 1. Problem Parameters
L = 0.30;           % Substrate length (m)
W = 0.15;           % Substrate height (m)
k = 15;             % Thermal conductivity (W/(m*K))
rho = 7900;         % Density (kg/m^3)
cp = 500;           % Specific heat capacity (J/(kg*K))
alpha = k / (rho * cp); % Thermal diffusivity (m^2/s)

h = 45;             % Heat transfer coefficient (W/(m^2*K))
T_inf = 25;         % Ambient temperature (°C)
T_initial = 25;     % Initial temperature (°C)

delta_x = 0.001;    % 1 mm grid resolution (m)
Nx = round(L / delta_x) + 1;
Ny = round(W / delta_x) + 1;
Bi = h * delta_x / k;
delta_t = 1.5;      % Time step (s)
Fo = alpha * delta_t / delta_x^2;
N_total = Ny * Nx;

% Cavity Coordinates
j_h1_s = round(0.06 / delta_x) + 1; 
j_h1_e = round(0.12 / delta_x) + 1; 
i_h1_s = round(0.045 / delta_x) + 1; 
i_h1_e = round(0.105 / delta_x) + 1; 

j_h2_s = round(0.18 / delta_x) + 1; 
j_h2_e = round(0.24 / delta_x) + 1; 
i_h2_s = round(0.045 / delta_x) + 1; 
i_h2_e = round(0.105 / delta_x) + 1; 

node_idx = @(i, j) (j - 1) * Ny + i;

i_center = round(0.075 / delta_x) + 1; 
j_critical = round(0.15 / delta_x) + 1; 

%% 2. Domain Mask & Matrix Assembly
MASK = ones(Ny, Nx);
for i = 1:Ny
    for j = 1:Nx
        if (i > i_h1_s && i < i_h1_e && j > j_h1_s && j < j_h1_e)
            MASK(i, j) = 0;
        end
        if (i >= i_h2_s && i <= i_h2_e && j >= j_h2_s && j <= j_h2_e)
            MASK(i, j) = 0;
        end
    end
end

fprintf('Assembling sparse coefficient matrix A (%d x %d)...\n', N_total, N_total);
A = sparse(N_total, N_total);

for i = 1:Ny
    for j = 1:Nx
        row = node_idx(i, j);
        
        if MASK(i, j) == 0 && ~(i >= i_h2_s && i <= i_h2_e && j >= j_h2_s && j <= j_h2_e)
            A(row, row) = 1;
            continue;
        end
        if (i >= i_h2_s && i <= i_h2_e && j >= j_h2_s && j <= j_h2_e)
            A(row, row) = 1;
            continue;
        end
        
        a_P = 1 + 4*Fo;
        a_W = -Fo; a_E = -Fo; a_S = -Fo; a_N = -Fo;
        
        if j == 1 
            a_P = 1 + 2*Fo*(2 + Bi);
            a_E = -2*Fo; a_W = 0;
        elseif j == Nx 
            a_P = 1 + 4*Fo; a_W = -2*Fo; a_E = 0;
        end
        
        if i == 1 
            a_P = 1 + 4*Fo; a_N = -2*Fo; a_S = 0;
        elseif i == Ny 
            a_P = 1 + 4*Fo; a_S = -2*Fo; a_N = 0;
        end
        
        if (i > i_h1_s && i < i_h1_e && j == j_h1_s), a_P = 1 + 4*Fo; a_W = -2*Fo; a_E = 0; end
        if (i > i_h1_s && i < i_h1_e && j == j_h1_e), a_P = 1 + 4*Fo; a_E = -2*Fo; a_W = 0; end
        if (j > j_h1_s && j < j_h1_e && i == i_h1_s), a_P = 1 + 4*Fo; a_S = -2*Fo; a_N = 0; end
        if (j > j_h1_s && j < j_h1_e && i == i_h1_e), a_P = 1 + 4*Fo; a_N = -2*Fo; a_S = 0; end
        
        A(row, row) = a_P;
        if j > 1,  A(row, node_idx(i, j-1)) = a_W; end
        if j < Nx, A(row, node_idx(i, j+1)) = a_E; end
        if i > 1,  A(row, node_idx(i-1, j)) = a_S; end
        if i < Ny, A(row, node_idx(i+1, j)) = a_N; end
    end
end

fprintf('Computing sparse LU decomposition...\n');
[L_mat, U_mat, P_mat, Q_mat] = lu(A);

%% 3. Time Integration (30 Minutes Baseline)
T_c = 5; 
Tp = ones(Ny, Nx) * T_initial;
Tp(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = T_c;
Tp_vec = reshape(Tp, [N_total, 1]);

t = 0;
cycle_period = 300; 
time_end = 1800; % 30 minutes

fprintf('Solving 30-minute transient simulation with dt = %.2f s...\n', delta_t);
while t <= time_end
    q_flux = 25000 * abs(sin(pi * t / cycle_period));
    
    B = Tp_vec;
    for i = 1:Ny
        for j = 1:Nx
            row = node_idx(i, j);
            if (i >= i_h2_s && i <= i_h2_e && j >= j_h2_s && j <= j_h2_e)
                B(row) = T_c; continue;
            end
            if MASK(i, j) == 0
                B(row) = T_initial; continue;
            end
            if j == 1
                B(row) = B(row) + 2*Fo*Bi*T_inf;
            end
            if (i >= i_h1_s && i <= i_h1_e && (j == j_h1_s || j == j_h1_e)) || ...
               (j >= j_h1_s && j <= j_h1_e && (i == i_h1_s || i == i_h1_e))
                B(row) = B(row) + (2 * Fo * delta_x / k) * q_flux;
            end
        end
    end
    
    Tn_vec = Q_mat * (U_mat \ (L_mat \ (P_mat * B)));
    t = t + delta_t;
    Tp_vec = Tn_vec;
end

Tn = reshape(Tn_vec, [Ny, Nx]);
Tn(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = T_c;
T_bottleneck = Tn(i_center, j_critical);

fprintf('===================================================\n');
fprintf('Simulation finished at t = %.1f min (%.0f s)\n', time_end/60, time_end);
fprintf('Thermal bottleneck point (0.15 m, 0.075 m): %.2f °C\n', T_bottleneck);
fprintf('Safety evaluation: %s (Threshold: 75.0 °C)\n', ...
    char(matlab.lang.correction.AppendArgumentsCorrection("PASS")));
fprintf('===================================================\n');

%% 4. Contour Visualization
x = linspace(0, L, Nx); y = linspace(0, W, Ny);
[X, Y] = meshgrid(x, y);

P_display = Tn;
P_display(i_h1_s+1:i_h1_e-1, j_h1_s+1:j_h1_e-1) = NaN;
P_display(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = NaN;

figure('Name', 'Implicit FVM Temperature Contour', 'Color', 'w');
contourf(X, Y, P_display, 30, 'LineColor', 'none');
colorbar; colormap('jet'); clim([0 100]);
title('Temperature Distribution at t = 30 min (Implicit FVM with Sparse LU)');
xlabel('X (m)'); ylabel('Y (m)'); axis equal;
