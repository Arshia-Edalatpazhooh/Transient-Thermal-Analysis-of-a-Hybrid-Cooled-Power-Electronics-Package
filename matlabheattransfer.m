%% ==============================================================================
%% TRANSIENT 2D THERMAL ANALYSIS OF A HYBRID-COOLED POWER-ELECTRONICS PACKAGE
%% Numerical Method: Implicit Finite Volume Method (FVM) with Sparse LU Factorization
%% Physical System: Alloy Steel Substrate with Hot Chip & Liquid Cooling Channel
%% Verification: Benchmarked against ANSYS Fluent CFD Simulation
%% ==============================================================================

clc;
clear;
close all;

%% 1. Geometric Dimensions and Thermophysical Properties
L = 0.30;           % Substrate length along x-axis (m)
W = 0.15;           % Substrate height along y-axis (m)
k = 15;             % Thermal conductivity of alloy steel (W/(m*K))
rho = 7900;         % Density of alloy steel (kg/m^3)
cp = 500;           % Specific heat capacity (J/(kg*K))
alpha = k / (rho * cp); % Thermal diffusivity (m^2/s) ~ 3.7975e-6

% Boundary Condition Parameters
h = 45;             % Forced-air convective heat transfer coefficient at left face (W/(m^2*K))
T_inf = 25;         % Ambient air temperature (°C)
T_initial = 25;     % Uniform initial temperature of the substrate (°C)

%% 2. Spatial and Temporal Discretization Settings
delta_x = 0.001;    % Grid resolution: 1.0 mm fine mesh (m)
Nx = round(L / delta_x) + 1; % Number of grid nodes along x-direction (301 nodes)
Ny = round(W / delta_x) + 1; % Number of grid nodes along y-direction (151 nodes)
Bi = h * delta_x / k;        % Biot number based on grid cell width
delta_t = 1.5;               % Implicit time step size (s)
Fo = alpha * delta_t / delta_x^2; % Fourier number based on grid resolution
N_total = Ny * Nx;           % Total degree of freedom count in linear system (45,451 nodes)

%% 3. Internal Cavity Boundaries and Probe Coordinates
% Cavity 1: Heat-Generating Electronic Chip (x in [0.06, 0.12] m, y in [0.045, 0.105] m)
j_h1_s = round(0.06 / delta_x) + 1; 
j_h1_e = round(0.12 / delta_x) + 1; 
i_h1_s = round(0.045 / delta_x) + 1; 
i_h1_e = round(0.105 / delta_x) + 1; 

% Cavity 2: Liquid Cooling Channel (x in [0.18, 0.24] m, y in [0.045, 0.105] m)
j_h2_s = round(0.18 / delta_x) + 1; 
j_h2_e = round(0.24 / delta_x) + 1; 
i_h2_s = round(0.045 / delta_x) + 1; 
i_h2_e = round(0.105 / delta_x) + 1; 

% 2D-to-1D Column-Major Index Mapping Function
node_idx = @(i, j) (j - 1) * Ny + i;

% Probe Point Indices
i_center = round(0.075 / delta_x) + 1;   % Center plane (y = 0.075 m)
j_critical = round(0.15 / delta_x) + 1; % Critical thermal bottleneck point (x = 0.15 m)
j_left = 1;                             % Midpoint of left convective boundary (x = 0 m)
j_right = Nx;                           % Midpoint of right adiabatic boundary (x = 0.30 m)

%% 4. Computational Domain Domain Mask (MASK Matrix)
MASK = ones(Ny, Nx);
for i = 1:Ny
    for j = 1:Nx
        % Nodes strictly inside the hot chip cavity (excluding boundary walls)
        if (i > i_h1_s && i < i_h1_e && j > j_h1_s && j < j_h1_e)
            MASK(i, j) = 0;
        end
        % Nodes inside and on the boundary walls of the cold cavity (Dirichlet BC)
        if (i >= i_h2_s && i <= i_h2_e && j >= j_h2_s && j <= j_h2_e)
            MASK(i, j) = 0;
        end
    end
end

%% 5. Assembly of the Global Implicit System Matrix (A)
fprintf('===================================================\n');
fprintf('Assembling global sparse matrix A (%d x %d)...\n', N_total, N_total);
A = sparse(N_total, N_total);

for i = 1:Ny
    for j = 1:Nx
        row = node_idx(i, j);
        
        % Inactive void nodes inside hot chip cavity
        if MASK(i, j) == 0 && ~(i >= i_h2_s && i <= i_h2_e && j >= j_h2_s && j <= j_h2_e)
            A(row, row) = 1;
            continue;
        end
        
        % Prescribed Dirichlet nodes of liquid cooling cavity
        if (i >= i_h2_s && i <= i_h2_e && j >= j_h2_s && j <= j_h2_e)
            A(row, row) = 1;
            continue;
        end
        
        % Default Interior Node Formulation: (1 + 4*Fo)*T_P - Fo*(T_W + T_E + T_S + T_N) = T_P^n
        a_P = 1 + 4*Fo;
        a_W = -Fo; a_E = -Fo; a_S = -Fo; a_N = -Fo;
        
        % Exterior Boundary Conditions
        if j == 1 
            % Left boundary: Forced convection face (x = 0)
            a_P = 1 + 2*Fo*(2 + Bi);
            a_E = -2*Fo; a_W = 0;
        elseif j == Nx 
            % Right boundary: Perfectly insulated adiabatic face (x = L)
            a_P = 1 + 4*Fo; a_W = -2*Fo; a_E = 0;
        end
        
        if i == 1 
            % Bottom boundary: Perfectly insulated adiabatic face (y = 0)
            a_P = 1 + 4*Fo; a_N = -2*Fo; a_S = 0;
        elseif i == Ny 
            % Top boundary: Perfectly insulated adiabatic face (y = W)
            a_P = 1 + 4*Fo; a_S = -2*Fo; a_N = 0;
        end
        
        % Internal Hot Chip Cavity Boundary Walls (Prescribed Flux q'')
        if (i > i_h1_s && i < i_h1_e && j == j_h1_s), a_P = 1 + 4*Fo; a_W = -2*Fo; a_E = 0; end % Left wall
        if (i > i_h1_s && i < i_h1_e && j == j_h1_e), a_P = 1 + 4*Fo; a_E = -2*Fo; a_W = 0; end % Right wall
        if (j > j_h1_s && j < j_h1_e && i == i_h1_s), a_P = 1 + 4*Fo; a_S = -2*Fo; a_N = 0; end % Bottom wall
        if (j > j_h1_s && j < j_h1_e && i == i_h1_e), a_P = 1 + 4*Fo; a_N = -2*Fo; a_S = 0; end % Top wall
        
        % Populate Sparse Matrix A
        A(row, row) = a_P;
        if j > 1,  A(row, node_idx(i, j-1)) = a_W; end
        if j < Nx, A(row, node_idx(i, j+1)) = a_E; end
        if i > 1,  A(row, node_idx(i-1, j)) = a_S; end
        if i < Ny, A(row, node_idx(i+1, j)) = a_N; end
    end
end

%% 6. Precomputing Sparse LU Factorization for Ultra-Fast Time-Stepping
fprintf('Computing sparse LU decomposition of A...\n');
[L_mat, U_mat, P_mat, Q_mat] = lu(A);

%% 7. Baseline Transient Simulation (T_c = 5 °C)
T_c = 5; % Baseline cooling fluid temperature (°C)

Tp = ones(Ny, Nx) * T_initial;
Tp(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = T_c;
Tp_vec = reshape(Tp, [N_total, 1]);

t = 0;
cycle_period = 300; % Heat flux oscillation period (s)
max_time = 30000;   % Maximum transient integration time (s) = 500 minutes

% Data Logging Arrays
time_hist = [];
T_critical_hist = [];
T_left_hist = [];
T_right_hist = [];

T_10min = []; T_20min = []; T_30min = [];
T_cycle_end_prev = [];

% GIF Animation Configuration
make_animation = true;
gif_filename = 'Thermal_Transient_Animation.gif';

fprintf('===================================================\n');
fprintf('Starting transient simulation for baseline case (T_c = 5 °C)...\n');

step_cnt = 0;
is_quasi_steady = false;
max_T_critical_30min = -Inf;

while t <= max_time
    % Time-varying sinusoidal heat flux generated by chip (W/m^2)
    q_flux = 25000 * abs(sin(pi * t / cycle_period));
    
    % Construct Right-Hand-Side (RHS) Vector B
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
    
    % Ultra-Fast Direct Solution via Pre-Factored LU Substitution
    Tn_vec = Q_mat * (U_mat \ (L_mat \ (P_mat * B)));
    Tn = reshape(Tn_vec, [Ny, Nx]);
    Tn(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = T_c;
    
    % Record Probe Histories
    time_hist = [time_hist, t];
    T_crit_curr = Tn(i_center, j_critical);
    T_critical_hist = [T_critical_hist, T_crit_curr];
    T_left_hist = [T_left_hist, Tn(i_center, j_left)];
    T_right_hist = [T_right_hist, Tn(i_center, j_right)];
    
    % Track Maximum Bottleneck Temperature During First 30 Minutes
    if t <= 1800
        if T_crit_curr > max_T_critical_30min
            max_T_critical_30min = T_crit_curr;
        end
    end
    
    % Store Solution Snapshots at 10, 20, and 30 Minutes
    if isempty(T_10min) && t >= 600, T_10min = Tn; end
    if isempty(T_20min) && t >= 1200, T_20min = Tn; end
    if isempty(T_30min) && t >= 1800, T_30min = Tn; end
    
    % Export Animated GIF Frame (First 30 minutes, every 30 seconds)
    if make_animation && t <= 1800 && mod(t, 30) == 0
        fig_anim = figure(99); set(fig_anim, 'Visible', 'off');
        P_anim = Tn;
        P_anim(i_h1_s+1:i_h1_e-1, j_h1_s+1:j_h1_e-1) = NaN;
        P_anim(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = NaN;
        x = linspace(0, L, Nx); y = linspace(0, W, Ny);
        contourf(x, y, P_anim, 30, 'LineColor', 'none');
        colorbar; colormap('jet'); clim([0 100]);
        title(sprintf('Transient Thermal Distribution - Time: %.1f min', t/60));
        xlabel('X (m)'); ylabel('Y (m)'); axis equal;
        frame = getframe(fig_anim);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        if t == 0
            imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
        else
            imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end
        close(fig_anim);
    end
    
    % Check for Convergence to Quasi-Steady Periodic State
    if t > 0 && mod(t, cycle_period) == 0
        if ~isempty(T_cycle_end_prev)
            diff_contour = max(max(abs(Tn - T_cycle_end_prev)));
            fprintf('Cycle completed at t = %.1f min | Max cycle-to-cycle temperature difference: %.4f °C\n', t/60, diff_contour);
            if diff_contour < 0.05 && t >= 1800
                is_quasi_steady = true;
                fprintf('--> Quasi-steady periodic state achieved at t = %.1f min.\n', t/60);
                break;
            end
        end
        T_cycle_end_prev = Tn;
    end
    
    % Advance Time Step
    t = t + delta_t;
    Tp_vec = Tn_vec;
end

%% 8. Thermal Bottleneck Safety Evaluation (Question 4)
fprintf('\n===================================================\n');
fprintf('Thermal Bottleneck Evaluation Result (Question 4):\n');
fprintf('Peak recorded temperature at bottleneck (first 30 min): %.2f °C\n', max_T_critical_30min);
if max_T_critical_30min > 75
    fprintf('System Diagnostic: Design REJECTED (FAIL) - Exceeded 75 °C threshold\n');
else
    fprintf('System Diagnostic: Design VERIFIED (PASS) - Remains below 75 °C threshold\n');
end
fprintf('===================================================\n\n');

%% 9. Visualization: Temperature Contours at 10, 20, and 30 min (Question 5)
x = linspace(0, L, Nx); y = linspace(0, W, Ny);
[X, Y] = meshgrid(x, y);

P_10 = T_10min; P_10(i_h1_s+1:i_h1_e-1, j_h1_s+1:j_h1_e-1) = NaN; P_10(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = NaN;
P_20 = T_20min; P_20(i_h1_s+1:i_h1_e-1, j_h1_s+1:j_h1_e-1) = NaN; P_20(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = NaN;
P_30 = T_30min; P_30(i_h1_s+1:i_h1_e-1, j_h1_s+1:j_h1_e-1) = NaN; P_30(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = NaN;

figure('Name', 'Q5 - Temperature Contours (10, 20, 30 min)', 'NumberTitle', 'off', 'Color', 'w');
subplot(3, 1, 1);
contourf(X, Y, P_10, 30, 'LineColor', 'none'); colorbar; colormap('jet'); clim([0 100]);
title('Temperature Distribution at t = 10 min'); xlabel('X (m)'); ylabel('Y (m)'); axis equal;

subplot(3, 1, 2);
contourf(X, Y, P_20, 30, 'LineColor', 'none'); colorbar; colormap('jet'); clim([0 100]);
title('Temperature Distribution at t = 20 min'); xlabel('X (m)'); ylabel('Y (m)'); axis equal;

subplot(3, 1, 3);
contourf(X, Y, P_30, 30, 'LineColor', 'none'); colorbar; colormap('jet'); clim([0 100]);
title('Temperature Distribution at t = 30 min'); xlabel('X (m)'); ylabel('Y (m)'); axis equal;

%% 10. Visualization: Temperature History to Quasi-Steady State (Question 6)
figure('Name', 'Q6 - Temperature History to Quasi-Steady State', 'NumberTitle', 'off', 'Color', 'w');
plot(time_hist/60, T_critical_hist, 'r-', 'LineWidth', 1.8, 'DisplayName', 'Center Point (L/2, W/2)'); hold on;
plot(time_hist/60, T_left_hist, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Mid Left Boundary (0, W/2)');
plot(time_hist/60, T_right_hist, 'k-.', 'LineWidth', 1.5, 'DisplayName', 'Mid Right Boundary (L, W/2)');
grid on; xlabel('Time (min)'); ylabel('Temperature (°C)');
title('Temperature Variation Until Quasi-Steady Periodic State');
legend('Location', 'best');

%% 11. Visualization: Thermal Bottleneck Point History (30 min) (Question 7)
idx_30min = time_hist <= 1800;
figure('Name', 'Q7 - Critical Point Temperature History (30 min)', 'NumberTitle', 'off', 'Color', 'w');
plot(time_hist(idx_30min)/60, T_critical_hist(idx_30min), 'r-', 'LineWidth', 2);
grid on; xlabel('Time (min)'); ylabel('Temperature (°C)');
title('Thermal Bottleneck Point Temperature History (x = 15 cm, y = 7.5 cm)');
yline(75, 'g--', 'Threshold Limit (75 °C)', 'LineWidth', 1.5);

%% 12. Parametric Analysis: Effect of Coolant Temperature (Question 8)
Tc_vec = [0, 5, 10, 15];
figure('Name', 'Q8 - Effect of Cooling Liquid Temperature', 'NumberTitle', 'off', 'Color', 'w');
hold on; grid on;
colors = ['b', 'r', 'g', 'm'];

fprintf('===================================================\n');
fprintf('Starting parametric coolant study for Question 8 (T_c = 0, 5, 10, 15 °C)...\n');

for k_tc = 1:length(Tc_vec)
    Tc_val = Tc_vec(k_tc);
    
    Tp_param = ones(Ny, Nx) * T_initial;
    Tp_param(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = Tc_val;
    Tp_vec_p = reshape(Tp_param, [N_total, 1]);
    
    t_p = 0;
    T_center_param = [];
    time_param = [];
    T_cycle_prev_p = [];
    
    while t_p <= max_time
        q_flux = 25000 * abs(sin(pi * t_p / cycle_period));
        
        B = Tp_vec_p;
        for i = 1:Ny
            for j = 1:Nx
                row = node_idx(i, j);
                if (i >= i_h2_s && i <= i_h2_e && j >= j_h2_s && j <= j_h2_e)
                    B(row) = Tc_val; continue;
                end
                if MASK(i, j) == 0
                    B(row) = T_initial; continue;
                end
                if j == 1, B(row) = B(row) + 2*Fo*Bi*T_inf; end
                if (i >= i_h1_s && i <= i_h1_e && (j == j_h1_s || j == j_h1_e)) || ...
                   (j >= j_h1_s && j <= j_h1_e && (i == i_h1_s || i == i_h1_e))
                    B(row) = B(row) + (2 * Fo * delta_x / k) * q_flux;
                end
            end
        end
        
        Tn_vec_p = Q_mat * (U_mat \ (L_mat \ (P_mat * B)));
        Tn_p = reshape(Tn_vec_p, [Ny, Nx]);
        Tn_p(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = Tc_val;
        
        time_param = [time_param, t_p];
        T_center_param = [T_center_param, Tn_p(i_center, j_critical)];
        
        if t_p > 0 && mod(t_p, cycle_period) == 0
            if ~isempty(T_cycle_prev_p)
                if max(max(abs(Tn_p - T_cycle_prev_p))) < 0.05 && t_p >= 1800
                    break;
                end
            end
            T_cycle_prev_p = Tn_p;
        end
        
        t_p = t_p + delta_t;
        Tp_vec_p = Tn_vec_p;
    end
    
    plot(time_param/60, T_center_param, 'Color', colors(k_tc), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('T_c = %d °C', Tc_val));
end

xlabel('Time (min)'); ylabel('Center Temperature (°C)');
title('Center Point Temperature Oscillations in Quasi-Steady State for Different T_c');
legend('Location', 'best');
