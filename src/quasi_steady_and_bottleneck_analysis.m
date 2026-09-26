%% Quasi-Steady Periodic State and Thermal Bottleneck Analysis
% Numerical Method: Transient 2D Finite Volume Method
% Physical Domain: Power Electronics Module with Hybrid Cooling
%
% Description: Runs the simulation until the system achieves a quasi-steady periodic
% state, defined by temperature difference between consecutive oscillation cycles
% falling below 0.05 °C. Monitors three strategic probe points:
%   1. Substrate Center / Thermal Bottleneck Point (x = L/2, y = W/2) = (0.15 m, 0.075 m)
%   2. Midpoint of Left Convective Boundary (x = 0, y = W/2) = (0 m, 0.075 m)
%   3. Midpoint of Right Adiabatic Boundary (x = L, y = W/2) = (0.30 m, 0.075 m)
% Verifies the thermal bottleneck operating temperature against the 75 °C safety threshold.

clc;
clear;
close all;

%% 1. Parameters
L = 0.30;
W = 0.15;
k = 15.0;
rho = 7900.0;
cp = 500.0;
alpha = k / (rho * cp);

h = 45.0;
T_inf = 25.0;
T_c = 5.0;
T_init = 25.0;

delta_x = 0.005;
Nx = round(L / delta_x) + 1;
Ny = round(W / delta_x) + 1;

Bi = h * delta_x / k;
dt_max = delta_x^2 / (2.0 * alpha * (2.0 + Bi));
delta_t = 1.0; % s
Fo = alpha * delta_t / (delta_x^2);

period = 300.0; % Cycle period (s)

% Cavity Coordinates
j_h1_s = round(0.06 / delta_x) + 1;
j_h1_e = round(0.12 / delta_x) + 1;
i_h1_s = round(0.045 / delta_x) + 1;
i_h1_e = round(0.105 / delta_x) + 1;

j_h2_s = round(0.18 / delta_x) + 1;
j_h2_e = round(0.24 / delta_x) + 1;
i_h2_s = round(0.045 / delta_x) + 1;
i_h2_e = round(0.105 / delta_x) + 1;

% Probe indices
i_mid = round(0.075 / delta_x) + 1;
j_center = round(0.15 / delta_x) + 1;
j_left = 1;
j_right = Nx;

%% 2. Initialization
Tp = ones(Ny, Nx) * T_init;
Tn = Tp;
Tp(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = T_c;
Tn(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = T_c;

time_history = [];
T_center_hist = [];
T_left_hist   = [];
T_right_hist  = [];

t = 0.0;
max_cycles = 60; % Up to 300 minutes (18,000 s)
cycle_count = 0;
prev_cycle_T = Tp;
tolerance = 0.05; % °C convergence criterion

fprintf('Starting Quasi-Steady Periodic Analysis (Threshold < %.2f °C)...\n', tolerance);

while cycle_count < max_cycles
    t = t + delta_t;
    q_flux = 25000.0 * abs(sin(pi * t / period));
    
    % Solid internal nodes
    for i = 2:Ny-1
        for j = 2:Nx-1
            if (i > i_h1_s && i < i_h1_e && j > j_h1_s && j < j_h1_e) || ...
               (i >= i_h2_s && i <= i_h2_e && j >= j_h2_s && j <= j_h2_e) || ...
               (i >= i_h1_s && i <= i_h1_e && j >= j_h1_s && j <= j_h1_e)
                continue;
            end
            Tn(i, j) = (1 - 4*Fo)*Tp(i, j) + Fo*(Tp(i+1, j) + Tp(i-1, j) + Tp(i, j+1) + Tp(i, j-1));
        end
    end
    
    % Left convection
    for i = 2:Ny-1
        Tn(i, 1) = (1 - 4*Fo - 2*Bi*Fo)*Tp(i, 1) + Fo*(2*Tp(i, 2) + Tp(i+1, 1) + Tp(i-1, 1) + 2*Bi*T_inf);
    end
    
    % Adiabatic boundaries
    for i = 2:Ny-1, Tn(i, Nx) = (1 - 4*Fo)*Tp(i, Nx) + Fo*(2*Tp(i, Nx-1) + Tp(i+1, Nx) + Tp(i-1, Nx)); end
    for j = 2:Nx-1, Tn(1, j)  = (1 - 4*Fo)*Tp(1, j)  + Fo*(2*Tp(2, j) + Tp(1, j+1) + Tp(1, j-1)); end
    for j = 2:Nx-1, Tn(Ny, j) = (1 - 4*Fo)*Tp(Ny, j) + Fo*(2*Tp(Ny-1, j) + Tp(Ny, j+1) + Tp(Ny, j-1)); end
    
    % Corners
    Tn(1, 1)   = (1 - 4*Fo - 2*Bi*Fo)*Tp(1, 1) + 2*Fo*(Tp(1, 2) + Tp(2, 1) + Bi*T_inf);
    Tn(Ny, 1)  = (1 - 4*Fo - 2*Bi*Fo)*Tp(Ny, 1) + 2*Fo*(Tp(Ny, 2) + Tp(Ny-1, 1) + Bi*T_inf);
    Tn(1, Nx)  = (1 - 4*Fo)*Tp(1, Nx) + 2*Fo*(Tp(1, Nx-1) + Tp(2, Nx));
    Tn(Ny, Nx) = (1 - 4*Fo)*Tp(Ny, Nx) + 2*Fo*(Tp(Ny, Nx-1) + Tp(Ny-1, Nx));
    
    % Cavity 1 walls
    for i = i_h1_s+1:i_h1_e-1
        Tn(i, j_h1_s) = (1 - 4*Fo)*Tp(i, j_h1_s) + Fo*(2*Tp(i, j_h1_s-1) + Tp(i+1, j_h1_s) + Tp(i-1, j_h1_s)) + (2*Fo*delta_x/k)*q_flux;
        Tn(i, j_h1_e) = (1 - 4*Fo)*Tp(i, j_h1_e) + Fo*(2*Tp(i, j_h1_e+1) + Tp(i+1, j_h1_e) + Tp(i-1, j_h1_e)) + (2*Fo*delta_x/k)*q_flux;
    end
    for j = j_h1_s+1:j_h1_e-1
        Tn(i_h1_s, j) = (1 - 4*Fo)*Tp(i_h1_s, j) + Fo*(2*Tp(i_h1_s-1, j) + Tp(i_h1_s, j+1) + Tp(i_h1_s, j-1)) + (2*Fo*delta_x/k)*q_flux;
        Tn(i_h1_e, j) = (1 - 4*Fo)*Tp(i_h1_e, j) + Fo*(2*Tp(i_h1_e+1, j) + Tp(i_h1_e, j+1) + Tp(i_h1_e, j-1)) + (2*Fo*delta_x/k)*q_flux;
    end
    
    % Cavity 1 corners
    Tn(i_h1_s, j_h1_s) = (1 - 4*Fo)*Tp(i_h1_s, j_h1_s) + 2*Fo*(Tp(i_h1_s-1, j_h1_s) + Tp(i_h1_s, j_h1_s-1)) + (4*Fo*delta_x/k)*q_flux;
    Tn(i_h1_e, j_h1_s) = (1 - 4*Fo)*Tp(i_h1_e, j_h1_s) + 2*Fo*(Tp(i_h1_e+1, j_h1_s) + Tp(i_h1_e, j_h1_s-1)) + (4*Fo*delta_x/k)*q_flux;
    Tn(i_h1_s, j_h1_e) = (1 - 4*Fo)*Tp(i_h1_s, j_h1_e) + 2*Fo*(Tp(i_h1_s-1, j_h1_e) + Tp(i_h1_s, j_h1_e+1)) + (4*Fo*delta_x/k)*q_flux;
    Tn(i_h1_e, j_h1_e) = (1 - 4*Fo)*Tp(i_h1_e, j_h1_e) + 2*Fo*(Tp(i_h1_e+1, j_h1_e) + Tp(i_h1_e, j_h1_e+1)) + (4*Fo*delta_x/k)*q_flux;

    % Cooling Cavity
    Tn(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = T_c;
    
    Tp = Tn;
    
    % Record Probe Histories
    time_history(end+1)  = t / 60.0; % in minutes
    T_center_hist(end+1) = Tp(i_mid, j_center);
    T_left_hist(end+1)   = Tp(i_mid, j_left);
    T_right_hist(end+1)  = Tp(i_mid, j_right);
    
    % Check end of periodic cycle
    if mod(t, period) == 0
        cycle_count = cycle_count + 1;
        max_diff = max(abs(Tp(:) - prev_cycle_T(:)));
        fprintf('Cycle %2d (%6.0f s | %4.1f min): Max Difference between cycles = %.4f °C\n', ...
            cycle_count, t, t/60, max_diff);
        if max_diff < tolerance
            fprintf('--> Quasi-steady periodic state achieved at t = %.1f min (Cycle %d)!\n', t/60, cycle_count);
            break;
        end
        prev_cycle_T = Tp;
    end
end

%% 3. Thermal Bottleneck Verification Summary
T_bottleneck_max = max(T_center_hist(end-round(period/delta_t):end));
failure_limit = 75.0;

fprintf('\n=======================================================\n');
fprintf('QUASI-STEADY STATE THERMAL BOTTLENECK VERIFICATION\n');
fprintf('Peak Bottleneck Temperature in Quasi-Steady Cycle: %.2f °C\n', T_bottleneck_max);
fprintf('Allowable Threshold: %.1f °C\n', failure_limit);
if T_bottleneck_max < failure_limit
    fprintf('STATUS: PASS (Safety Margin: +%.2f °C)\n', failure_limit - T_bottleneck_max);
else
    fprintf('STATUS: FAIL (Exceeded Threshold)\n');
end
fprintf('=======================================================\n');

%% 4. Visualization: Temperature Histories
figure('Name', 'Temperature Variation until Quasi-Steady Periodic State', 'Color', 'w');
plot(time_history, T_center_hist, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Center Point (L/2, W/2)');
hold on;
plot(time_history, T_left_hist, 'b--', 'LineWidth', 1.2, 'DisplayName', 'Mid Left Boundary (0, W/2)');
plot(time_history, T_right_hist, 'k-.', 'LineWidth', 1.2, 'DisplayName', 'Mid Right Boundary (L, W/2)');
grid on;
xlabel('Time (min)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Temperature (°C)', 'FontSize', 11, 'FontWeight', 'bold');
title('Temperature Variation Until Quasi-Steady Periodic State', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'southeast');

figure('Name', 'Bottleneck Point History vs Failure Limit', 'Color', 'w');
plot(time_history, T_center_hist, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Bottleneck Point (0.15 m, 0.075 m)');
hold on;
yline(failure_limit, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Failure Limit (75 °C)');
grid on;
ylim([20, 80]);
xlabel('Time (min)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Temperature (°C)', 'FontSize', 11, 'FontWeight', 'bold');
title('Temperature History at Thermal Bottleneck Point (x = 0.15 m, y = 0.075 m)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast');
