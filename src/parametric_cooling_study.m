%% Effect of Cooling Fluid Temperature (T_c) on Thermal Oscillations
% Numerical Method: Transient 2D Finite Volume Method (Explicit)
% Physical Domain: Power Electronics Module with Hybrid Cooling
%
% Description: Evaluates the sensitivity of the quasi-steady periodic temperature
% response at the substrate center point to variations in the cooling fluid
% temperature: T_c = 0 °C, 5 °C, 10 °C, and 15 °C.
% Plots the single-cycle thermal oscillations once periodicity is established.

clc;
clear;
close all;

%% 1. Baseline System Parameters
L = 0.30;
W = 0.15;
k = 15.0;
rho = 7900.0;
cp = 500.0;
alpha = k / (rho * cp);

h = 45.0;
T_inf = 25.0;
T_initial = 25.0;

delta_x = 0.005;
Nx = round(L / delta_x) + 1;
Ny = round(W / delta_x) + 1;

Bi = h * delta_x / k;
delta_t = 1.0; % s
Fo = alpha * delta_t / (delta_x^2);

period = 300.0; % Oscillation period (s)
time_to_steady = 50 * period; % 250 minutes to guarantee quasi-steady regime

% Cavities
j_h1_s = round(0.06 / delta_x) + 1;
j_h1_e = round(0.12 / delta_x) + 1;
i_h1_s = round(0.045 / delta_x) + 1;
i_h1_e = round(0.105 / delta_x) + 1;

j_h2_s = round(0.18 / delta_x) + 1;
j_h2_e = round(0.24 / delta_x) + 1;
i_h2_s = round(0.045 / delta_x) + 1;
i_h2_e = round(0.105 / delta_x) + 1;

i_mid = round(0.075 / delta_x) + 1;
j_mid = round(0.15 / delta_x) + 1;

Tc_values = [0, 5, 10, 15];
colors = {'b-', 'g--', 'r-', 'm:'};

figure('Name', 'Cooling Fluid Temperature Sensitivity', 'Color', 'w');
hold on;

%% 2. Loop over Cooling Fluid Temperatures
for idx = 1:length(Tc_values)
    T_c = Tc_values(idx);
    fprintf('Running simulation for T_c = %d °C...\n', T_c);
    
    Tp = ones(Ny, Nx) * T_initial;
    Tn = Tp;
    Tp(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = T_c;
    Tn(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = T_c;
    
    t = 0.0;
    while t < time_to_steady
        t = t + delta_t;
        q_flux = 25000.0 * abs(sin(pi * t / period));
        
        % Interior nodes
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
        
        % Adiabatic
        for i = 2:Ny-1, Tn(i, Nx) = (1 - 4*Fo)*Tp(i, Nx) + Fo*(2*Tp(i, Nx-1) + Tp(i+1, Nx) + Tp(i-1, Nx)); end
        for j = 2:Nx-1, Tn(1, j)  = (1 - 4*Fo)*Tp(1, j)  + Fo*(2*Tp(2, j) + Tp(1, j+1) + Tp(1, j-1)); end
        for j = 2:Nx-1, Tn(Ny, j) = (1 - 4*Fo)*Tp(Ny, j) + Fo*(2*Tp(Ny-1, j) + Tp(Ny, j+1) + Tp(Ny, j-1)); end
        
        % Corners
        Tn(1, 1)   = (1 - 4*Fo - 2*Bi*Fo)*Tp(1, 1) + 2*Fo*(Tp(1, 2) + Tp(2, 1) + Bi*T_inf);
        Tn(Ny, 1)  = (1 - 4*Fo - 2*Bi*Fo)*Tp(Ny, 1) + 2*Fo*(Tp(Ny, 2) + Tp(Ny-1, 1) + Bi*T_inf);
        Tn(1, Nx)  = (1 - 4*Fo)*Tp(1, Nx) + 2*Fo*(Tp(1, Nx-1) + Tp(2, Nx));
        Tn(Ny, Nx) = (1 - 4*Fo)*Tp(Ny, Nx) + 2*Fo*(Tp(Ny, Nx-1) + Tp(Ny-1, Nx));
        
        % Hot cavity walls
        for i = i_h1_s+1:i_h1_e-1
            Tn(i, j_h1_s) = (1 - 4*Fo)*Tp(i, j_h1_s) + Fo*(2*Tp(i, j_h1_s-1) + Tp(i+1, j_h1_s) + Tp(i-1, j_h1_s)) + (2*Fo*delta_x/k)*q_flux;
            Tn(i, j_h1_e) = (1 - 4*Fo)*Tp(i, j_h1_e) + Fo*(2*Tp(i, j_h1_e+1) + Tp(i+1, j_h1_e) + Tp(i-1, j_h1_e)) + (2*Fo*delta_x/k)*q_flux;
        end
        for j = j_h1_s+1:j_h1_e-1
            Tn(i_h1_s, j) = (1 - 4*Fo)*Tp(i_h1_s, j) + Fo*(2*Tp(i_h1_s-1, j) + Tp(i_h1_s, j+1) + Tp(i_h1_s, j-1)) + (2*Fo*delta_x/k)*q_flux;
            Tn(i_h1_e, j) = (1 - 4*Fo)*Tp(i_h1_e, j) + Fo*(2*Tp(i_h1_e+1, j) + Tp(i_h1_e, j+1) + Tp(i_h1_e, j-1)) + (2*Fo*delta_x/k)*q_flux;
        end
        
        % Hot cavity corners
        Tn(i_h1_s, j_h1_s) = (1 - 4*Fo)*Tp(i_h1_s, j_h1_s) + 2*Fo*(Tp(i_h1_s-1, j_h1_s) + Tp(i_h1_s, j_h1_s-1)) + (4*Fo*delta_x/k)*q_flux;
        Tn(i_h1_e, j_h1_s) = (1 - 4*Fo)*Tp(i_h1_e, j_h1_s) + 2*Fo*(Tp(i_h1_e+1, j_h1_s) + Tp(i_h1_e, j_h1_s-1)) + (4*Fo*delta_x/k)*q_flux;
        Tn(i_h1_s, j_h1_e) = (1 - 4*Fo)*Tp(i_h1_s, j_h1_e) + 2*Fo*(Tp(i_h1_s-1, j_h1_e) + Tp(i_h1_s, j_h1_e+1)) + (4*Fo*delta_x/k)*q_flux;
        Tn(i_h1_e, j_h1_e) = (1 - 4*Fo)*Tp(i_h1_e, j_h1_e) + 2*Fo*(Tp(i_h1_e+1, j_h1_e) + Tp(i_h1_e, j_h1_e+1)) + (4*Fo*delta_x/k)*q_flux;

        % Cold cavity
        Tn(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = T_c;
        Tp = Tn;
    end
    
    % Record one steady-state cycle
    cycle_time = 0:delta_t:period;
    cycle_T = zeros(size(cycle_time));
    for c_idx = 1:length(cycle_time)
        t = t + delta_t;
        q_flux = 25000.0 * abs(sin(pi * t / period));
        
        % Update
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
        for i = 2:Ny-1
            Tn(i, 1) = (1 - 4*Fo - 2*Bi*Fo)*Tp(i, 1) + Fo*(2*Tp(i, 2) + Tp(i+1, 1) + Tp(i-1, 1) + 2*Bi*T_inf);
        end
        for i = 2:Ny-1, Tn(i, Nx) = (1 - 4*Fo)*Tp(i, Nx) + Fo*(2*Tp(i, Nx-1) + Tp(i+1, Nx) + Tp(i-1, Nx)); end
        for j = 2:Nx-1, Tn(1, j)  = (1 - 4*Fo)*Tp(1, j)  + Fo*(2*Tp(2, j) + Tp(1, j+1) + Tp(1, j-1)); end
        for j = 2:Nx-1, Tn(Ny, j) = (1 - 4*Fo)*Tp(Ny, j) + Fo*(2*Tp(Ny-1, j) + Tp(Ny, j+1) + Tp(Ny, j-1)); end
        
        Tn(1, 1)   = (1 - 4*Fo - 2*Bi*Fo)*Tp(1, 1) + 2*Fo*(Tp(1, 2) + Tp(2, 1) + Bi*T_inf);
        Tn(Ny, 1)  = (1 - 4*Fo - 2*Bi*Fo)*Tp(Ny, 1) + 2*Fo*(Tp(Ny, 2) + Tp(Ny-1, 1) + Bi*T_inf);
        Tn(1, Nx)  = (1 - 4*Fo)*Tp(1, Nx) + 2*Fo*(Tp(1, Nx-1) + Tp(2, Nx));
        Tn(Ny, Nx) = (1 - 4*Fo)*Tp(Ny, Nx) + 2*Fo*(Tp(Ny, Nx-1) + Tp(Ny-1, Nx));
        
        for i = i_h1_s+1:i_h1_e-1
            Tn(i, j_h1_s) = (1 - 4*Fo)*Tp(i, j_h1_s) + Fo*(2*Tp(i, j_h1_s-1) + Tp(i+1, j_h1_s) + Tp(i-1, j_h1_s)) + (2*Fo*delta_x/k)*q_flux;
            Tn(i, j_h1_e) = (1 - 4*Fo)*Tp(i, j_h1_e) + Fo*(2*Tp(i, j_h1_e+1) + Tp(i+1, j_h1_e) + Tp(i-1, j_h1_e)) + (2*Fo*delta_x/k)*q_flux;
        end
        for j = j_h1_s+1:j_h1_e-1
            Tn(i_h1_s, j) = (1 - 4*Fo)*Tp(i_h1_s, j) + Fo*(2*Tp(i_h1_s-1, j) + Tp(i_h1_s, j+1) + Tp(i_h1_s, j-1)) + (2*Fo*delta_x/k)*q_flux;
            Tn(i_h1_e, j) = (1 - 4*Fo)*Tp(i_h1_e, j) + Fo*(2*Tp(i_h1_e+1, j) + Tp(i_h1_e, j+1) + Tp(i_h1_e, j-1)) + (2*Fo*delta_x/k)*q_flux;
        end
        
        Tn(i_h1_s, j_h1_s) = (1 - 4*Fo)*Tp(i_h1_s, j_h1_s) + 2*Fo*(Tp(i_h1_s-1, j_h1_s) + Tp(i_h1_s, j_h1_s-1)) + (4*Fo*delta_x/k)*q_flux;
        Tn(i_h1_e, j_h1_s) = (1 - 4*Fo)*Tp(i_h1_e, j_h1_s) + 2*Fo*(Tp(i_h1_e+1, j_h1_s) + Tp(i_h1_e, j_h1_s-1)) + (4*Fo*delta_x/k)*q_flux;
        Tn(i_h1_s, j_h1_e) = (1 - 4*Fo)*Tp(i_h1_s, j_h1_e) + 2*Fo*(Tp(i_h1_s-1, j_h1_e) + Tp(i_h1_s, j_h1_e+1)) + (4*Fo*delta_x/k)*q_flux;
        Tn(i_h1_e, j_h1_e) = (1 - 4*Fo)*Tp(i_h1_e, j_h1_e) + 2*Fo*(Tp(i_h1_e+1, j_h1_e) + Tp(i_h1_e, j_h1_e+1)) + (4*Fo*delta_x/k)*q_flux;
        
        Tn(i_h2_s:i_h2_e, j_h2_s:j_h2_e) = T_c;
        Tp = Tn;
        cycle_T(c_idx) = Tp(i_mid, j_mid);
    end
    
    plot(cycle_time, cycle_T, colors{idx}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('T_c = %d °C', T_c));
end

grid on;
xlim([0, period]);
ylim([46, 66]);
xlabel('Time within One Quasi-Steady Cycle (s)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Center Point Temperature (°C)', 'FontSize', 11, 'FontWeight', 'bold');
title('Effect of Cooling Fluid Temperature (T_c) on Center Point Oscillations', ...
      'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'east');
