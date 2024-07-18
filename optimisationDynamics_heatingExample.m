%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Switched optimisation for fair heat sharing
% Description: Heat sharing example from the paper 'Switching dynamics that 
% converges to the KKT point of a nonlinear optimization problem',
% submitted to XXX
% Authours: Joel Ferguson
% Version: 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Clear workspace and set default settings before execution
clear all
clc

% Set Figure default values
set(0,'DefaultTextInterpreter','latex');
set(0,'DefaultLegendInterpreter','latex');
set(0,'DefaultAxesFontSize',11);
set(0,'DefaultLineLineWidth',2.0);
set(0,'DefaultAxesLineWidth',0.5);
set(0,'defaultAxesXGrid','on')
set(0,'defaultAxesYGrid','on')
set(0,'defaultAxesNextPlot','add')

%% Define heating system parameters
C1 = 1;
C2 = 1;
E11 = 0.5;
E12 = -0.5;
E22 = 0.5;
R1 = 2;
R2 = 2;
Ta = 10;
Ts = 30;
T1s = 23;
T2s = 23;
A = 1;
q = 0;
epsilon = 1;
bar_m = 3;

%% Problem definition
% Initial start point for optimisation
z0 = [23; 23; 1];

% Define objective function; z1=T1, z2=T2, z3=m
L1 = 1;
L2 = 1;
Lm = 0.1;
f = @(z) 0.5*L1*(z(1) - T1s)^2 + 0.5*L2*(z(2) - T2s)^2 + 0.5*Lm*z(3)^2;

% The code below can be used to verify the objective function gains ensure
% a locally optimal solution as per (87)
% Y = E11 + inv(R1) - E12*(E22 + inv(R2))*E12.';
% L1 + E12*inv(E22 + inv(R2))*L2*inv(E22 + inv(R2))*E12.' - Y*inv(A)*Lm*inv(A)*Y/(2*epsilon^2);

% Define the equality constraints as per (71)
A_eq = @(z) [-E11-1/R1-A*z(3), -E12, A*(Ts-T1s);
            -E12, -E22-1/R2, 0]*diag([1/L1, 1/L2, 1/Lm]);
d_eq = [Ta/R1 + q - (E11 + 1/R1)*T1s - E12*T2s;
        Ta/R2 - E12*T1s - (E22+1/R2)*T2s];

% Define the inequality constraints as per (72)
A_ineq = @(z) [1 0 0];
d_ineq = [T1s - Ts + epsilon];

%% Run optimisation dynamics
% Simulation time
t_end = 6;

% Tuning parameters for optimisation
kappa1 = 1;
kappa2 = 1;
deltaT = 0.1;

% Run optimisation dynamics
[opt_dynamics_zStar, opt_dynamics_time, opt_dynamics_trajectory, opt_dynamics_cost] = optimisationDynamics(f,z0,A_eq,d_eq,A_ineq,d_ineq,t_end,kappa1,kappa2,deltaT);

% Print result
display(opt_dynamics_zStar)

%% Solve optimisation using fmincon
% Set up the linear constrain terms (none)
Aie = [];
bie = [];
Aineq = [];
beq = [];
lb = [];
ub = [];

% Create nonlinear contraint function formatted for fmincon
N = length(z0);
syms z_sym [N 1]
df_dz = matlabFunction(jacobian(f(z_sym),z_sym).','vars',{z_sym});
nonlcon_wrap = @(z) nonlcon(z,df_dz,A_eq,d_eq,A_ineq,d_ineq);

% Call the fmincon function to perform the optimization
[z_opt_fmincon, fval, exitflag] = fmincon(f, z0, Aie, bie, Aineq, beq, lb, ub, nonlcon_wrap);

% Print results from fmincon
display(z_opt_fmincon)

%% Plot results
% Compute objective function gradient - needed for constriant evaluation
df_dz = matlabFunction(jacobian(f(z_sym),z_sym).','vars',{z_sym});
% Evaluate equality constraints
for i=1:length(opt_dynamics_time)
    zi = opt_dynamics_trajectory(i,:).';
    g_eq(:,i) = A_eq(zi)*df_dz(zi) + d_eq;
end

% Plot the trajectories of the equality constraints
figure(1)
subplot(2,1,2)
plot(opt_dynamics_time,g_eq);
legend('$g_{eq_1}$','$g_{eq_2}$')
ylabel('Equality constraints')

% Plot the trajectory of the optimisation dynamics
subplot(2,1,1)
plot(opt_dynamics_time, opt_dynamics_trajectory)
legend('$T_1$','$T_2$','m')
ylabel('State evolution $z(t)$')

%% Functions
function [c, ceq] = nonlcon(w,df_dw,A_eq,d_eq,A_ineq,d_ineq)
    if isempty(d_ineq)
        c = [];
    else
        c = A_ineq(w)*df_dw(w) + d_ineq;
    end
    
    if isempty(d_eq)
        ceq = [];
    else
        ceq = A_eq(w)*df_dw(w) + d_eq;
    end
end