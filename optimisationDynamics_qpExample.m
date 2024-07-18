%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Switched optimisation for QP problem
% Description: QP example from the paper 'Switching dynamics that 
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

%% General problem definition
% Define QP problem as per (57)
L = [1 -1; -1 2];
K = [-2; -6];
Bineq = [1 1; -1 2];
cineq = [-2; -2];

% Initial start point for optimisation
z0 = [-0.25; 0];

%% Convert the problem into form (1) for the optimisation dynamics
% Objective function
f = @(z) 0.5*z.'*L*z + K.'*z;

% Equality constraint definition
A_eq = @(z) [];
d_eq = [];

% Inequality constraint definition
A_ineq = @(z) Bineq/L;
d_ineq = cineq - (Bineq/L)*K;

%% Run optimisation dynamics
% Simulation time
t_end = 10;

% Tuning parameters for optimisation
kappa1 = 1;
kappa2 = 1;
deltaT = 0.1;

% Run optimisation dynamics
[opt_dynamics_zStar, opt_dynamics_time, opt_dynamics_trajectory, opt_dynamics_cost] = optimisationDynamics(f,z0,A_eq,d_eq,A_ineq,d_ineq,t_end,kappa1,kappa2,deltaT);

% Print result
display(opt_dynamics_zStar)

%% Solve optimisation using fmincon for comparison
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
% Define region to plot results
z1_span = -0.5:0.01:1;
z2_span = -0.5:0.01:2;

% Compute the inequality constraint lines over the specified domain
for i=1:length(z1_span)
    z2_ineq1(i) = -(Bineq(1,1)*z1_span(i)+cineq(1))/Bineq(1,2);
    z2_ineq2(i) = -(Bineq(2,1)*z1_span(i)+cineq(2))/Bineq(2,2);
end

% Evaluate the objective function on the specified domain
[Z1,Z2]=meshgrid(z1_span,z2_span);
for i=1:length(z1_span)
    for j=1:length(Z2)
        F(i,j) = f([z1_span(i); z2_span(j)]);
    end
end

% Generate a height map and contour plot of the objective function
contourf(Z1, Z2, F.', 100, 'LineStyle', 'none')
hold on
contour(Z1, Z2, F.', 10, 'k')

% Plot the inequality constriants
plot(z1_span,z2_ineq1,'r')
plot(z1_span,z2_ineq2,'y')
ylim([-0.5 2])

% Plot the optimisatino dynamics trajectory
plot(opt_dynamics_trajectory(:,1), opt_dynamics_trajectory(:,2),'g')
legend('Objective function height map','Objective function contour lines','Inequality constraint 1','Inequality constraint 2','Optimisation dynamics trajectory')
xlabel('$z_1$')
ylabel('$z_2$')

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