%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Switched optimisation for Rosenbrock's function
% Description: Rosenbrock's function example from the paper 'Switching 
% dynamics that converges to the KKT point of a nonlinear optimization 
% problem',
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

%% Problem definition
% Initial start point for optimisation
z0 = [1.0; -1.0];

% Define Rosenbrock's function as per https://au.mathworks.com/help/optim/ug/fmincon.html
f = @(z) 100*(z(2) - z(1)^2)^2 + (1 - z(1))^2;

% Equality constraint definition
A_eq = @(z) [];
d_eq = [];

% Inequality constraint definition
a = -2;
b = 1;
c = 0.75;
A_ineq = @(z) [0.5*(a+b)+0.5*b*z(1), b*z(1)^2 + (a+b)*z(1) + b/200];
d_ineq = [a+b+c];

%% Run optimisation dynamics
% Simulation time
t_end = 20;

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
% Define region to plot results
z1_span = 0:0.01:1.5;
z2_span = -1.0:0.01:1.5;

% Compute the inequality constraint lines over the specified domain
for i=1:length(z1_span)
    z2_ineq(i) = (-a*z1_span(i)-c)/b;
end

% Evaluate the objective function on the specified domain
[Z1,Z2]=meshgrid(z1_span,z2_span);
for i=1:length(z1_span)
    for j=1:length(Z2)
        F(i,j) = log(f([z1_span(i); z2_span(j)]));
    end
end

% Generate a height map and contour plot of the objective function
contourf(Z1, Z2, F.', 100, 'LineStyle', 'none')
hold on
contour(Z1, Z2, F.', 10, 'k')

% Plot the inequality constriants
plot(z1_span,z2_ineq,'r')
ylim([-1 1.5])

% Plot the optimisatino dynamics trajectory
plot(opt_dynamics_trajectory(:,1), opt_dynamics_trajectory(:,2),'g')
legend('Objective function height map','Objective function contour lines','Inequality constraint','Optimisation dynamics trajectory')
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