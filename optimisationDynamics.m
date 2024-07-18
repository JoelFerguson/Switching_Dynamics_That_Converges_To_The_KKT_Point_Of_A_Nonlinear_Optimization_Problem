%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Switched optimisation dynamics
% Description: Simulation files for the paper 'Switching dynamics that 
% converges to the KKT point of a nonlinear optimization problem',
% submitted to XXX
% Authours: Joel Ferguson
% Version: 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Main code for execution of the opimisation dynamics
function [opt_dynamics_zStar, opt_dynamics_time, opt_dynamics_trajectory, opt_dynamics_cost] = optimisationDynamics(f,z0,A_eq,d_eq,A_ineq,d_ineq,t_end,kappa1,kappa2,deltaT)
% % % % % % % % % % % % % % % % INITIALISE CONSTRAINTS % % % % % % % % % % % % % % % %
% Determine dimension of descision variables
N = length(z0);
% Create symbolic variables for gradient computations
syms z_sym [N 1]

% Compute the objective function gradient
df_dz = matlabFunction(jacobian(f(z_sym),z_sym).','vars',{z_sym});

% Determine the number of equality constraints
m = length(d_eq);
if m > 0
    % Define equality constraint and compute gradients
    g_eq = @(z) A_eq(z)*df_dz(z) + d_eq;
    dgeq_dz = matlabFunction(jacobian(g_eq(z_sym),z_sym),'vars',{z_sym});
else
    % Define empty matricies if there are no equality constraints
    g_eq = @(z) [];
    dgeq_dz = @(z) [];
end

% Determine the number of inequality constraints
p = length(d_ineq);
if p > 0
    % Define inequality constraint and compute gradients
    g_ineq = @(z) A_ineq(z)*df_dz(z) + d_ineq;
    dgineq_dz = matlabFunction(jacobian(g_ineq(z_sym),z_sym),'vars',{z_sym});
else
    % Define empty matricies if there are no inequality constraints
    g_ineq = @(z) [];
    dgineq_dz = @(z) [];
end

% Verify that initial conditions satisfy inequality constraints. Throw
% error if initial conditions do not satisfy the constraints
g_ineq0 = g_ineq(z0);
for i=1:p
    assert(g_ineq0(i) <= 0,"ERROR: Initial conditions for z do not satisfy inequality constraints")
end

% % % % % % % % % % % % % % % % CONSTRUCT OPTIMISATION DYNAMICS % % % % % % % % % % % % % % % %
% Create indicator vector to track which inequality constraints are active
ineq_idx = logical(false(p,1)); % false = inactive; true = active

% Compute the constraint A_A and d_A terms as per (13)
A = @(z,ineq_idx) constructAmatrix(z,A_eq,A_ineq,ineq_idx,p);
d = @(z,ineq_idx) constructDmatrix(d_eq,d_ineq,ineq_idx,p);

% Compute the gradient of the active constraint vector as per (16)
dg_dz = @(z,ineq_idx) construct_dgdz_matrix(z,dgeq_dz,dgineq_dz,ineq_idx,p);

% Define the matrix B_A as per (16)
B = @(z,ineq_idx) dg_dz(z,ineq_idx)*A(z,ineq_idx).';

% Check that the matrix B_A is positive definite at the initial conditions.
% If not, Assumption 6 is violated.
B_min_eig = min(eig(B(z0,ineq_idx) + B(z0,ineq_idx).'));
if ~isempty(B_min_eig)
    assert(B_min_eig > 1e-10, "ERROR: The matrix B_A is not positive-definite at the initial conditions, violating Assumption 6.")
end

% Construct left annihilator matrix G^perp as per (19)
G_perp = @(z,ineq_idx) null_extended(dg_dz(z,ineq_idx), N).';

% Contruct optimisation dynamics as per (20)
dz = @(z,ineq_idx) construct_dz(kappa1,kappa2,df_dz,A,B,d,G_perp,z,ineq_idx);

% Evaluate the inequality constraints at initial conditions to determine if 
% any constraints should be initially active by Algorithm 1
for i=1:p
    dgineq_dz0 = dgineq_dz(z0);
    dz0 = dz(z0,ineq_idx);
    if (abs(g_ineq0(i)) <= 1e-10 && dgineq_dz0(i,:)*dz0 >=0)
        ineq_idx(i) = true;
    end
end

% % % % % % % % % % % % % % % % RUN OPTIMISATION DYNAMICS % % % % % % % % % % % % % % % %
% Create wrapper function for ODE solver input structure
dz_wrap = @(t,z) dz(z,ineq_idx);

% Define stop event used to detect wne inequality constraints intersect
% with 0 or when the constraint gradients change signs. This prompts
% execution of Algorithm 1 to update the active constraint set.
stopEventWrapper = @(t,z) stopEvent(t,z,deltaT,g_ineq,ineq_idx,dz,dgineq_dz,p,A_eq,A_ineq,dgeq_dz,B);

% Specify simulation times and settings
t_sim = [0 t_end];
options = odeset('reltol',1e-6,'Events',stopEventWrapper);

% Define variables to store the solution results
res.z = z0.';
res.t = t_sim(1);

% Solve optimisation dynamics iteratively over the full simulation time.
% The ODE solver will stop when potential updates to the active constraint
% set are detected.
while res.t(end) < t_sim(end)
    % Define initial conditions for the next ODE solver iteration.
    z0_i = res.z(end,:).';

    % Define time for the next ODE solver iteration. The solver should
    % start from the termination time of the last solve.
    t_sim_i = [res.t(end), t_sim(end)];

    % Run ODE solver.
    [res.t_i, res.z_i] = ode23s(dz_wrap,t_sim_i,z0_i,options);

    % Pack results from latest iteration with existing results.
    res.t = [res.t; res.t_i];
    res.z = [res.z; res.z_i];

    % Check each of the constraints at the termination point and determine
    % if constraints should be added to the active set as per Algorithm 1.
    z_end = res.z(end,:).';
    g_ineq_end = g_ineq(z_end);
    dgineq_dz_end = dgineq_dz(z_end);
    for i=1:p
        if(~ineq_idx(i))
            dz_end = dz(z_end,ineq_idx);
            if (abs(g_ineq_end(i)) <= 1e-10 && dgineq_dz_end(i,:)*dz_end >= 0)
                ineq_idx(i) = true;
            end
        end
    end

    % Check each of the constraints at the termination point and determine
    % if constraints should be removed from the active set as per Algorithm 1.
    for i=1:p
        if(ineq_idx(i))
            ineq_idx_tmp = ineq_idx;
            ineq_idx_tmp(i) = false;
            dz_tmp = dz(z_end,ineq_idx_tmp);
            dgineq_i_dt_tmp = dgineq_dz_end(i,:)*dz_tmp;
            if(dgineq_i_dt_tmp < 0)
                ineq_idx(i) = false;
            end
        end
    end

    % Test is the matrix B_A is positive definite as required by Assumption
    % 6.
    B_min_eig = min(eig(B(z_end,ineq_idx) + B(z_end,ineq_idx).'));
    if ~isempty(B_min_eig)
        assert(B_min_eig > 1e-10, "ERROR: The matrix B_A is not positive-definite along solution trajectory, violating Assumption 6.")
    end

    % Redefine the wrapper function for ODE solver using the updated
    % constraint set
    dz_wrap = @(t,z) dz(z,ineq_idx);
    
    % Redefine the stop event wrapper functino using the updated active
    % constraint set.
    stopEventWrapper = @(t,z) stopEvent(t,z,deltaT,g_ineq,ineq_idx,dz,dgineq_dz,p,A_eq,A_ineq,dgeq_dz,B);
    options = odeset('reltol',1e-6,'Events',stopEventWrapper);
end

% % % % % % % % % % % % % % % % EVALUATE OPTIMISATION RESULTS % % % % % % % % % % % % % % % %
% Compute the value of the objective function along the optimisation
% dynamics solution
for i=1:length(res.t)
    res.f(i) = f(res.z(i,:).');
end
% Extract the final value of the dynamics
opt_dynamics_zStar = res.z(end,:).';
% Return the solution time vector, solution trajectory and cost function
% trajectory
opt_dynamics_time = res.t;
opt_dynamics_trajectory = res.z;
opt_dynamics_cost = res.f;

end

%% Define functions used internally to the main optimisation dynamics routine
% Matlab required explicit 'if' satement handling of the case where there
% are no constraints, compared to when there are some constriants. Some of 
% the functions below are defined to handle this scenario.

% Definition of the optimisation dynamics as per (20)
function dz = construct_dz(kappa1,kappa2,df_dz,A,B,d,Gp,z,ineq_idx)
    if isempty(A(z,ineq_idx))
        dz = -kappa2*Gp(z,ineq_idx).'*Gp(z,ineq_idx)*df_dz(z);
    else
        dz = -(kappa1*A(z,ineq_idx).'*(B(z,ineq_idx)\A(z,ineq_idx)) + kappa2*Gp(z,ineq_idx).'*Gp(z,ineq_idx))*df_dz(z) - kappa1*A(z,ineq_idx).'*(B(z,ineq_idx)\d(z,ineq_idx));
    end
end

% Computation of the G_perp matrix as per (19). This is an extension of
% Matlab's 'null' function, extended to handle empty inputs.
function null_matrix = null_extended(A,N)
    if isempty(A)
        null_matrix = eye(N);
    else
        null_matrix = null(A);
    end
end

% Construction of the A_A matrix as per (13)
function A = constructAmatrix(z,A_eq,A_ineq,ineq_idx,p)
    A_ineq_val = A_ineq(z);
    A = A_eq(z);
    for i=1:p
        if(ineq_idx(i))
            A = [A; A_ineq_val(i,:)];
        end
    end
end

% Construction of the d_A matrix as per (13)
function d = constructDmatrix(d_eq,d_ineq,ineq_idx,p)
    d = d_eq;
    for i=1:p
        if(ineq_idx(i))
            d = [d; d_ineq(i)];
        end
    end
end

% Construction of the constraint gradient dg_dz (16)
function dgdz = construct_dgdz_matrix(z,dgeq_dz,dgineq_dz,ineq_idx,p)
    dgdz = dgeq_dz(z);
    dgineq_dz_val = dgineq_dz(z);
    for i=1:p
        if(ineq_idx(i))
            dgdz = [dgdz; dgineq_dz_val(i,:)];
        end
    end
end

% Stop event function that is used to detect times when a potential switch
% in active inequality constraints occurs. The function evaluates:
% - Value of the inequality constriants
% - Value of the inequality constraint gradients of active constraints
% - Minimum eigenvalue of B_A matrix
% If a zero-crossing is detected for any of these values, the ODE solver is
% stopped and algorithm 1 is run.
function [value, isTerminal, direction] = stopEvent(t,z,deltaT,g_ineq,ineq_idx,dz,dgineq_dz,p,A_eq,A_ineq,dgeq_dz,B)
    persistent Ts
    if isempty(Ts)
        Ts = 0;
    end

    % Evaluate inequality constraints
    g_ineq_value = g_ineq(z);

    % Evaluate gradient of active inequality constraints
    dgineq_dz_value = dgineq_dz(z);
    % For active constraints determine the time derivative if constraint
    % was inactive
    dgineq_dt_tmp = ones(p,1);
    if t >= Ts + deltaT
        for i=1:p
            if(ineq_idx(i))
                ineq_idx_tmp = ineq_idx;
                ineq_idx_tmp(i) = false;
                dz_tmp = dz(z,ineq_idx_tmp);
                dgineq_dt_tmp(i) = dgineq_dz_value(i,:)*dz_tmp;
                if dgineq_dt_tmp(i) < 0
                    Ts = t;
                end
            end
        end
    end

    % Evaluate the eigenvalue of BA matrix to verify assumption 6
    BA = B(z,ineq_idx);
    BA_min_eig = min(eig(BA + BA.'));

    % Extract only the inactive constraints for examination
    value = [g_ineq_value(~ineq_idx);
            dgineq_dt_tmp(ineq_idx);
            BA_min_eig];

    % Stop integration when zero detected    
    isTerminal = ones(size(value));
    
    % detect all zeros
    direction = zeros(size(value));
end