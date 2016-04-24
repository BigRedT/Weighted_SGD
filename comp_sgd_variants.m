close all;
clear;

%% Params
N1 = 100;
mu1 = [0, 15];
sigma1 = [10, 0; 0, 10];

N2 = 100;
mu2 = [0,-15];
sigma2 = [10, 0; 0, 10];

C = 1;
max_iter = 10000;
lrgd = 0.0001;
tolgd = 1e-5;
lambda = 0.5;

%SGD Params
w0 = [1; 1];
tol = 1e-2;

%% Generate data
[x, y] = gen_data(N1, N2, mu1, mu2, sigma1, sigma2, false);
x = x./10;

%% Find optimal solution by gradient descent
disp('Gradient Descent')
w_star = gradient_descent(x,y,C,lrgd,tolgd);

%% Find residual and learning rate
residual = compute_residual(w_star, x, y, C);
L = comp_L(x, C);

lr_uniform = get_lr( L, tol, 1, residual );
lr_fully_weighted = get_lr( L, tol, 0, residual );
lr_partially_weighted = get_lr( L, tol, lambda, residual );

%% Run weighted sgd with uniform sampling
disp('Uniform SGD')
w_uniform = sgd(w0, w_star, x,y,C,lr_uniform,1,tol);

%% Run weighted sgd fully weighted sampling
disp('Fully Weighted')
w_fully_weighted = sgd(w0, w_star, x,y,C,lr_fully_weighted,0,tol);

%% Run weighted sgd partially weighted sampling
disp('Partially Weighted')
w_partially_weighted = sgd(w0, w_star, x,y,C,lr_partially_weighted,lambda,tol);

% Show solutions
figure; hold on;
vis_data(x, y, N1, N2);
plot_sol(w_star,x,'k');
plot_sol(w_uniform(:,end),x,'r')
plot_sol(w_fully_weighted(:,end),x,'g');
plot_sol(w_partially_weighted(:,end),x,'b');
hold off;
legend({'pos','neg','optimal','uniform','fully weighted','partially weighted'})

% Plot convergence
figure; hold on;
plot_convergence(w_uniform, w_star, 'r');
plot_convergence(w_fully_weighted, w_star, 'g');
plot_convergence(w_partially_weighted, w_star, 'b');
hold off;
legend({'uniform','fully weighted','partially weighted'});

% Compute k for sgd variants
k_uniform = compute_k(w0, w_uniform(:,end), w_star, x, C, residual, 1);
k_fully_weighted = compute_k(w0, w_fully_weighted(:,end), w_star, x, C, residual, 0);
k_partially_weighted = compute_k(w0, w_partially_weighted(:,end), w_star, x, C, residual, lambda);

% 
% 
% figure; hold on;
% plot([1:max_iter], [1:max_iter], '-.k');
% plot([1:max_iter], k_uniform, 'r');
% plot([1:max_iter], k_fully_weighted, 'g');
% plot([1:max_iter], k_partially_weighted, 'b');
% hold off;
% ylim([1,max_iter]);
% xlim([1, max_iter]);
% legend({'linear','uniform','fully weighted','partially weighted'});
% 
