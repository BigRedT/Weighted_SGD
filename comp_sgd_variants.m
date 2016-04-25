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
lambdas = [0.2:0.2:0.9];

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

%% Find lambda best
% k_cand = [];
% lambda_cand = 0:0.01:1;
% for i = 1:numel(lambda_cand)
%     k_cand = [k_cand; compute_k(w0, tol, w_star, x, C, residual, lambda_cand(i))];
% end
% [~, min_ind] = min(k_cand);
% lambda_best = lambda_cand(min_ind);
% disp(['Lambda Best: ', num2str(lambda_best)]);
% lambda = lambda_best;

lr_uniform = get_lr( L, tol, 1, residual );
lr_fully_weighted = get_lr( L, tol, 0, residual );

lr_partially_weighted = zeros(1, numel(lambdas));
for i = 1:numel(lambdas)
    lr_partially_weighted(i) = get_lr( L, tol, lambdas(i), residual );
end

%% Run weighted sgd with uniform sampling
disp('Uniform SGD')
w_uniform = sgd(w0, w_star, x,y,C,lr_uniform,1,tol);

%% Run weighted sgd fully weighted sampling
disp('Fully Weighted')
w_fully_weighted = sgd(w0, w_star, x,y,C,lr_fully_weighted,0,tol);

%% Run weighted sgd partially weighted sampling
disp('Partially Weighted')
w_partially_weighted = cell(1, numel(lambdas));
for i = 1:numel(lambdas)
    w_partially_weighted{i} = sgd(w0, w_star, x,y,C,lr_partially_weighted(i),lambdas(i),tol);
end

% Show solutions
figure; hold on;
vis_data(x, y, N1, N2);
plot_sol(w_star,x,'k');
plot_sol(w_uniform(:,end),x,'r')
plot_sol(w_fully_weighted(:,end),x,'g');
legendsol = {'pos','neg','optimal','uniform','fully weighted','partially weighted'};
colors = ['kcmbgrkcmbgr'];
for i = 1:numel(lambdas)
    curr_w_partially_weighted = w_partially_weighted{i};
    plot_sol(curr_w_partially_weighted(:,end),x,['-.', colors(i)]);
    legendsol{end+1} = ['lambda = ', num2str(lambdas(i))];
end
hold off;
legend(legendsol);
set(gca, 'fontsize', 16);

% Plot convergence
figure; hold on;
plot_convergence(w_fully_weighted, w_star, 'g', true);
plot_convergence(w_uniform, w_star, 'r', true);
legendconv = {'uniform','fully weighted','partially weighted'};
for i = 1:numel(lambdas)
    curr_w_partially_weighted = w_partially_weighted{i};
    plot_convergence(curr_w_partially_weighted, w_star, ['-.', colors(i)]);
    legendconv{end+1} = ['lambda = ', num2str(lambdas(i))];
end
hold off;
legend(legendconv);
set(gca, 'fontsize', 16);