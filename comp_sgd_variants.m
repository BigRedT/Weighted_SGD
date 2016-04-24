close all

%% Params
N1 = 100;
mu1 = [0, 2];
sigma1 = [10, 0; 0, 1];

N2 = 100;
mu2 = [0,-2];
sigma2 = [10, 0; 0, 1];

C = 0.01;
tol = 1e-5;
max_iter = 10000;
lr = 0.0001;
lambda = 0.5;

%% Generate data
[x, y] = gen_data(N1, N2, mu1, mu2, sigma1, sigma2, false);

%% Find optimal solution by gradient descent
disp('Gradient Descent')
w_star = gradient_descent(x,y,C,lr,tol);

%% Run weighted sgd with uniform sampling
disp('Uniform SGD')
w_uniform = sgd(x,y,C,lr,1,max_iter);

%% Run weighted sgd fully weighted sampling
disp('Fully Weighted')
w_fully_weighted = sgd(x,y,C,lr,0,max_iter);

%% Run weighted sgd partially weighted sampling
disp('Partially Weighted')
w_partially_weighted = sgd(x,y,C,lr,lambda,max_iter);

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
legend({'uniform','fully weighted','partially weighted'})