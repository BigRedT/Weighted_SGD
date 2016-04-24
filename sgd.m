function w_sol = sgd(x,y,C,lr,lambda,max_iter)

N = size(x,1);
w_0 = [1; 1];
curr_tol = 10^5;
w_old = w_0;
k = 0;
L = comp_L(x, C);
p = lambda + (1-lambda)*L/(N*mean(L));
w_sol = [];
for iter=1:max_iter
    i = discretesample(p, 1);
    w_new = w_old - lr.*grad_f_i(w_old, x(i,:), y(i), C, N)/(p(i)+0.000001);
    curr_tol = norm(w_new - w_old);
    w_old = w_new;
    w_sol = [w_sol, w_old];
end


end