function w_sol = sgd(w_0, w_star, x,y,C,lr,lambda,tol)

N = size(x,1);
curr_tol = 10^5;
w_old = w_0;
k = 0;
L = comp_L(x, C);
p = lambda + (1-lambda)*L/(mean(L));
w_sol = [];

while curr_tol > tol & k < 10^5
    i = discretesample(p, 1);
    k = k + 1;
    w_new = w_old - (lr).*grad_f_i(w_old, x(i,:), y(i), C, N)/(p(i)+0.000001);
    curr_tol = norm(w_new - w_star);
    w_old = w_new;
    w_sol = [w_sol, w_old];
end

disp(['Itr# ', num2str(k), ' Tol: ', num2str(curr_tol)]);

end