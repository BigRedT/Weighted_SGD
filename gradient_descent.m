function w_sol = gradient_descent(x,y,C,lr,tol)
w_0 = [1; 1];
curr_tol = 10^5;
w_old = w_0;
k = 0;
while curr_tol > tol
    k = k + 1;
    w_new = w_old - lr.*grad_F(w_old, x, y, C);
    curr_tol = norm(w_new - w_old);
    w_old = w_new;
    %disp(['Itr# ' num2str(k), ' Tol: ', num2str(curr_tol)]);
end
disp(['Itr# ' num2str(k), ' Tol: ', num2str(curr_tol)]);

w_sol = w_new;

end


