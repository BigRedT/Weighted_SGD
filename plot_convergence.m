function plot_convergence(w, w_star, c)
n = size(w,2);
iter = [1:n];
w_diff = w - repmat(w_star,[1,n]);
err = sum(w_diff.^2,1);
plot(iter, err, c);
end