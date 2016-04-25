function plot_convergence(w, w_star, c, isbold)
if(nargin == 3)
    isbold = false;
end
n = size(w,2);
iter = [1:n];
w_diff = w - repmat(w_star,[1,n]);
err = sum(w_diff.^2,1);
if(isbold)
    plot(iter, err, c, 'LineWidth', 4);
else
    plot(iter, err, c, 'LineWidth', 2);
end
xlabel('Number of Iterations (k)');
ylabel('||w_k - w_0||^2');
end