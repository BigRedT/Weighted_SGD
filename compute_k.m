function outk = compute_k(w0, w, w_star, x, C, residual, lambda)

N = size(w, 2);
epsilon0 = norm(w0-w_star).^2;
L = comp_L(x, C);
meanL = mean(L);
mu = 1;
outk = [];

for i = 1:N
    
    wk = w(:, i);
    epsilonk = (norm(wk-w_star).^2);
    
    currk = 2*log10((2*epsilon0/epsilonk))*( ...
        min([meanL/(1-lambda), max(L)/lambda])/mu + ...
        (min([1/lambda, meanL/((1-lambda)*min(L))])*residual)/((mu^2)*epsilonk) ...
        );
    
    outk = [outk, currk];
end

end