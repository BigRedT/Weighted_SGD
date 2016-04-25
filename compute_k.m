function outk = compute_k(w0, tol, w_star, x, C, residual, lambda)

L = comp_L(x, C);
meanL = mean(L);
mu = 1;

epsilon0 = norm(w0-w_star).^2;
epsilonk = tol;

outk = 2*log10((2*epsilon0/epsilonk))*( ...
    min([meanL/(1-lambda), max(L)/lambda])/mu + ...
    (min([1/lambda, meanL/((1-lambda)*min(L))])*residual)/((mu^2)*epsilonk) ...
    );
end