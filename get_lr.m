function [ lr ] = get_lr( L, tol, lambda, residual )

meanL = mean(L);
numer = tol;
denom_l = 2*tol*min([meanL/(1-lambda), max(L)/lambda]);
denom_r = 2*residual*min([1/lambda, meanL/((1-lambda)*min(L))]);
lr = numer/(denom_l+denom_r);
end

