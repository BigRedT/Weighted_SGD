function g = grad_f_i(w,x,y,C,N)
g = w + C*N*(-x.*repmat(y.*sigmoid(y.*(x*w)),[1,2]))';
end