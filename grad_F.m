function g = grad_F(w,x,y,C)
g = w + C*sum(-x.*repmat(y.*sigmoid(y.*(x*w)),[1,2]),1)';
end