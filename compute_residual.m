function residual = compute_residual(w, x, y, C)

residual = 0;
N = size(x, 1);
for i = 1:N
    residual = residual + norm(grad_f_i(w,x(i, :),y(i),C,N))^2;
end
residual = residual/N; 

end
