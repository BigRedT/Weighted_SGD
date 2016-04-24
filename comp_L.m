function L = comp_L(x, C)
N = size(x,1);
L = 1 + 0.25*C*N*sum(x.^2,2);