function [x, y] = gen_data(N1, N2, mu1, mu2, sigma1, sigma2, vis)

x = [mvnrnd(mu1, sigma1, N1); mvnrnd(mu2, sigma2, N2)];
y = [ones(N1,1); -ones(N2,1)];
if (vis)
    vis_data(x, y, N1, N2)
end
end

