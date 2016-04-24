function vis_data(x, y, N1, N2)
scatter(x(1:N1,1),x(1:N1,2),'+')
scatter(x(N1+1:end,1),x(N1+1:end,2),'o')
end