function plot_sol(w,x,c)
x_min = min(x(:,1));
x_max = max(x(:,1));
x_vals = [x_min:0.1:x_max];
y_vals = -(w(1).*x_vals)./w(2); 
plot(x_vals, y_vals,c);
xlabel('x_1');
ylabel('x_2');

