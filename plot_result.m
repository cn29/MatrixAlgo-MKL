% plot
xs = 1:smax;
curve1 = plot(xs, timelist1, 'LineWidth', 2);
hold on;
curve2 = plot(xs, timelist2,'r', 'LineWidth', 2);
hold on;
curve3 = plot(xs, timelist3, 'b--', 'LineWidth', 2);
hold on;
curve4 = plot(xs, timelist4, 'r--', 'LineWidth', 2);
title('with or without A*x')
xlabel('s');
ylabel('running time in second');
m1 = "time of normal algo";
m2 = "cpu time of normal algo";
m3 = "time without A*x";
m4 = "cpu time without A*x";
legend([curve1, curve2, curve3, curve4], [m1, m2, m3, m4]);
