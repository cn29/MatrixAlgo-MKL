
% sparse matrix test
n = 200000;
k = 400;
smax = 20;
A = sprand(n, n, 0.0001);
Uk = rand(n, k);
UkT = Uk.';
x = randn(n, 1);
timer2 = 0;
timer3 = 0;
iter = 10;


timestart = tic;
%% for application one: normal_alg, A*x + Uk*(UkT*x)
disp(['App1 starts...'])
disp(['tictoc capturing...'])
timelist1 = [];
for s = 1:smax
    disp(['s = ' num2str(s)])
    t2 = tic;
    for i = 1:iter
        r = normal_alg(A, x, Uk, UkT, s);
    end
    timer2 = toc(t2);
    timelist1(s) = timer2/iter;
end

disp(['cputime capturing...'])
timelist2 = [];
for s = 1:smax
    start = cputime;
    for i = 1:iter
        r = normal_alg(A, x, Uk, UkT, s);
    end
    timer3 = cputime-start;
    timelist2(s) = timer3/iter;
end

%% for application two: removeA_alg, A*x are removed
disp(['App1 starts...'])
disp(['tictoc capturing...'])

timelist3 = [];
for s = 1:smax
    t2 = tic;
    for i = 1:iter
        r = removeA_alg(x, Uk, UkT, s);
    end
    timer2 = toc(t2);
    timelist3(s) = timer2/iter;
end

disp(['cputime capturing...'])
timelist4 = [];
for s = 1:smax
    start = cputime;
    for i = 1:iter
        r = removeA_alg(x, Uk, UkT, s);
    end
    timer3 = cputime-start;
    timelist4(s) = timer3/iter;
end


disp(['(timer2) tic toc: ' 9 num2str(timer2)])
disp(['(timer3) CPU time: ' 9 num2str(timer3)])

toc(timestart)
%%  plot results
xs = 1:smax;
plot(xs, timelist1, 'LineWidth', 2);
hold on;
plot(xs, timelist2,'r', 'LineWidth', 2);
hold on;
plot(xs, timelist3, '--', 'LineWidth', 2);
hold on;
plot(xs, timelist4, 'r--', 'LineWidth', 2);
title('with or without A*x')

%% functions
function r = normal_alg(A, x, Uk, UkT, s)
    for i = 1:s
        x = A * x + Uk * (UkT * x);
    end
    r = x;
end

function r = removeA_alg(x, Uk, UkT, s)
    for i = 1:s
        x = Uk * (UkT * x);
    end
    r = x;
end

