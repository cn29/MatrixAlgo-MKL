
% sparse matrix test
n = 20000;
k = 100;
smax = 20;
A = sprand(n, n, 0.0001);
Uk = rand(n, k);
UkT = Uk.';
x = randn(n, 1);
diagv = rand(k,1);
diagm = diag(diagv);
timer2 = 0;
timer3 = 0;
iter = 4;

timestart = tic;
%% for application one: normal_alg, A*x + Uk*(UkT*x)
disp(['App1 starts...'])
disp(['tictoc capturing...'])
timelist1 = [];
fprintf('Current s =   ');
for s = 1:smax
    fprintf('\b\b');
    if s >= 10
        fprintf('%d', s);
    else
        fprintf(' %d', s);
    end
    t2 = tic;
    for i = 1:iter
        r = new_alg(A, x, Uk, UkT, s, diagm, 1.01);
    end
    
    timer2 = toc(t2);
    timelist1(s) = timer2/iter;
end

%% for application two: removeA_alg, A*x are removed
fprintf('\n');
disp(['App1 starts...'])
disp(['tictoc capturing...'])

timelist3 = [];
fprintf('Current s =   ');
for s = 1:smax
    fprintf('\b\b');
    if s >= 10
        fprintf('%d', s);
    else
        fprintf(' %d', s);
    end
    t2 = tic;
    for i = 1:iter
        r = normal_alg(A, x, Uk, UkT, s);
    end
    timer2 = toc(t2);
    timelist3(s) = timer2/iter;
end

disp(['(timer2) tic toc: ' 9 num2str(timer2)])
disp(['(timer3) CPU time: ' 9 num2str(timer3)])

toc(timestart)
%%  plot results
xs = 1:smax;
plot(xs, timelist1, 'LineWidth', 2);
hold on;
%plot(xs, timelist2,'r', 'LineWidth', 2);
hold on;
plot(xs, timelist3, 'r', 'LineWidth', 2);
hold on;
%plot(xs, timelist4, 'r--', 'LineWidth', 2);
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

function r = new_alg(A, x, Uk, UkT, s, diagm, sigma)
    d = UkT * x;
    diagms = {};
    Wk = {};
    b = {};
    diagms{1} = diagm^(0);
    for j = 1:s
        diagms{j+1} = diagm^(j);
        W = zeros(size(diagm));
        for i = 1:j
            W = W + diagms{i}*(diagms{j+1}+sigma)^(j-i);
        end
        W = sigma*W;
        W = W + diagms{j+1};
        bj = W*d;
        b{j} = bj;
    end
    for j = 1:s
        x = A*x + sigma * Uk * b{j};
    end
    r = x;
end
