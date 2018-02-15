
% sparse matrix test
n = 20000;
k = 100;
smax = 20;
A = sprand(n, n, 0.0002);
Uk = rand(n, k);
UkT = Uk.';
x = randn(n, 1);
diagv = rand(k,1);
% diagm = diag(diagv);
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
    t1 = tic;
    for i = 1:iter
        r = new_alg(A, x, Uk, UkT, s, diagv, 1.01);
    end
    
    timer2 = toc(t1);
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
        r = normal_alg(A, x, Uk, UkT, s, 1.01);
    end
    timer2 = toc(t2);
    timelist3(s) = timer2/iter;
end


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
function r = normal_alg(A, x, Uk, UkT, s, sigma)
    for i = 1:s
        x = A * x + sigma * (Uk * (UkT * x));
    end
    r = x;
end

function r = removeA_alg(x, Uk, UkT, s)
    for i = 1:s
        x = Uk * (UkT * x);
    end
    r = x;
end

function r = new_alg(A, x, Uk, UkT, s, diagv, sigma)
    d = UkT * x;
    diagvs = {};
    Wk = {};
    b = {};
    diagvs{1} = diagv.^0;
    for j = 1:s
        diagvs{j+1} = diagv.^(j);
        W = zeros(size(diagv));
        for i = 1:j
            W = W + diag(diagvs{i}.'*(diagvs{j+1}+sigma).^(j-i));
        end
        W = sigma*W;
        W = W + diag(diagvs{j+1});
        bj = W*d;
        b{j} = bj;
    end
    for j = 1:s
        x = A*x + sigma * Uk * b{j};
    end
    r = x;
end
