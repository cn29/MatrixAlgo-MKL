
% sparse matrix test
n = 40000;
k = 400;
smin = 1;
smax = 19;
A = sprand(n, n, 0.0001);
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
% fprintf('Current s =   ');
for s = smin:smax
%     fprintf('\b\b');
%     if s >= 10
%         fprintf('%d', s);
%     else
%         fprintf(' %d', s);
%     end
    new_alg(A, x, Uk, UkT, s, diagv, 1.01);
    t1 = tic;
    timer2 = 0;
    for i = 1:iter
        [r,timepart1, timepart2, timeall] = new_alg(A, x, Uk, UkT, s, diagv, 1.01);
        timer2 = timer2 + timeall;
    end
    %timer2 = toc(t1);
    timelist1(s) = timer2/iter;
    fprintf('s = %d \n', s)
    fprintf('Part1:\t%.4f \tPortion: %.4f%% \n',timepart1, 100*timepart1/(timepart1+timepart2))
    fprintf('Part2:\t%.4f \tPortion: %.4f%% \n',timepart2, 100*timepart2/(timepart1+timepart2))
    
end

%% for application two: removeA_alg, A*x are removed
fprintf('\n');
disp(['App1 starts...'])
disp(['tictoc capturing...'])

timelist3 = [];
fprintf('Current s =   ');
for s = smin:smax
    fprintf('\b\b');
    if s >= 10
        fprintf('%d', s);
    else
        fprintf(' %d', s);
    end
    r = normal_alg(A, x, Uk, UkT, s, 1.01);
    t2 = tic;
    timer2 = 0;
    for i = 1:iter
        [r,timeall] = normal_alg(A, x, Uk, UkT, s, 1.01);
        timer2 = timer2 + timeall;
    end
    % timer2 = toc(t2);
    timelist3(s) = timer2/iter;
end


toc(timestart)
%%  plot results
xs = 1:smax;
plot(xs, timelist1, 'b', 'LineWidth', 2);
hold on;
%plot(xs, timelist2,'r', 'LineWidth', 2);
hold on;
plot(xs, timelist3, 'r', 'LineWidth', 2);
hold on;
%plot(xs, timelist4, 'r--', 'LineWidth', 2);
title('with or without A*x')

%% functions
function [r, timeall] = normal_alg(A, x, Uk, UkT, s, sigma)
    t1 = tic;
    for i = 1:s
        x = A * x + sigma * (Uk * (UkT * x));
    end
    r = x;
    timeall = toc(t1);
end

function r = removeA_alg(x, Uk, UkT, s)
    for i = 1:s
        x = Uk * (UkT * x);
    end
    r = x;
end

function [r, timepart1, timepart2, timeall] = new_alg(A, x, Uk, UkT, s, diagv, sigma)
    t1 = tic;
    d = UkT * x;
    diagvs = [];
    Wk = [];
    b = [];
    diagvs(:,:,1) = diagv.^0;
    diagvpa = diagv+sigma;
    diagvpas(:,:,1) = diagvpa.^0;
    timepart1 = 0;
    timepart2 = 0;
    for j = 1:s
        t2 = tic;
        diagvpas(:,:,j+1) = diagvpas(:,:,j).'*diagvpa;
        diagvs(:,:,j+1) = diagvs(:,:,j).'*diagv;
        W = zeros(size(diagv));
        for i = 1:j
            W = W + diagvs(:,:,i).' * diagvpas(:,:,j-i+1);
        end
        W = sigma*W;
        W = W + diagvs(:,:,j+1);
        bj = diag(W)*d;
        timepart1 = timepart1 + toc(t2);
        t3 = tic;
        % b(:,:,j) = bj;
        x = A*x + sigma * (Uk * bj);
        timepart2 = timepart2 + toc(t3);
    end
    r = x;
    timeall = toc(t1);
end