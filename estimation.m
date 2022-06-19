
clear;
TASKID = str2double(getenv('SLURM_ARRAY_TASK_ID'));
MODE = string(getenv('MODE'));
batchSize = str2double(getenv('batchSize'));

% TASKID = 0;
% MODE = "fmincon";
% batchSize = 10;

tol = 1e-7;

%% load data

D = readmatrix("data/data.csv");
[T,n] = size(D);
n = n - 1;
D = reshape(D(:,2:end)', [n 1 T]);

%% parameter search

func = @(param)loglikelihood(param, D);
nonlcon = @(param)stability(param, n);

options = optimoptions(MODE, ...
    'PlotFcns', 'optimplotfval', 'Display', 'iter', ...
    'MaxFunctionEvaluations', inf, 'MaxIterations', inf, ...
    'StepTolerance', tol, 'FunctionTolerance', tol);
[param, lb, ub] = define_param();

sample = sobolset(size(param,1));
exitflags = zeros(batchSize, 1) + nan;
x0s = zeros(batchSize, size(param,1)) + nan;
xs = zeros(batchSize, size(param,1)) + nan;
fvals = zeros(batchSize, 1) + nan;
grads = zeros(batchSize, size(param,1)) + nan;
hessians = zeros(batchSize, size(param,1)*size(param,1)) + nan;
serrs = zeros(batchSize, size(param,1)) + nan;

cnt = 0;
seq = 0;
while cnt < batchSize
    cnt = cnt + 1;
	rej = 0;
    while true
        x0 = 2 * sample(1 + TASKID + seq * 1000, :);
        x0 = x0(:);
	    rej = rej + 1;
        seq = seq + 1;
        if ~isnan(func(x0)); break; end
    end
	if rej > 0
	    disp(string(rej) + "resampling because objective undefined at initial point.");
	end

    if MODE == "fmincon"
        [x, fval, exitflag, output, ~, grad, hessian] = fmincon(func, x0, [], [], [], [], lb, ub, nonlcon, options);
    elseif MODE == "fminunc"
        [x, fval, exitflag, output, grad, hessian] = fminunc(func, x0, options);
    end
    exitflags(cnt) = exitflag;
    x0s(cnt,:) = x0';
    xs(cnt,:) = x';
    fvals(cnt) = fval;
    grads(cnt,:) = grad';
    hessians(cnt,:) = hessian(:)';
    try
        chol(hessian);
        serrs(cnt,:) = sqrt(diag(inv(hessian)))';
    catch
        disp("result hessian not positive definite.");
    end

    writematrix(exitflags, sprintf("output/estimation/%s_%s_%d.dat", MODE, "exitflags", TASKID));
    writematrix(x0s, sprintf("output/estimation/%s_%s_%d.dat", MODE, "x0s", TASKID));
    writematrix(xs, sprintf("output/estimation/%s_%s_%d.dat", MODE, "xs", TASKID));
    writematrix(fvals, sprintf("output/estimation/%s_%s_%d.dat", MODE, "fvals", TASKID));
    writematrix(grads, sprintf("output/estimation/%s_%s_%d.dat", MODE, "grads", TASKID));
    writematrix(hessians, sprintf("output/estimation/%s_%s_%d.dat", MODE, "hessians", TASKID));
    writematrix(serrs, sprintf("output/estimation/%s_%s_%d.dat", MODE, "serrs", TASKID));
end
