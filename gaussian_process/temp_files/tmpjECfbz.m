
rand('twister', 0);
randn('state', 0);

a='Load the data, it should contain X and y.'
load 'tmpfUAlkS.mat'
X = double(X)
y = double(y)

% Load GPML
addpath(genpath('/work2/09909/smata/stampede3/gp-structure-search/source/gpml'));

% Set up model.
meanfunc = {@meanZero}
hyp.mean = [];

covfunc = {@covProd, {{@covMask, {[1 0 0], {@covPeriodic}}}, {@covSum, {{@covProd, {{@covMask, {[1 0 0], {@covPeriodic}}}, {@covMask, {[0 0 1], {@covSEiso}}}}}, {@covProd, {{@covMask, {[0 1 0], {@covSEiso}}}, {@covMask, {[0 0 1], {@covSEiso}}}}}}}}}
hyp.cov = [ 6.923729793346858 -4.0985637220727185 -2.2237330686482384 -0.5813888111391353 1.77595778568622 1.9630350056839216 0.5762744263532523 -3.0301121188821463 -4.644952387504164 -8.496610176678828 -1.4775497132760147 2.5071465361350755 ]

likfunc = @likGauss
hyp.lik = -11.624888323309083

% Repeat...
[hyp_opt, nlls] = minimize(hyp, @gp, -int32(100 * 3 / 3), @infExact, meanfunc, covfunc, likfunc, X, y);
% ...optimisation - hopefully restarting optimiser will make it more robust to scale issues
% [hyp_opt, nlls_2] = minimize(hyp_opt, @gp, -int32(100 * 3 / 3), @infExact, meanfunc, covfunc, likfunc, X, y);
% nlls = [nlls_1; nlls_2];
best_nll = nlls(end)

% Compute Hessian numerically for laplace approx
num_hypers = length(hyp_opt.cov);
hessian = NaN(num_hypers+1, num_hypers+1);
delta = 1e-6;
a='Get original gradients';
[nll_orig, dnll_orig] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, X, y);
for d = 1:(num_hypers+1)
    dhyp_opt = hyp_opt;
    if d <= num_hypers
        dhyp_opt.cov(d) = dhyp_opt.cov(d) + delta;
    else
        dhyp_opt.lik = dhyp_opt.lik + delta;
    end
    [nll_delta, dnll_delta] = gp(dhyp_opt, @infExact, meanfunc, covfunc, likfunc, X, y);
    hessian(d, :) = ([dnll_delta.cov, dnll_delta.lik] - [dnll_orig.cov, dnll_orig.lik]) ./ delta;
end
hessian = 0.5 * (hessian + hessian');
hessian = hessian + 1e-6*max(max(hessian))*eye(size(hessian));

save( '/scratch/09909/smata/induction_modeling/gaussian_process/temp_files/tmpRhTu_G.out', 'hyp_opt', 'best_nll', 'nlls', 'hessian' );
% exit();

fprintf('\nWriting completion flag\n');
ID = fopen('/scratch/09909/smata/induction_modeling/gaussian_process/temp_files/tmpob8XCA.flg', 'w');
fprintf(ID, 'Goodbye, world');
fclose(ID);
fprintf('\nGoodbye, World\n');
quit()
