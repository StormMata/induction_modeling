
rand('twister', 0);
randn('state', 0);

a='Load the data, it should contain X and y.'
load 'tmpnSRHz5.mat'
X = double(X)
y = double(y)

% Load GPML
addpath(genpath('/work2/09909/smata/stampede3/gp-structure-search/source/gpml'));

% Set up model.
meanfunc = {@meanZero}
hyp.mean = [];

covfunc = {@covSum, {{@covProd, {{@covMask, {[1 0 0], {@covSum, {@covConst, @covLINscaleshift}}}}, {@covMask, {[0 1 0], {@covPeriodic}}}, {@covMask, {[0 0 1], {@covSEiso}}}}}, {@covProd, {{@covMask, {[0 0 1], {@covSEiso}}}, {@covMask, {[0 0 1], {@covPeriodic}}}, {@covSum, {{@covProd, {{@covMask, {[0 1 0], {@covSum, {@covConst, @covLINscaleshift}}}}, {@covMask, {[0 1 0], {@covSEiso}}}}}, {@covMask, {[1 0 0], {@covSEiso}}}}}}}}}
hyp.cov = [ -2.013604193939733 -0.9073068474057846 0.9395039122099283 2.4984568562691156 -2.1114162834493304 1.002132373567421 -0.17693735979545538 -2.309079754720957 -1.0430937690338333 0.43353057634698333 3.1583998709951464 -1.4802292731591509 -2.073615959788093 -13.337424530291548 -1.6782672757631134 0.7041537442136397 -6.007038736007914 -6.399749246668841 -3.210388776795705 -4.170475674438063 ]

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

save( '/scratch/09909/smata/induction_modeling/gaussian_process/temp_files/tmpymUAaG.out', 'hyp_opt', 'best_nll', 'nlls', 'hessian' );
% exit();

fprintf('\nWriting completion flag\n');
ID = fopen('/scratch/09909/smata/induction_modeling/gaussian_process/temp_files/tmpGOPQcT.flg', 'w');
fprintf(ID, 'Goodbye, world');
fclose(ID);
fprintf('\nGoodbye, World\n');
quit()
