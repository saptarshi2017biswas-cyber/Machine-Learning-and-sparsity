%% Question-1

function Q1()
disp('>>> ENTERED Q1')

%% ================= PARAMETERS =================
m = 40;        % rows
n = 100;       % columns
k = 10;        % sparsity (non-zero entries)

rng(0);        % reproducibility

%% ================= DATA GENERATION =================
A = randn(m, n);

x_true = zeros(n,1);
idx = randperm(n, k);
x_true(idx) = randn(k,1);

b = A * x_true;

fprintf('True sparsity: %d non-zero elements\n', nnz(x_true));

%% ================= L1 OPTIMIZATION (LASSO) =================
lambda_l1 = 0.1;   % tune this if needed

cvx_begin quiet
    variable x_l1(n)
    minimize( sum_square(A*x_l1 - b) + lambda_l1 * norm(x_l1,1) )
cvx_end

%% ================= L2 OPTIMIZATION (RIDGE) =================
lambda_l2 = 0.1;

cvx_begin quiet
    variable x_l2(n)
    minimize( sum_square(A*x_l2 - b) + lambda_l2 * norm(x_l2,2) )
cvx_end

%% ================= ERROR ANALYSIS =================
err_l1 = norm(x_l1 - x_true);
err_l2 = norm(x_l2 - x_true);

fprintf('\n===== ERROR COMPARISON =====\n');
fprintf('L1 Error  : %.6f\n', err_l1);
fprintf('L2 Error  : %.6f\n', err_l2);

%% ================= SPARSITY CHECK =================
threshold = 1e-3;

sparsity_true = nnz(x_true);
sparsity_l1 = sum(abs(x_l1) > threshold);
sparsity_l2 = sum(abs(x_l2) > threshold);

fprintf('\n===== SPARSITY =====\n');
fprintf('True x  : %d non-zero\n', sparsity_true);
fprintf('L1 x    : %d non-zero\n', sparsity_l1);
fprintf('L2 x    : %d non-zero\n', sparsity_l2);

%% ================= SUPPORT RECOVERY =================
support_true = find(abs(x_true) > 0);
support_l1   = find(abs(x_l1) > threshold);

correct_support = intersect(support_true, support_l1);

fprintf('\n===== SUPPORT RECOVERY =====\n');
fprintf('Recovered correctly (L1): %d out of %d\n', ...
        length(correct_support), length(support_true));


%% ================= LAMBDA SWEEP (OPTIONAL) =================
lambda_vals = logspace(-3,1,10);
sparsity_vals = zeros(length(lambda_vals),1);

for i = 1:length(lambda_vals)
    lam = lambda_vals(i);
    
    cvx_begin quiet
        variable x_temp(n)
        minimize( sum_square(A*x_temp - b) + lam * norm(x_temp,1) )
    cvx_end
    
    sparsity_vals(i) = sum(abs(x_temp) > threshold);
end