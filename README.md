Consider the following undetermined system of equations
Ax = b.
Generate a real matrix A of dimension 40 × 100 and real vector x of dimention 100 × 1.
You could use “randn” MATLAB command to do that. Now make x sparse by setting
any of its 90 entries to zero. Generate b by using Ax = b. This step is called data
genetation.
Assume that you are now only given A and b, and you need to calculate x by solving the
basis basis pursuit denoising problem
min
x∈Rn
||Ax − b||2
2 + λ||x||1, λ ≥ 0 (1)
Code this optimization in CVX in either QP or as any generic convex program. Read
through CVX on how to do that. Note that λ is a tuning parameter. You should tune
λ values to see which one gives the sparse vector. Note that λ = 0 will not promote any
sparsity. Your optimization should giving x hich is same as actual x. Compare your result
by solving the problem with ℓ2 regularization.
min
x∈Rn
||Ax − b||2
2 + λ||x||2, λ ≥ 0 (2)
Do you get x which is sparse in above case?
