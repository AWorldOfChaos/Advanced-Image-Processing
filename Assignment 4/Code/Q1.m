% Setting seed for reproducibility
rng(1);

% Size of the signal 
n = 500;
% Number of measurements
m = 200;
% Number of non-zero elements (sparsity) in the signal
s = 18;
% Non-zero elements in x are drawn randomly from Uniform(0, 1000)
x_max = 1000;

% Choosing s indices randomly out of n to place the non-zero elements
ind = randperm(n, s);
% Creating the original signal x
x = zeros(n, 1);
x(ind) = x_max * rand(s, 1);
% Norm of the original signal x
norm_x = sqrt(sum(x .^ 2, 1));

% Measurement matrix consists of elements 1/sqrt(m) and -1/sqrt(m) with
% equal probability of 0.5 each (Bernoulli random numbers)
p = 0.5;
phi = 2 .* binornd(1, p, m, n) - 1;
phi = phi / sqrt(m);

% Calculating the standard deviation of the noise signal
sigma = 0.05 * sum(abs(sum(phi .* (x(:, ones(m, 1))'), 2)), 1) / m;
% Noise signal is drawn from Gaussian(0, sigma^2)
eta = normrnd(0, sigma, m, 1);
% Finding the measured signal (size m vector)
y = phi * x + eta;

% Choosing a random set of size 0.9*m as the reconstruction set
R = randperm(m, 0.9 * m);
tot = (1:m)';
% Remaining elements form the validation set
V = setdiff(tot, R);
val = length(V);

% Set of possible values of lambda
Gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 50, 100]';
obs = length(Gamma);
% Initializing arrays for validation and RMS error terms for each lambda
VE = zeros(obs, 1);
RMSE = zeros(obs, 1);

for g = 1:obs
    lambda = Gamma(g);
    tar_gap = 0.000001;
    % Solving the LASSO problem over set R for this lambda
    x_g = l1_ls(phi(R, :), phi(R, :)', length(R), n, y(R), lambda, tar_gap, true);
    % Finding the validation error as given in the problem
    VE(g) = sum((y(V) - sum(phi(V, :) .* (x_g(:, ones(val, 1))'), 2)).^2, 1) / val;
    % Finding the root mean square error as given in the problem
    RMSE(g) = sqrt(sum((x - x_g) .^2, 1)) / norm_x;
end

% Plotting a graph between VE and log(lambda)
fig1 = figure;
plot(log10(Gamma), VE, 'b-o');
title('Validation Error Plot')
xlabel('log(lambda)');
ylabel('Validation Error');
% Saving the plot
saveas(fig1, 'VE vs lambda.png');

% Plotting a graph between RMSE and log(lambda)
fig2 = figure;
plot(log10(Gamma), RMSE, 'r-x');
title('Root Mean Square Error Plot')
xlabel('log(lambda)');
ylabel('Root Mean Square Error');
% Saving the plot
saveas(fig2, 'RMSE vs lambda.png');

% Finding the lambda for which VE is minimum
[M1,I1] = min(VE);
% Finding the lambda for which RMSE is minimum
[M2,I2] = min(RMSE);

disp('The optimal value of lambda while using validation error is ' + string(Gamma(I1)));
disp('The optimal value of lambda while using root mean square error is ' + string(Gamma(I2)));