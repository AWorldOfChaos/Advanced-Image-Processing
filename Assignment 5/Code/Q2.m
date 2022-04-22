% Setting seed for reproducibility
rng(0);

% Size of the signal
n = 128;
% Value of c in eigenvalue of covariance matrix
c = 1;
% Number of signals considered for each m
no_of_signals = 10;

% List of exponent alpha
alpha_list = [0, 3];
% List of number of measurements
m_list = [40, 50, 64, 80, 100, 120];
% Random Orthonormal Matrix while making covariance matrix
U = RandOrthMat(n);

fig = figure;
for l = 1:length(alpha_list)
    alpha = alpha_list(l);
    
    % Root of diagonalized matrix for the covariance matrix
    diagonal_elem = zeros(n, 1);
    for i = 1:n
        diagonal_elem(i) = sqrt(c * i ^ (-alpha));
    end

    % Finding the covariance matrix
    RootLambda = diag(diagonal_elem);
    A = U * RootLambda;
    Cov = A * A';
    % Initializing RMSE for given alpha
    RMSE = zeros(length(m_list), 1);
    
    for k = 1:length(m_list)
        % Number of measurements m
        m = m_list(k);
        % Measurement matrix
        Phi = randn(m, n) / sqrt(m);
    
        % Reconstructing for 10 different signals
        for ns = 1:no_of_signals
            x = A * randn(n, 1);
            norm_x = sqrt(sum(x .^ 2, 1));

            % Noiseless compressed measurement vector
            measurement = Phi * x;
            sigma = 0.01 * mean(abs(measurement));

            % Noise vector
            eta = sigma * randn(m, 1);
            % Measured signal
            y = Phi * x + eta;

            % Reconstructed signal using MAP estimate
            x_recon = inv(Phi' * Phi + sigma^2 * inv(Cov)) * Phi' * y;
            RMSE(k) = RMSE(k) + sqrt(sum((x - x_recon) .^2, 1)) / norm_x;
        end
        
        RMSE(k) = RMSE(k) / no_of_signals;
    end
    
    % Plotting a graph between log of RMSE and m
    plot(m_list, RMSE, '-o');
    hold on
end

ylabel('RMSE');
xlabel('m');
legend('alpha = 0', 'alpha = 3', 'Location', 'west');
title('Root Mean Square Error vs Number of Measurements');
% Saving the plot
saveas(fig, 'RMSE vs m.png');