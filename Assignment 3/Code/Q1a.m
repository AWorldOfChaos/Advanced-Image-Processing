% Seed set for reproducibility
rng(1);

% Reading the Barbara image file, and converting the values to double
x = double (imread("barbara256.png"));

% Dimension of one side of the image
N = 256;

% Gaussian noise of variance 3
sigma = sqrt(3);
noise = sigma * randn(N, N);

% Image with random Gaussian noise of variance 3
y = x + noise;

% 1D DCT matrix of size 8x8
DCT = dctmtx(8);

% Taking kronecker product and its transpose to find the 2D 64x64 DCT matrix
DCT = kron(DCT, DCT);
DCT = DCT';

% Matrix to store the reconstructed value for the NxN original image
recon = zeros(N, N);

% Matrix to store the number of times a reconstructed value for each
% index was calculated using the ISTA algorithm
occur = zeros(N, N);

% The measurement matrix is simply the 64x64 identity matrix
% So A is same as DCT and there is no compression in the measurement
A = DCT;

% Parameters used in the ISTA algorithm (set optimally)
alpha = 1.1;
lambda = 1;

% Dividing the image into 8x8 patches, and reconstructing each patch
for i = 1:N-7
    for z = 1:N-7
        
        % Theta can be randomly initialized, and so it is set to 0
        theta = zeros(64, 1);
        
        % Vectorizing the corresponding patch of the noise image
        y_small = reshape(y(i: i+7, z: z+7), 64, 1);
        
        % Running the ISTA algorithm for 10 iterations (gives good results)
        for iterations = 1:10
            % soft function has been implemented at the end of this file
            theta = soft(theta + (1/alpha) * A' * (y_small - A * theta), lambda / (2 * alpha));
        end
        
        % Using the found theta to make reconstructed 8x8 patch
        x_recon = A * theta;
        x_recon = reshape(x_recon, 8, 8);

        % Adding the reconstructed patch over the patch found till now
        % Average will be taken after all patches have been worked upon
        for p = 0:7
            for q = 0:7
                recon(i+p, z+q) = recon(i+p, z+q) + x_recon(p+1, q+1);
                occur(i+p, z+q) = occur(i+p, z+q) + 1;
            end
        end

    end
end

% Taking average of reconstructions over all overlapping pixels
recon = recon ./ occur;

% Relative mean squared error between reconstructed and original image
mse = sqrt(sum((recon(:) - x(:)).^2, 'all') / sum(x(:).^2, 'all'));
disp('Root Mean Squared Error is ' + string(mse));

% Saving the reconstructed images
fig = figure();
hold on

subplot(1,3,1);
imshow(x,[]);
title('Original Image');

subplot(1,3,2);
imshow(y,[]);
title('Noisy Image');

subplot(1,3,3);
imshow(recon,[]);
title('Reconstructed Image');

saveas(fig, 'q1a_reconstruction.png');

% soft function (used in ISTA algorithm)
% Gives solution to x in y = x + lambda * sgn(x)
% where sgn(x) is the signum function, applied element-wise
function x = soft(y, lambda)

    x = zeros(size(y));
    
    for i = 1:size(y)
        if y(i)>=lambda
            x(i) = y(i)-lambda;
        elseif y(i)<=-lambda
            x(i) = y(i)+lambda;
        else
            x(i) = 0;
        end
    end
    
end
