% Setting seed for reproducibility
rng(4);

% Reading and padding 2 consecutive slices
slice_50 = im2double(imread("slice_50.png"));
slice_50 = padarray(slice_50, [37,19],  'both');
slice_51 = im2double(imread("slice_51.png"));
slice_51 = padarray(slice_51, [37,19],  'both');

% Padded image is of dimensions NxN
N = 255;

% Taking 18 different angles uniformly chosen from (0,180) for both slices
angles_1 = 180 * rand(1, 18);
angles_2 = 180 * rand(1, 18);

% Finding the Radon Transforms along these angles
slice_50_projections = radon(slice_50, angles_1);
slice_51_projections = radon(slice_51, angles_2);

% Projection size for each slice
proj_size = size(slice_50_projections, 1);

% Creating one single forward model matrix and its adjoint for both slices
A = coupled_forward_model_matrix (@idct2, N, proj_size, angles_1, angles_2);
At = coupled_forward_model_matrix_adjoint (@dct2, N, proj_size, angles_1, angles_2);

% Arguments in the l1_ls algorithm
lambda = 0.1;
tar_gap = 1e-6;

% Finding the reconstructed image using coupled compressed sensing
theta = l1_ls(A, At, 2 * proj_size * 18, 2*N*N, [slice_50_projections(:); slice_51_projections(:)], lambda, tar_gap, true);
theta1 = theta(1:N*N);
delta_theta1 = theta(1+N*N: end);
theta2 = theta1 + delta_theta1;

slice_50_recon = idct2(reshape(theta1, N, N));
slice_51_recon = idct2(reshape(theta2, N, N));

% Saving the reconstructed images
imwrite(slice_50_recon,'q3c_slice_50_reconstruction.png');
imwrite(slice_51_recon,'q3c_slice_51_reconstruction.png');
