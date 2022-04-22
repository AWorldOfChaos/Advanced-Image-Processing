% Reading and padding 2 consecutive slices
slice_50 = im2double(imread("slice_50.png"));
slice_50 = padarray(slice_50, [37,19],  'both');
slice_51 = im2double(imread("slice_51.png"));
slice_51 = padarray(slice_51, [37,19],  'both');

% Padded image is of dimensions NxN
N = 255;

% Taking 18 uniformly separated angles in [0,180)
angles = 0:10:179;

% Finding the Radon Transforms along these angles
slice_50_projections = radon(slice_50, angles);
slice_51_projections = radon(slice_51, angles);

% Projection size for each slice (which is same, in fact)
proj_size_50 = size(slice_50_projections, 1);
proj_size_51 = size(slice_51_projections, 1);

% Creating the forward model matrices and their adjoints for both slices
A_50 = forward_model_matrix (@idct2, N, proj_size_50, angles);
A_51 = forward_model_matrix (@idct2, N, proj_size_51, angles);
At_50 = forward_model_matrix_adjoint (@dct2, N, proj_size_50, angles);
At_51 = forward_model_matrix_adjoint (@dct2, N, proj_size_51, angles);

% Arguments in the l1_ls algorithm
lambda = 0.1;
tar_gap = 1e-6;

% Finding the reconstructed image using independent compressed sensing
min_lasso_50 = l1_ls(A_50, At_50, proj_size_50 * 18, N*N, slice_50_projections(:), lambda, tar_gap, true);
min_lasso_51 = l1_ls(A_51, At_51, proj_size_51 * 18, N*N, slice_51_projections(:), lambda, tar_gap, true);
slice_50_recon = idct2(reshape(min_lasso_50, N, N));
slice_51_recon = idct2(reshape(min_lasso_51, N, N));

% Saving the reconstructed images
imwrite(slice_50_recon,'q3b_slice_50_reconstruction.png');
imwrite(slice_51_recon,'q3b_slice_51_reconstruction.png');
