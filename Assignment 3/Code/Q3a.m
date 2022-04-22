% Reading and padding 2 consecutive slices
slice_50 = im2double(imread("slice_50.png"));
slice_50 = padarray(slice_50, [37,19],  'both');
slice_51 = im2double(imread("slice_51.png"));
slice_51 = padarray(slice_51, [37,19],  'both');

% Taking 18 uniformly separated angles in [0,180)
angles = 0:10:179;

% Finding the Radon Transforms along these angles
slice_50_projections = radon(slice_50, angles);
slice_51_projections = radon(slice_51, angles);

% Performing Filtered Back-Propagation using Ram-Lak filter
slice_50_recon = iradon(slice_50_projections, angles, 'linear', 'Ram-Lak', 1, 255);
slice_51_recon = iradon(slice_51_projections, angles, 'linear', 'Ram-Lak', 1, 255);

% Saving the reconstructed images
imwrite(slice_50_recon,'q3a_slice_50_reconstruction.png');
imwrite(slice_51_recon,'q3a_slice_51_reconstruction.png');