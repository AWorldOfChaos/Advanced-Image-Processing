% Seed set for reproducibility
rng(12345);

% Array of different number of frames taken
T_array = [3, 5, 7];

% Height of cropped frame taken
H = 120;

% Width of cropped frame taken
W = 240;

for t = 1:length(T_array)
    disp('Number of frames: ' + string(T_array(t)));
    % Reading the video, for the first T frames
    [video, audio] = mmread('cars.avi', 1:T_array(t));
    
    % Original dimensions of the video
    orig_H = video.height;
    orig_W = video.width;
    
    % Making array of size HxW for each of the T frames
    F = zeros(H, W, T_array(t));

    for i = 1:T_array(t)  
        % Taking out the bottom right HxW patch of the frame
        cut_frame = video.frames(i).cdata(orig_H-H+1: orig_H, orig_W-W+1: orig_W,:,1);
        % Converting to grayscale
        F(:,:,i) = double(rgb2gray(cut_frame));
    end
    
    % Random code pattern of size HxWxT
    C = randi([0,1], H, W, T_array(t));
    
    % Finding the coded snapshot without noise
    noiseless = sum(C.*F, 3);
    
    % Gaussian noise of standard deviation 2
    sigma = 2;
    noise = sigma * randn(H, W);
    
    % Adding the noise to the coded snapshot and saving it
    E = noise + noiseless;
    imwrite(mat2gray(E), string(T_array(t)) + '_coded_snapshot.png');
    
    % Size of 8x8xT patch
    N = 64*T_array(t);

    % 1D DCT matrix of size 8x8
    DCT = dctmtx(8);
    
    % Taking kronecker product and its transpose to find the 2D 64x64 DCT matrix
    DCT = kron(DCT, DCT);
    DCT = DCT';

    % Making a block diagonal matrix (Phi) with T occurences of the 2D DCT basis matrix
    temp = repmat({DCT}, T_array(t), 1);
    threeD = blkdiag(temp{:});
    
    % Matrix to store the reconstructed value for the HxWxT original frames
    recon = zeros(H, W, T_array(t));
    
    % Matrix to store the number of times a reconstructed value for each
    % index was calculated using the OMP procedure
    occur = zeros(H, W, T_array(t));
    
    A = zeros(64, N);
    for i = 1:H-7
        for z = 1:2:W-7
            for j = 1:T_array(t)
                % Matrix constructed by combining diag(C_t) column-wise (Psi)
                A(:, (j-1) * 64 + 1: j * 64) = diag(reshape(C(i: i+7, z: z+7, j), 64, 1));
            end

            % A matrix is product of Phi and Psi (measurement matrix)
            A_fin = A * threeD;
            
            % Orthogonal Matching Pursuit Algorithm
            y = reshape(E(i: i+7, z: z+7, 1), 64, 1);
            r = y;
            % Reconstructed theta
            theta = [];
            % Support set
            T = [];
            
            % Error term (explained in report)
            epsilon = 9 * sigma * sigma * 64;

            while(norm(r)^2 > epsilon)
                m = 0;
                j = 1;
                % Finding argument k for which max is attained
                for k = 1:N
                    if abs(dot(r, A_fin(:,k)) / norm(A_fin(:,k))) > m
                        j = k;
                        m = abs(dot(r, A_fin(:,k)) / norm(A_fin(:,k)));
                    end
                end
                
                % Add j to the support set
                T(end + 1) = j;
                % Pseudo inverse used to find argument theta for which min is attained
                theta = pinv(A_fin(:, T)) * y;
                r = y - A_fin(:, T) * theta;
            end
            
            % Using the found theta to make reconstructed 8x8xT patch
            F_recon = threeD(:, T) * theta;
            F_recon = reshape(F_recon, 8, 8, T_array(t));
            
            for x = 0:7
                for y = 0:7
                    for j = 1:T_array(t)
                        recon(i+x, z+y, j) = recon(i+x, z+y, j) + F_recon(x+1, y+1, j);
                        occur(i+x, z+y, j) = occur(i+x, z+y, j) + 1;
                    end
                end
            end
            
        end
    end
    
    % Taking average of reconstructions over all overlapping pixels
    recon = recon ./ occur;
    
    % Relative mean squared error between reconstructed and original frames
    mse = sqrt(sum((recon(:) - F(:)).^2, 'all') / sum(F(:).^2, 'all'));
    disp('Relative Mean Squared Error with ' +  string(T_array(t)) + ' frames is ' + string(mse));
    
    % Saving the reconstructed frames
    for i = 1:T_array(t) 
        imwrite(mat2gray([recon(:,:,i) F(:,:,i)]), string(T_array(t)) + '_' + string(i) + '.png');
    end
end
