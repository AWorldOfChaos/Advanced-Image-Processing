% Implementation of Forward Model Matrix A for independent CS
classdef forward_model_matrix
    properties
        transform_function
        data_length
        proj_size
        angles
    end
    
    methods
        function obj = forward_model_matrix (transform_function, data_length, proj_size, angles)
            obj.transform_function = transform_function;
            obj.data_length = data_length;
            obj.proj_size = proj_size;
            obj.angles = angles;
        end
        
        function product = mtimes(A, x)
            x = reshape(x, A.data_length, A.data_length);
            theta = A.transform_function(x);
            product = radon(theta, A.angles);
            product = product(:);
        end
    end
end