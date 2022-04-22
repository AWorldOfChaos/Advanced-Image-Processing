% Implementation of Forward Model Matrix Adjoint At for independent CS
classdef forward_model_matrix_adjoint
    properties
        transform_function
        data_length
        proj_size
        angles
    end
    
    methods
        function obj = forward_model_matrix_adjoint (transform_function, data_length, proj_size, angles)
            obj.transform_function = transform_function;
            obj.data_length = data_length;
            obj.proj_size = proj_size;
            obj.angles = angles;
        end
        
        function product = mtimes(At, y)
            y = reshape(y, At.proj_size, size(At.angles, 2));
            theta = iradon(y, At.angles, 'linear', 'Ram-Lak', 1, At.data_length); 
            product = At.transform_function(theta);
            product = product(:);
        end
    end
end