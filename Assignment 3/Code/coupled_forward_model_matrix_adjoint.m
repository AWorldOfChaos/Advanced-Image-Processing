% Implementation of Forward Model Matrix Adjoint At for Coupled CS
classdef coupled_forward_model_matrix_adjoint
    properties
        transform_function
        data_length
        proj_size
        angles_1
        angles_2
    end
    
    methods
        function obj = coupled_forward_model_matrix_adjoint (transform_function, data_length, proj_size, angles_1, angles_2)
            obj.transform_function = transform_function;
            obj.data_length = data_length;
            obj.proj_size = proj_size;
            obj.angles_1 = angles_1;
            obj.angles_2 = angles_2;
        end
        
        function product = mtimes(At, y)
            len = length(y);
            y1 = y(1:len/2);
            y2 = y(len/2+1:end);
            y1 = reshape(y1, At.proj_size, size(At.angles_1, 2));
            y2 = reshape(y2, At.proj_size, size(At.angles_2, 2));
            theta1 = iradon(y1, At.angles_1, 'linear', 'Ram-Lak', 1, At.data_length);
            theta2 = iradon(y2, At.angles_2, 'linear', 'Ram-Lak', 1, At.data_length);
            x1 = At.transform_function(theta1);
            x2 = At.transform_function(theta2);
            product = [x1(:) + x2(:); x2(:)];
        end
    end
end