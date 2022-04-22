% Implementation of Forward Model Matrix A for Coupled CS
classdef coupled_forward_model_matrix
    properties
        transform_function
        data_length
        proj_size
        angles_1
        angles_2
    end
    
    methods
        function obj = coupled_forward_model_matrix (transform_function, data_length, proj_size, angles_1, angles_2)
            obj.transform_function = transform_function;
            obj.data_length = data_length;
            obj.proj_size = proj_size;
            obj.angles_1 = angles_1;
            obj.angles_2 = angles_2;
        end
        
        function product = mtimes(A, x)
            len = length(x);
            x1 = x(1:len/2);
            x2 = x(len/2+1:end);
            x1 = reshape(x1, A.data_length, A.data_length);
            x2 = reshape(x2, A.data_length, A.data_length);
            theta1 = A.transform_function(x1);
            theta2 = A.transform_function(x2);
            R1U_theta1 = radon(theta1, A.angles_1);
            R2U_theta1 = radon(theta1, A.angles_2);
            R2U_theta2 = radon(theta2, A.angles_2);
            product = [R1U_theta1(:); R2U_theta1(:) + R2U_theta2(:)];
        end
    end
end