function par = Exercise1(k)
%Load Data
load('Data.mat');
Input = Input;
Output = Output;
if nargin<1
    k = 5;
end

%main 
[optimal_p1,optimal_p2] = get_complexity(Input,Output,k);
disp(['When k = ' num2str(k) ',' 'the optimal value of p1 is ' num2str(optimal_p1) ]);
disp(['When k = ' num2str(k) ',' 'the optimal value of p2 is ' num2str(optimal_p2) ]);
par = get_parameters(Input,Output,optimal_p1,optimal_p2);
save('params','par');
Simulate_robot(0,0.05);
Simulate_robot(1,0);
Simulate_robot(1,0.05);
Simulate_robot(-1,-0.05);


%show the parameters
disp(['when k=' num2str(k) ' parameter column 1: ']);
disp(par{1});
disp(['when k=' num2str(k) ' parameter column 2: ']);
disp(par{2});
disp(['when k=' num2str(k) ' parameter column 3: ']);
disp(par{3});

%split data samples into traning set and validation set
function [Input_tra,Output_tra,Input_val,Output_val] = cross_val(Input,Output,k,K)
        Input_val = Input(:,1+((K-1)*size(Input,2)/k):K*size(Input,2)/k);
        Output_val = Output(:,1+((K-1)*size(Input,2)/k):K*size(Input,2)/k);
        Input_ = Input;
        Input_(:,1+((K-1)*size(Input,2)/k):K*size(Input,2)/k) = [];
        Input_tra = Input_;
        Output_ = Output;
        Output_(:,1+((K-1)*size(Input,2)/k):K*size(Input,2)/k) = [];
        Output_tra = Output_;
end

%compute the optimal value of p1 and p2 which denotes the complexity 
function [optimal_p1,optimal_p2] = get_complexity(Input,Output,k)
error_ori = zeros(k,6);
error_pos = zeros(k,6);
for K = 1:k
    %get the training set and validation set
    [Input_tra,Output_tra,Input_val,Output_val] = cross_val(Input,Output,k,K);
    
    %theta
    for p2 = 1:6;
        Input_tra_tem = Input_tra';
        Input_val_tem = Input_val';
        X = ones(size(Input_tra_tem,1),1);
        X_val = ones(size(Input_val_tem,1),1);
        for p = 1:p2
            add_1 = (Input_tra_tem).^p;
            add_2 = (Input_tra_tem(:,1).*Input_tra_tem(:,2)).^p;
            X = [X add_1 add_2];
            add_1_val = (Input_val_tem).^p;
            add_2_val = (Input_val_tem(:,1).*Input_val_tem(:,2)).^p;
            X_val = [X_val add_1_val add_2_val];
        end

        Y_theta = Output_tra(3,:)';
        Y_theta_val = Output_val(3,:)';
        W_theta = inv((X'*X))*X'*Y_theta;
        Y_pre_val = X_val * W_theta;
        %orientation_error = sum(((Y_pre_val - Y_theta_val).^2).^0.5)/size(Y_theta_val,1);
        orientation_error = sum(sqrt((Y_pre_val - Y_theta_val).^2))/size(Y_theta_val,1);
        error_ori(K,p2) = orientation_error;
    end
 
    %x and y
    for p1 = 1:6;
        Input_tra_tem = Input_tra';
        Input_val_tem = Input_val';
        X = ones(size(Input_tra_tem,1),1);
        X_val = ones(size(Input_val_tem,1),1);
        for p = 1:p1
            add_1 = (Input_tra_tem).^p;
            add_2 = (Input_tra_tem(:,1).*Input_tra_tem(:,2)).^p;
            X = [X add_1 add_2];
            add_1_val = (Input_val_tem).^p;
            add_2_val = (Input_val_tem(:,1).*Input_val_tem(:,2)).^p;
            X_val = [X_val add_1_val add_2_val];
        end
        Y_deltax = Output_tra(1,:)';
        Y_deltay = Output_tra(2,:)';
        Y_deltax_val = Output_val(1,:)';
        Y_deltay_val = Output_val(2,:)';
        W_deltax = (X'*X)^(-1)*X'*Y_deltax;
        W_deltay = (X'*X)^(-1)*X'*Y_deltay;
        Y_pre_val_deltax = X_val * W_deltax;
        Y_pre_val_deltay = X_val * W_deltay;
        pos_error=sum(sqrt((Y_pre_val_deltax-Y_deltax_val).^2+(Y_pre_val_deltay-Y_deltay_val).^2))/size(Y_pre_val_deltax,1);
        error_pos(K,p1) = pos_error;
    end
end
error_orien = mean(error_ori);
error_posi = mean(error_pos);
[~, optimal_p2] = min(error_orien);
[~, optimal_p1] = min(error_posi);
end

%compute the parameters under optimal value of p1 and p2
function par = get_parameters(Input,Output,p1,p2)
    par = cell(1,3);
    Input_tem = Input';
    X = ones(size(Input_tem,1),1);
    for p = 1:p2
        add_1 = (Input_tem).^p;
        add_2 = (Input_tem(:,1).*Input_tem(:,2)).^p;
        X = [X add_1 add_2];        
    end
    Y_theta = Output(3,:)';
    W_theta = inv((X'*X))*X'*Y_theta;
    par{3} = W_theta;
    Input_tem = Input';
    X = ones(size(Input_tem,1),1);
    for p = 1:p1
        add_1 = (Input_tem).^p;
        add_2 = (Input_tem(:,1).*Input_tem(:,2)).^p;
        X = [X add_1 add_2];
    end
    Y_deltax = Output(1,:)';
    Y_deltay = Output(2,:)';
    W_deltax = (X'*X)^(-1)*X'*Y_deltax;
    W_deltay = (X'*X)^(-1)*X'*Y_deltay;
    par{1} = W_deltax;
    par{2} = W_deltay;
end
end

