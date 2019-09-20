function Exercise1()
    %priors are stored in prior cell, means are stored in mean_cell and covariance matrices are stored in cov_cell 
    
    %load data
    load('dataGMM.mat');
    data = Data';
    [num_data,dim_data] = size(data);
    max_iteration = 200;

    % k-means algorithm to initialize the mean value and covariance 
    k = 4;
    [index,mean_matrix] = kmeans(data,k);
    data_cell = {};
    mean_cell = {};
    prior_cell = {};
    cov_cell = {};
    for j = 1:k
        data_cell{j} = data(find(index == j),:);
        mean_cell{j} = mean(data_cell{j},1);
        prior_cell{j} = size(data_cell{j},1)/num_data;
        cov_cell{j} = cov(data_cell{j});
    end

    %calculate the initial log likelihood
    pw_given_x = {};
    log_likelihood = 0;
    for i = 1:num_data
        ln = 0;
        for j = 1:k
            prob = gaussian(data(i,:),cov_cell{j},mean_cell{j});
            ln = ln + prior_cell{j}*prob;
        end
        log_likelihood = log_likelihood + log(ln);
    end
    
    % EM algorithm
    for iter = 1:max_iteration
        log_likelihood_old = log_likelihood;
        log_likelihood = 0;
        
        %E-step
        for i = 1:num_data
            for j = 1:k
                pw_given_x{j}(i) = e_step(data(i,:),mean_cell,prior_cell,cov_cell,j,k);
            end
        end
        
        %M-step
        [mean_cell,cov_cell,prior_cell] = m_step(pw_given_x,data,k);
        
        %compute the log likelihood to check whether the algorithm has converged
        for i = 1:num_data
            ln = 0;
            for j = 1:k
                probability = gaussian(data(i,:),cov_cell{j},mean_cell{j});
                ln = ln + prior_cell{j}*probability;
            end
            log_likelihood = log_likelihood + log(ln);
        end
        if log_likelihood == log_likelihood_old
            disp(['EM algorithm has converged after ' num2str(iter) ' iterations']);
            break
        end
    end
    
    %display priors, means and covariance matrices
    disp('Priors are: ');
    for j = 1:k
        disp(prior_cell{j});
    end
    disp('Means are: ');
    for j = 1:k
        disp(mean_cell{j});
        disp(' ');
    end
    disp('Covariance metrices are: ');
    for j = 1:k
        disp(cov_cell{j});
        disp(' ');
    end
    
    % plot the density value
    [X, Y] = meshgrid(-0.1:0.2/99:0.1,-0.1:0.2/99:0.1);
    density = combinepdf(X,Y,mean_cell,cov_cell, prior_cell,k);
    surf(X,Y,density);
    title('Gaussian Mixture Model');
    xlabel('X');
    ylabel('Y');
    zlabel('Z(Density)');
end


function [mean_cell,cov_cell,prior_cell] = m_step(pw_given_x_cell,x_matrix,k)
    %The input pw_given_x_cell is a cell
    %x_matrix is data matrix with shape (number of samples,dimension)
    %k is the number of components used in GMM
    mean_cell_local = {};
    cov_cell_local = {};
    prior_cell_local = {};
    nk = {};
    [num,~] = size(x_matrix);
    %calculate nk
    for j = 1:k
        nk{j} = 0;
        for i = 1:num
            nk{j} = nk{j} + pw_given_x_cell{j}(i);
        end
    end
    %update prior
    for j = 1:k
        prior_cell_local{j} = nk{j} / num;
    end
    %update mean
    for j = 1:k
        mean_cell_local{j} = [0 0];
        for i = 1: num
            mean_cell_local{j} = mean_cell_local{j} + pw_given_x_cell{j}(i) * x_matrix(i,:);
        end
        mean_cell_local{j} = mean_cell_local{j} ./ nk{j};
    end
    %update covariance
    for j = 1:k
        cov_cell_local{j} = [0 0;0 0];
        for i = 1:num
            cov_cell_local{j} = cov_cell_local{j} + pw_given_x_cell{j}(i) * (x_matrix(i,:)-mean_cell_local{j})'*(x_matrix(i,:)-mean_cell_local{j});
        end
        cov_cell_local{j} = cov_cell_local{j} ./ nk{j};
    end
    mean_cell = mean_cell_local;
    cov_cell = cov_cell_local;
    prior_cell = prior_cell_local;
end
        
function pw_given_x = e_step(x,mean_cell,prior_cell,cov_cell,class,k)
    %input shape of x is (1*dimension)
    p = gaussian(x,cov_cell{class},mean_cell{class});
    numerator = prior_cell{class} * p;
    denominator = 0;
    for j = 1:k
        p_2 = gaussian(x,cov_cell{j},mean_cell{j});
        denominator = denominator + prior_cell{j} * p_2;
    end
    pw_given_x = numerator / denominator;
end

function p = gaussian(x,covariance,mean_value)
    %calculate the pdf and the shape of input x is 1 * dimention, the shape of covariance matrix is
    %(2*2),the shape of mean_value is also (1*dimension)
    [m,n] = size(x);
    p = (1/sqrt((2*pi)^n*det(covariance)))*exp(-0.5*(x-mean_value)*inv(covariance)*(x-mean_value)');
end

function [density] = combinepdf(x,y,mean,covariance, prior_cell,k)
    %combine four Gaussian distribution(weighted sum of four gaussian distribution)
    %input mean and covariance are cells;
    %k is the number of components 
    density = 0;
    for j = 1:k
        density = density + mvnpdf([x(:),y(:)],mean{j},covariance{j})*prior_cell{j};
    end
    density = reshape(density,size(y));
end
