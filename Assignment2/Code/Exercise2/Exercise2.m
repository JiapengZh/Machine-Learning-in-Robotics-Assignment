function Exercise2()


    %load data 
    %A is transition probability matrix
    %B is observation probability matrix
    %pi is initial state probability vector
    A = load('A.txt');
    B = load('B.txt');
    B = B';
    pi = load('pi.txt');
    Observation = load('Test.txt');

    [num_states,num_observation] = size(B);
    [T,num_sequences] = size(Observation);
    alpha = zeros(num_states,T);
    log_likelihood = zeros(1,num_sequences);

    %The classification results are stored in Labels which is a cell.
    Labels = cell(1,num_sequences);

    %Forward procedure
    for i = 1:num_sequences
        %initialization
        for j = 1:num_states
            observation = Observation(1,i);
            alpha(j,1) = pi(1,j) * B(j,observation);
        end
        %Induction
        for t = 2:T
            for j = 1:num_states
                observation = Observation(t,i);
                sum_term = alpha(:,t-1)'*A(:,j);
                alpha(j,t) = sum_term * B(j,observation);
            end
        end
        %Termination
        p_o_given_lambda = sum(alpha(:,T));
        log_likelihood(1,i) = log(p_o_given_lambda);
    end

    %label each sequence and store the results in cell 'Labels'
    for i = 1:num_sequences
        if log_likelihood(1:i) > (-115)
            Labels{1,i} = 'gesture1';
        else
            Labels{1,i} = 'gesture2';
        end
    end

    %display the results
    for i = 1:num_sequences
        disp(['The ' num2str(i) 'th sequence is classified as ' Labels{1,i}]);
    end
end

