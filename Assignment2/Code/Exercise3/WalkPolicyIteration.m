function WalkPolicyIteration(s)
    %the function will display reward matrix, value of gamma, number of
    %iterations and plot the learned policy cartoon

    %define reword matrix
    R = [0 0 0 0;
        0 1 -1 -1;
        0 -1 -1 -1;
        0 0 0 0;
        -1 -1 0 1;
        0 0 0 0;
        0 0 0 0;
        -1 1 0 0;
        -1 -1 0 -1;
        0 0 0 0;
        0 0 0 0;
        -1 1 0 -1;
        0 0 0 0;
        0 0 -1 1;
        0 -1 -1 1;
        0 1 0 0];
    disp('The Reward matrix is:');
    disp(R);

    %define state transition matrix
    transition_matrix = [2 4 5 13;
                        1 3 6 14;
                        4 2 7 15;
                        3 1 8 16;
                        6 8 1 9 ;
                        5 7 2 10;
                        8 6 3 11;
                        7 5 4 12;
                        10 12 13 5;
                        9 11 14 6;
                        12 10 15 7;
                        11 9 16 8;
                        14 16 9 1;
                        13 15 10 2;
                        16 14 11 3;
                        15 13 12 4];
                 
    %iniitialize policy
    policy = ceil(rand(16,1)*4);

    %Discounted parameter
    gamma = 0.8;

    %initialize value function
    V = zeros(16,1);

    %Policy Iteration
    iteration = 0;
    iter_continue = 1;
    while iter_continue
        iteration = iteration +1;
    
        %(a)step policy evaluation
        eva_continue = 1;
        while eva_continue
            V_old = V;
            for i = 1:16
                V(i,1) = R(i,policy(i,1)) + gamma * V(transition_matrix(i,policy(i,1)),1);
            end
            if V_old == V
                eva_continue = 0;
            end
        end
        %(b)step policy improvement
        action = zeros(4,1);
        policy_old = policy;
        for i = 1:16
            for act = 1:4
                action(act) = R(i,act) + gamma * V(transition_matrix(i,act),1);
            end
            [~,index] = max(action);
            policy(i) = index;
        end
        if policy_old == policy
            iter_continue = 0;
            disp(['When gamma is ' num2str(gamma) ' and start state is ' num2str(s) ', the number of iteration is: ' num2str(iteration)]);
        end
    end

    %plot the results
    s_matrix = zeros(1,16);
    s_matrix(1,1) = s;
    for i = 1:15
        s_matrix(i+1) = transition_matrix(s_matrix(i),policy(s_matrix(i)));
    end
    walkshow(s_matrix);
    title(["Learned Policy with Policy Iteration and Start State "+num2str(s)]);
end
