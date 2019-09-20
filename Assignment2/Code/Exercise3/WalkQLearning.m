function WalkQLearning(s)
    %store the initial state
    initial_s = s;
    % epsilon
    epsilon = 0.2;

    %learning rate alpha
    learning_rate = 0.6;

    %initialize Q
    Q = zeros(16,4);

    %Discounted factor
    gamma = 0.8;

    %number of iteration
    T = 80000;
    
    %Q-Learning Algorithm
    for i = 1:T
        %choose a from s using epsilon-greedy policy
        a = epsilon_greedy(epsilon,Q,s);
        %observe r and s prime
        [s_prime,r] = SimulateRobot(s,a);
        %update Q(s,a)
        max_term = max(Q(s_prime,:));
        Q(s,a) = Q(s,a) + learning_rate * (r + gamma * max_term - Q(s,a));
        s = s_prime;
    end

    %get the policy
    policy = zeros(16,1);
    for state = 1:16
        [~,policy(state)] = max(Q(state,:));
    end

    % plot the policy "cartoon"
    s_matrix = zeros(1,16);
    s_matrix(1) = initial_s;
    for i = 1:15
        s_matrix(i+1) = SimulateRobot(s_matrix(i),policy(s_matrix(i)));
    end
    walkshow(s_matrix);
    title(["Learned Policy with Q Learning and Start State "+num2str(initial_s)]);

end

function [current_policy] = epsilon_greedy(epsilon,Q,state)
    %output the action according to epsilon greedy policy
    rand_number = rand;
    if rand_number > (1-epsilon)
        current_policy = ceil(rand*4);
    else
        [~,current_policy] = max(Q(state,:));
    end     
end