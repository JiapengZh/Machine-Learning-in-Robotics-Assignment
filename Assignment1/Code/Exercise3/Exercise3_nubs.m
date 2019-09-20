function Exercise3_nubs(gesture_l,gesture_o,gesture_x,k)

if nargin < 4
    k = 7;
end

[centers_l,labels_l,X_l,Y_l,Z_l] = nub_split(gesture_l,k);
[centers_o,labels_o,X_o,Y_o,Z_o] = nub_split(gesture_o,k);
[centers_x,labels_x,X_x,Y_x,Z_x] = nub_split(gesture_x,k);
%plot color
[m,n,d] = size(gesture_l);
C = zeros(7,3);
%blue
C(1,:) = [0 0 1];
%black
C(2,:) = [0 0 0];
%red
C(3,:) = [1 0 0];
%green
C(4,:) = [0 1 0];
%magenta
C(5,:) = [1 0 1];
%yellow
C(6,:) = [1 1 0];
%cyan
C(7,:) = [0 1 1];

Colors_l = zeros(m*n,3);
Colors_o = zeros(m*n,3);
Colors_x = zeros(m*n,3);

% l-gesture
[centers_l,labels_l,X_l,Y_l,Z_l] = nub_split(gesture_l,k);
for i = 1:size(labels_l,1)
    Colors_l(i,:) = C(labels_l(i,1),:);
end
subplot(2,10,[1,4]);
scatter3(X_l,Y_l,Z_l,50,Colors_l);
title('l-gesture');

% o-gesture
[centers_o,labels_o,X_o,Y_o,Z_o] = nub_split(gesture_o,k);
for i = 1:size(labels_o,1)
    Colors_o(i,:) = C(labels_o(i,1),:);
end
subplot(2,10,[7,10]);
scatter3(X_o,Y_o,Z_o,50,Colors_o);
title('o-gesture');

% x-gesture
[centers_x,labels_x,X_x,Y_x,Z_x] = nub_split(gesture_x,k);
for i = 1:size(labels_x,1)
    Colors_x(i,:) = C(labels_x(i,1),:);
end
subplot(2,10,[14,17]);
scatter3(X_x,Y_x,Z_x,50,Colors_x);
title('x-gesture');

%function non-uniform binary split algorithm
function [centers,labels,X,Y,Z] = nub_split(gesture,k)
%transpose data into shape (600,3)
    [m,n,d] = size(gesture);
    data = zeros(m*n,d);
    for i = 1:m
        for j = 1:n
            data((i-1)*10+j,:) = gesture(i,j,:);
        end
    end
%initialization
    v = [0.08 0.05 0.02];
    labels = ones(size(data,1),1);
    centers = zeros(k,3);
    centers(1,:) = mean(data,1);
    X = data(:,1);
    Y = data(:,2);
    Z = data(:,3);
%nub-split
    for num_class = 1:k-1
        dis = zeros(num_class,1);
        for label = 1:num_class 
            for num = 1:size(data,1)
                if labels(num) == label
                     dis(label) = dis(label) + sqrt(sum((data(num,:)-centers(label,:)).^2));
                end
            end
        end
        
        [max_dis,max_dis_index] = max(dis);
        for num = 1:size(data,1)
            if labels(num) == max_dis_index  
                d1 = sqrt(sum((data(num,:)-(centers(max_dis_index,:)+v)).^2));
                d2 = sqrt(sum((data(num,:)-(centers(max_dis_index,:)-v)).^2));
                if d1 < d2
                    labels(num) = max_dis_index;
                else
                    labels(num) = num_class+1;
                end
            end
        end
        centers(max_dis_index,:) = mean(data(labels == max_dis_index,:));
        centers(num_class+1,:) = mean(data(labels == num_class+1,:));
    end
end
end

        

        
        
                    
    

