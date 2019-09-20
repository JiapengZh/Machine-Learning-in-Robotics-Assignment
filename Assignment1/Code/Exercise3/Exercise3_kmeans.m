function Exercise3_kmeans(gesture_l,init_cluster_l,gesture_o,init_cluster_o,gesture_x,init_cluster_x,num_clusters)

if nargin < 7
    num_clusters = 7;
end

[X_l,Y_l,Z_l,clusters_l,Color_l] = clustering(gesture_l,init_cluster_l,num_clusters);
[X_o,Y_o,Z_o,clusters_o,Color_o] = clustering(gesture_o,init_cluster_o,num_clusters);
[X_x,Y_x,Z_x,clusters_x,Color_x] = clustering(gesture_x,init_cluster_x,num_clusters);
%plot
subplot(2,10,[1,4]);
scatter3(X_l,Y_l,Z_l,50,Color_l);
title('l-gesture');

subplot(2,10,[7,10]);          
scatter3(X_o,Y_o,Z_o,50,Color_o);
title('o-gesture');

subplot(2,10,[14,17]); 
scatter3(X_x,Y_x,Z_x,50,Color_x);
title('x-gesture');

%function which oprates k-means algorithm
function [X,Y,Z,clusters,Color] = clustering(gesture,init_cluster,num_clusters)
    %initialization 
    [m,n,d] = size(gesture);
    %[a,b] = size(init_cluster);
    a = num_clusters;
    l = zeros(m*n,d);
    for i = 1:m
        for j = 1:n
            l(10*(i-1)+j,:) = gesture(i,j,:);
        end
    end
    X = zeros(m*n,1);
    Y = zeros(m*n,1);
    Z = zeros(m*n,1);
    Color = zeros(m*n,3);
    center = init_cluster; 
    clusters = zeros(m*n,1);
    distortion = 0;
    stop = 0;

    %K-means cluster
    while stop == 0
        distortion_pre = distortion;
        distortion = 0;
        for i = 1:m*n
            dis = zeros(a,1);
            for k = 1:a
                dis(k) = norm(l(i,:)-center(k,:));
            end
            [~,cluster_index] = min(dis);
            clusters(i) = cluster_index;
        end
        center = zeros(a,3);
        for k = 1:a
            center(k,:) = mean(l(clusters==k,:));
        end
        for k = 1:a
            distortion = distortion + norm(l(clusters==k,:)-center(k,:));
        end
        if norm(distortion - distortion_pre) <= 1*10^(-6)
                stop = 1;
        end

    %calculate the output
    C = zeros(7,3);
    C(1,:) = [0 0 1];
    C(2,:) = [0 0 0];
    C(3,:) = [1 0 0];
    C(4,:) = [0 1 0];
    C(5,:) = [1 0 1];
    C(6,:) = [1 1 0];
    C(7,:) = [0 1 1];
    for k = 1:a
            Color(clusters==k,:) = repmat(C(k,:),length(find(clusters==k)),1);
    end
    X = l(:,1);
    Y = l(:,2);
    Z = l(:,3);

end
end
end