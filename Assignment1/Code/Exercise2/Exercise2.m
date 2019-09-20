function [optimal_d,min_error,confusion_matrix] = Exercise2(dmax)
if nargin < 1
    dmax = 60;
end

%loading data
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
images_test = loadMNISTImages('t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');
error_rate = zeros(1,dmax); 
pre_labels_set = cell(dmax,1);

%PCA
images_ = images - repmat(mean(images,2),[1,size(images,2)]);
cov_matrix = cov(images');
[eigenvector, eigenvalue] = eig(cov_matrix);
[eigenvalue,index] = sort(diag(eigenvalue),'descend');
eigenvector = eigenvector(:,index);
images_test = images_test-repmat(mean(images,2),[1,size(images_test,2)]);

%ML classifier from d=1 to dmax
for d = 1:dmax
    pre_labels = zeros(size(labels_test));
    projected_images = eigenvector(:,1:d)'*images_;
    projected_test_images = eigenvector(:,1:d)'*images_test;
    tem_label = cell(10,1);
    for i = 0:9
        image = projected_images(:,labels == i);
        mu = mean(image,2);
        sigma = cov(image');
        tem_label{i+1} = mvnpdf(projected_test_images',mu',sigma);
    end
    
    for i = 1:size(labels_test,1)
        current = tem_label{1}(i);
        index = 0;
        for class = 2:10
            if tem_label{class}(i) > current
                index = class-1;
                current = tem_label{class}(i);
            end
        end
        pre_labels(i) = index;
    end
    pre_labels_set{d} = pre_labels;
    count = 0;
    for i = 1: size(labels_test)
        if pre_labels(i) ~= labels_test(i)
            count = count +1;
        end 
    end
    error_rate(d) = count / size(labels_test,1)*100;
end
%find the optimal d and its minnimum value of classification error
[~,optimal_d] = min(error_rate);
min_error = error_rate(optimal_d);
disp(['The optimal value of d is ' num2str(optimal_d)]);
disp(['The minimum classification error is ' num2str(min_error) '%']);

%plot of classification error (from d=1 to 60)
plot(1:dmax,error_rate(1:dmax),'b');
grid on;
xlabel('dimension (d)','fontsize',16);
set(gca,'XTick',[1:1:60]);
% text(15,error_rate(15),['      reference point (15,' num2str(error_rate(15)) ')'],'Color','k');
text(optimal_d,(min_error+2),['optimal point (' num2str(optimal_d) ',' num2str(min_error) ')'],'Color','r');
ylabel('test error rate (%)','fontsize',16);
title('Classification Error(from d=1 to d=60)','fontsize',18);
hold on;
%plot(15,error_rate(15),'ok');
plot(optimal_d,error_rate(optimal_d),'xr','Markersize',15);
legend('classification error (%)','optimal point');

%confusion matrix
disp(['The confusion matrix for d = ' num2str(optimal_d) ':']);
confusion_matrix = confusionmat(labels_test,pre_labels_set{optimal_d}); 
%figure;
%optimal_d_confusionmatrix = confusionchart(C,order,'Title',['Confusion Matrix (optimal d=' num2str(optimal_d) ')'],'RowSummary','row-normalized','ColumnSummary','column-normalized');
helperDisplayConfusionMatrix(confusion_matrix);
end


