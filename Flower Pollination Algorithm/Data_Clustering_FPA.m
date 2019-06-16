%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notes: This demo program contains the basic Flower Pollination        %
% algorithm for clustering problem without fine-tuning the parameters.  %
% Though this demo works very well, it is expected that this            %
% illustration with randomly generated synthetic dataset is much less   %
% efficient than the work reported in the paper.                        %
% (Citation details):                                                   %
% J.Senthilnath, Sushant Kulkarni, S.Suresh, X.S.Yang, J.A.Benediktsson %
% (2019) "FPA Clust: Evaluation of Flower Pollination Algorithm for     %
% Data Clustering", Evolutionary Intelligence,                          %
% DOI: 10.1007/s12065-019-00254-1                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [CEP] = Data_Clustering_FPA ()
% two classes of data generated with mean of 10 and 20, standard deviation
% of 2
clear all; clc; close all
c1mean = 10;  c2mean = 15;
cstd = 2;
c1a1 = cstd.*randn(100,1)+c1mean;
c1a2 = cstd.*randn(100,1)+c1mean;
c2a1 = cstd.*randn(100,1)+c2mean;
c2a2 = cstd.*randn(100,1)+c2mean;
% data consisting of two atrributes for 100 instances of each class along with
% appended class labels
xdata = [c1a1 c1a2 ones(100,1);
    c2a1 c2a2 2*ones(100,1)];
temp = [75 75];                      % percentage of data for training in each class
[q1,q2] =size(xdata);
train_data =  xdata(:,1:(q2-1));     % dataset excluding the class labels
train_labels = xdata(:,q2);          % extract class labels
% extract unique class labels and its initial index in the dataset
[uniq_labels,uniq_idx] = unique(train_labels);
noofcl = length(uniq_labels);        % no of unique classes
uniq_idx(noofcl+1) = q1+1;           % location of the final cell
ftrain =[]; ftest = [];
for i =1:noofcl
    % extract data corresponding to each cluster/class using index of
    % class labels
    a1 = xdata(uniq_idx(i):(uniq_idx(i+1)-1),:);
    % randomly allocate certain portion for training and testing
    [train1,test1] = shuffel(a1,temp(i));
    % append training & testing data each time for every class
    ftrain = [ftrain ; train1];
    ftest = [ftest; test1];
end
%% plotting data for visualization
figure(1)
plot(ftrain(1:75,1),ftrain(1:75,2),'rd',ftrain(76:150,1),ftrain(76:150,2),'bd','LineWidth',1.5); hold on
plot(ftest(1:25,1),ftest(1:25,2),'co',ftest(26:50,1),ftest(26:50,2),'mo','LineWidth',1.5); hold on;
xlabel('X'); ylabel('Y'); xlim([0 25]); ylim([0 25]); grid on
legend('Class 1 Training','Class 2 Training','Class 1 Testing','Class 2 Testing','Location','NorthEastOutside')
title('Flower Pollination  based clustering algorithm','FontName','Times','Fontsize',12)
%% extracting cluster centers from training data
[q1,q2] =size(ftrain);
train_data =  ftrain(:,1:(q2-1));          % separate from class labels
train_labels = ftrain(:,q2);                 % extract class labels
% extract unique class labels and its initial index in training dataset
[uniq_labels,uniq_idx] = unique(train_labels);
noofcl = length(uniq_labels);            % no. of classes
noofattr = q2-1;                               % no. of attributes (p)
uniq_idx(noofcl+1) = q1;                 % storing index of last row of data
opt_ctr = zeros(noofcl,(q2-1));           % clusters = class x no of attributes
% find the optimal cluster center for each class using Flower Pollination
for i=1:noofcl
    % extract maximum & minimum from each of the "p" attributes
    fprintf('Optimal cluster: %d\n',i);
    clusmaxmin(1,:) = max(train_data(uniq_idx(i):(uniq_idx(i+1)-1),:));
    clusmaxmin(2,:) = min(train_data(uniq_idx(i):(uniq_idx(i+1)-1),:));
    % extract optimal cluster centers for each class by minimzing cost
    % function using Flower Pollination Algorithm. Pass the training
    % dataset, upper-lower limits of the dataset and number of attributes
    % to FPA and compute optimal cluster center
    opt_ctr(i,:) = Flower_Pollination(train_data(uniq_idx(i):(uniq_idx(i+1)-1),:),clusmaxmin, noofattr);
end
%% Cluster the data into respective classes using optimal cluster centers
test_data = ftest;
[q1,q2] =size(test_data);                % q1 indicates total no. of attributes
testdata  =  test_data(:,1:(q2-1));      % separate from class labels
testlabels  = test_data(:,q2);           % extract class labels
z = zeros(1,noofcl);
idx = zeros(1,q1);
% assign each attribute to the nearest cluster center
for i = 1:q1
    % compute distance of attribute from all cluster centers
    for j = 1:noofcl
        z(j) = pdist2(testdata(i,:),opt_ctr(j,:));
    end
    least = min(z);             % find the least distance among all
    idx(i) = find(z == least);  % assign it to that corresponding class/cluster
end
%% Display the results
% generate confusion matrix in comparison with actual class labels
display('The optimal centers are ')
disp(opt_ctr)
CM = confusionmat(testlabels,idx);
display('The confusion matrix is ')
disp(CM)
% overall accuracy
total = trace(CM);
% display('The overall accuracy is ')
efficiency = (total/q1)*100;
CEP = 100 - efficiency;
fprintf('The classification error percentage is: %d\n',CEP);
function [xtrain, xtest] = shuffel(matrix, ratio)
number_elements = round((ratio/100) * size(matrix,1));  % training ratio
row_selection = randperm(size(matrix,1));
xtrain = matrix(row_selection(1:number_elements),:);
xtest = matrix(row_selection(number_elements+1:size(matrix,1)),:);
end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%