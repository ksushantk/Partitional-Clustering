%%
%%-------------------------------------------------------------------------
% This is a simple demo version implementing the basic                   %
% Bat algorithm for clustering problem without fine-tuning the           %
% parameters. Though this demo works very well, it is expected           %  
% that this demo is much less efficient than the work reported in        %
% the following papers: (Citation details):                              % 
% J. Senthilnath, Sushant Kulkarni, J.A. Benediktsson and X.S. Yang      %
% (2016) "A novel approach for Multi-Spectral Satellite Image            %
% Classification based on the Bat Algorithm clustering problems",        %
% IEEE Letters for Geoscience and Remote Sensing Letters,                %
%  Vol. 13, No. 4, pp.599�603.                                           %
%%-------------------------------------------------------------------------
function [] = Main()
clear all; close all; clc
tic
% Two classes of data generated - mean [10,20], standard deviation - 2
clear all; clc; close all
c1mean = 10;  c2mean = 20; cstd = 2;
c1a1 = cstd.*randn(100,1)+c1mean;
c1a2 = cstd.*randn(100,1)+c1mean;
c2a1 = cstd.*randn(100,1)+c2mean;
c2a2 = cstd.*randn(100,1)+c2mean;
% data consisting of two atrributes for 100 instances of each class 
% along with appended class labels
xdata = [c1a1 c1a2 ones(100,1); 
             c2a1 c2a2 2*ones(100,1)];  
temp = [75 75];                            % percentage of data for training in each class
[q1,q2] =size(xdata);
train_data =  xdata(:,1:(q2-1));      % dataset excluding the class labels
train_labels = xdata(:,q2);             % extract class labels
% extract unique class labels and its initial index in the dataset
[uniq_labels,uniq_idx] = unique(train_labels); 
noofcl = length(uniq_labels);        % no of unique classes
uniq_idx(noofcl+1) = q1+1;         % location of the final cell
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
%% plotting data for visualization  'MarkerSize',10
plot(ftrain(1:75,1),ftrain(1:75,2),'rd',ftrain(76:150,1),ftrain(76:150,2),'bd','LineWidth',1.5); hold on
plot(ftest(1:25,1),ftest(1:25,2),'co',ftest(26:50,1),ftest(26:50,2),'mo','LineWidth',1.5); hold on;
xlabel('X'); ylabel('Y'); xlim([0 30]); ylim([0 35]); grid on
title('Data clustering using Bat Algorithm','FontName','Times','Fontsize',12)
%% ========================================%
% estimating the sizes of the datasets
[q1,q2] =size(ftrain);
train_data =  ftrain(:,1:(q2-1));              % separate from class labels
train_labels = ftrain(:,q2);                     % extract class labels
% extract unique class labels and its initial index in training dataset
[uniq_labels,uniq_index] = unique(train_labels);
noofcl = length(uniq_labels);                 % no. of classes
noofattr = q2-1;                                    % no. of attributes (p)
uniq_index(noofcl+1) = q1;                  % storing index of last row no.
opt_ctr = zeros(noofcl,(q2-1));             % clusters = class x no of attributes
%% find the optimal cluster center for each class using Bat Algorithm
for i =1:noofcl
    
    % Extract maximum & minimum from each of the "p" attributes
    clusmaxmin(1,:) = max(train_data(uniq_index(i):(uniq_index(i+1)-1),:)); 
    clusmaxmin(2,:) = min(train_data(uniq_index(i):(uniq_index(i+1)-1),:));
    
    % Extract optimal cluster centers for each class by minimzing cost
    % function using Bat Algorithm. 
    % Pass the training dataset along with its upper-lower limits and 
    % no. of attributes to Bat Algorithm for computing optimal cluster center
    opt_ctr(i,:) = Bat_Algorithm(train_data(uniq_index(i):(uniq_index(i+1)-1),:),clusmaxmin, noofattr);
   
end
%% Cluster the data into respective classes using optimal cluster centers
[q1,q2] =size(ftest);
test_data  =  ftest(:,1:(q2-1));              % separate from class labels
test_labels  = ftest(:,q2);                     % extract class labels
z = zeros(1,noofcl);                         % store results of distance from all the centers
idx = zeros(1,q1);
% assign each row/data to the nearest cluster center
for i = 1:q1
    % compute distance of attribute from all cluster centers
    for j = 1:noofcl
        z(j) = pdist2(test_data(i,:),opt_ctr(j,:));
    end
    least = min(z);                         % find the least distance among all 
    idx(i) = find(z == least);           % assign it to that corresponding cluster
end
%% Display the results
% generate confusion matrix in comparison with actual class labels
fprintf('\n The optimal centers are \n')
fprintf('%f %f \n', opt_ctr)
CM = confusionmat(test_labels,idx);
fprintf('\n The confusion matrix is \n')
fprintf('%d %d \n', CM)
% overall accuracy
total = trace(CM);
efficiency = (total/q1)*100;
fprintf('The Overall Accuracy is %.2f \n', efficiency);
toc; % measures computational time
function [xtrain, xtest] = shuffel(matrix, ratio)
    
    number_elements = round((ratio/100) * size(matrix,1));  % training ratio
    row_selection = randperm(size(matrix,1));
    xtrain = matrix(row_selection(1:number_elements),:);
    xtest = matrix(row_selection(number_elements+1:size(matrix,1)),:);    
    
end
end