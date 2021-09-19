%% Problem 1
clear; clc
load('SampleData1.mat')  

%% A)
scatter(TrainingData(1,:), TrainingData(2,:),[], TrainingLabels, 'filled')

%% B)
P = 0.8;
index = randperm(length(TrainingData));
index_train = index(1:round(P*length(TrainingData)));
index_test = index(round(P*length(TrainingData))+1:end);

X_train = TrainingData(:, index_train);
Y_train = TrainingLabels(:, index_train);

X_test = TrainingData(:, index_test);
Y_test = TrainingLabels(:, index_test);

%% C)
% we will search in for loop to find best parameters
Neurons = [1:30 50 100 200 500];

error_min = 10000;
best_param = [];

for neuron = Neurons
    net = feedforwardnet(neuron);

    % Setup Division of Data for Training, Validation, Testing. We have indexes
    % of each part from the last part so we use divideind as divideFcn
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = index_train;
    net.divideParam.valInd = index_test;
    net.divideParam.testInd = [];

    % Train the Network
    [net,tr] = train(net, TrainingData, TrainingLabels);
    
    % Find best parameters for model
    Y_o = net(X_test);
    if(norm(Y_o - Y_test) < error_min)
        error_min = norm(Y_o - Y_test);
        best_param = neuron;
    end

end

fprintf("Best hyper parameters of the MLP model is : \n Number of hidden neurons = %d \n", best_param);


% print the best net performence

net = feedforwardnet(best_param);

% Setup Division of Data for Training, Validation, Testing. We have indexes
% of each part from the last part so we use divideind as divideFcn
net.divideFcn = 'divideind';
net.divideParam.trainInd = index_train;
net.divideParam.valInd = index_test;
net.divideParam.testInd = [];

% Train the Network
[net,tr] = train(net, TrainingData, TrainingLabels);

%% D)
% we will search in for loop to find best parameters
Spreads = [0.001, 0.01, 0.1, 1, 10];
MNs = [1:30, 50, 80, 100, 200, 320];

error_min = 10000;
best_param = [];

for spread = Spreads
    for MN = MNs
        net = newrb(X_train, Y_train, 0, spread, MN);
        Y_o = sim(net, X_test);
        if(norm(Y_o - Y_test) < error_min)
            error_min = norm(Y_o - Y_test);
            best_param = [spread MN];
        end
    end
end

fprintf("Best hyper parameters of the RBF model is : \n Spread = %d \n MN = %d \nThe best RBF model norm2 error : %d \n", best_param(1), best_param(2), error_min);

%% Problem 2
clear; clc
load('SampleData2.mat')  

% A) function kMeans (input, k, c, N_iteration)
% B) lvqClustering(input, k, c, N_iteration)
% C)


%% Watch raw data
scatter(DataNew(1,:), DataNew(2,:), 'filled');
title("Raw data")

%% Kmeans for 5 clusters

% generate init points
%INIT_POINTS = [-20, -20, 5, 5, 20; -10 10 -20 20 0];
INIT_POINTS = DataNew(:, randperm(length(DataNew)));
INIT_POINTS = INIT_POINTS(:, 1:5);

[each_node_center, centers] = kMeans(DataNew, 5, INIT_POINTS, 10);

figure()
scatter(DataNew(1,:), DataNew(2,:), [], each_node_center, 'filled');
title("KMeans clustering output")

%% lvq for 5 clusters

each_node_center = lvqClustering(DataNew, 5, [], 30);

figure()
scatter(DataNew(1,:), DataNew(2,:), [], each_node_center, 'filled');
title("LVQ clustering output")



