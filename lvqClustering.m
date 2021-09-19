function [out1] = lvqClustering(input, k, c, N_iteration)

% clustering using competlayer
net = competlayer(k);
net.trainParam.epochs = N_iteration;

% train network
[net, tr] = train(net, input);

% get output
classes = vec2ind(net(input));
out1 = classes;

end

