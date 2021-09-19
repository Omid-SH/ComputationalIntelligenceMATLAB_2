function [out1,out2] = kMeans(input, k, c, N_iteration)

if( k ~= size(c,2) )
    fprintf("Error! init centers and cluster numbers mismatch!")
    return;
end
   
if( size(input,1) ~= size(c,1) )
    fprintf("Error! features mismatch")
    return;
end

f = size(input,1); % features size
n = size(input,2); % data size
dist = zeros(n, k); % distence of each cluster and data
CNs = zeros(1,n); % the nearest cluster

for iteration = 1:N_iteration

    for i = 1:n
        for j = 1:k
                % calculate distance for each data and center
                dist(i,j)= norm(input(:,i)-c(:,j));
        end
        
        % Define clusters
        [~, CN] = min(dist(i,:)); 
        % minimum distance and the cluster which the sample belongs to
        CNs(i) = CN;
    end
    
    % update clusters centers
    for i = 1:k
        PC = (CNs==i);
        c(:,i) = mean(input(:,PC), 2);
    end
    
    out1 = CNs;
    out2 = c;
end

