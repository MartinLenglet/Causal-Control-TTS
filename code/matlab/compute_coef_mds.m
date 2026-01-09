%% compute_coef_mds.m
% Calculate Coordonates of embeddings with MDS.
%
% If reduce_size = true, only distances superior to the threshold are taken
% into account to build the MDS space (reduce computation if used on a
% large number of embeddings).
%
%
function [lambda, W, nbr_emb, X] = compute_coef_mds(all_emb_data, reduce_size, select_best_dim_number)    
    % Global parameters
    nbr_emb = size(all_emb_data, 2);
    if reduce_size
        dist_threshold = 0.25;
%         dist_threshold = 0.1;
        
        taken_index = 1;
        for i_emb = 2:nbr_emb
            fprintf('Reduce size matrix MDS: %d/%d\n', i_emb, nbr_emb);
            emb_1 = all_emb_data(:,i_emb)';
            take_this_index = true;
            for i_taken = 1:length(taken_index)
                emb_2 = all_emb_data(:,i_taken);
                d = 1 - emb_1*emb_2/(norm(emb_1)*norm(emb_2));
%                 d = norm(emb_1 - emb_2');
                if (d<dist_threshold)
                    take_this_index = false;
                    break;
                end
            end
            if take_this_index
                taken_index = [taken_index, i_emb];
            end
        end
        
        all_emb_data = all_emb_data(:,taken_index);
        nbr_emb = size(all_emb_data, 2);
    end
    dist = zeros(nbr_emb);
    
    for i_emb_1 = 1:nbr_emb-1
        fprintf('Distance matrix MDS: %d/%d\n', i_emb_1, nbr_emb-1);
        emb_1 = all_emb_data(:,i_emb_1)';
        for i_emb_2 = i_emb_1+1:nbr_emb
            emb_2 = all_emb_data(:,i_emb_2);
            % Compute vector distance
            d = 1 - emb_1*emb_2/(norm(emb_1)*norm(emb_2));
            % d = norm(emb_1-emb_2);
            dist(i_emb_1, i_emb_2) = max(d,0);
        end 
    end
    
    % MDS
    fprintf('Start calulation MDS\n');
    [coef_mds,lambda] = cmdscale(dist+dist');
    
%     select_best_dim_number = true;
    
    if select_best_dim_number
        best_dim_number = find_best_number_of_dim(lambda);
        coef_mds = coef_mds(:,1:best_dim_number);
        lambda = lambda(1:best_dim_number);
    else
        lambda = lambda(1:size(coef_mds,2));
    end
    
    fprintf('End calulation MDS\n');
    W = all_emb_data'\coef_mds;
    X = coef_mds\(all_emb_data');
%     Z = W \ eye(512);
%     Z =  (all_emb_data'*W) \ all_emb_data';
end