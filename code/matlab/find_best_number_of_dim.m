function best_dim_number = find_best_number_of_dim(lambda)
    normalized_lambda = lambda(lambda>0)/sum(lambda(lambda>0));
    
    target_variance = 0.9; % for scree-plot method
    choice_dim = 30; % for choice method
    
    method = 'scree-plot';
    if strcmp(method, 'scree-plot')
        best_dim_number = 1;
        while sum(normalized_lambda(1:best_dim_number)) < target_variance
            best_dim_number = best_dim_number + 1;
        end
    elseif strcmp(method, 'scree-plot-gap')
        intra_pair_diff = normalized_lambda(1:end-1) - normalized_lambda(2:end);
        extra_pair_diff = abs(intra_pair_diff(1:end-1) - intra_pair_diff(2:end));
        [~, best_dim_number] = max(extra_pair_diff);
        best_dim_number = best_dim_number + 1;
    elseif strcmp(method, 'choice')
        best_dim_number = choice_dim;
    elseif strcmp(method, 'kaiser-guttman')
        best_dim_number = sum(lambda>1);
    elseif strcmp(method, 'broken-stick')
        best_dim_number = 1;
        bk = 0;
        for i = 1:length(normalized_lambda)
            bk = bk + (1/i)/length(normalized_lambda);
        end
        while normalized_lambda(best_dim_number) > bk
            bk = bk - (1/best_dim_number)/length(normalized_lambda);
            best_dim_number = best_dim_number + 1;
        end
    end
end