%% compute_coef_lda.m
% Calculate Coordonates of embeddings with LDA.
%
% LDA space is calculated with phonetic embeddings, and character
% embeddings are projected in this space
%
function [coef_lda, lambda, W] = compute_coef_lda(all_char_index, all_emb_data, nbr_embeddings, models, selected_dim)    
    % Global parameters
    nbr_models = length(models(:,1));
    start_index = 1;
    start_index_phonem = 1;
    all_indexes = [];
    targets = [];
    
    % Standardisation and Normalisation of embeddings
    emb_size = size(all_emb_data,1);
    for i_emb = 1:emb_size
        % Standardisation
        % all_emb_data(i_emb,:) = (all_emb_data(i_emb,:) - mean(all_emb_data(i_emb,:)))/std(all_emb_data(i_emb,:));
        % Normalisation
        % all_emb_data(i_emb,:) = (all_emb_data(i_emb,:) - min(all_emb_data(i_emb,:)))/(max(all_emb_data(i_emb,:))-min(all_emb_data(i_emb,:)));
    end
    
    % Calculate LDA coef
    for i_model = 1:nbr_models       
        % Calculate LDA coef with phonetic only
        if (~models{i_model, 3})
            char_index_from_model = all_char_index{1,i_model};
            nbr_unique_char = length(char_index_from_model);
            for i_char = 1:nbr_unique_char
                all_indexes_by_model = char_index_from_model{i_char,2} + start_index - 1;
                all_indexes = [all_indexes; all_indexes_by_model'];
                targets = [targets; (repmat(i_char,length(all_indexes_by_model),1) + start_index_phonem -1)];
            end     
            start_index_phonem = start_index_phonem + nbr_unique_char;
        end
        
        start_index = start_index + nbr_embeddings(i_model);
    end

    % Compute LDA
    inputs = all_emb_data(:,all_indexes)';
    [~, W, lambda] = LDA(inputs,targets);
    coef_lda = all_emb_data' * W(:,selected_dim);
    lambda = real(lambda);
    % lambda = lambda(selected_dim);
end