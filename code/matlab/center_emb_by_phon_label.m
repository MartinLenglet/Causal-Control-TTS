function centered_emb_matrix = center_emb_by_phon_label(emb_matrix, labels, labels_index_in_values)
    nbr_labels = length(labels(:,1));
    
    nbr_emb = size(emb_matrix, 1);
    nbr_dim = size(emb_matrix, 2);
    
    centered_emb_matrix = zeros(nbr_emb, nbr_dim);
    
    for i_label = 1:nbr_labels
        current_label = labels{i_label, 1};
        list_current_label = labels{i_label, 2};
        length_list = length(list_current_label);
        
        list_values = zeros(length_list, nbr_dim);
        index_to_upload = [];
        index_to_delete = [];
        for i = 1:length_list
            if ~isempty(find(labels_index_in_values == list_current_label(i)))
                index_in_values = find(labels_index_in_values == list_current_label(i));
                index_to_upload = [index_to_upload, index_in_values];
                list_values(i, :) = emb_matrix(index_in_values, :);
            else
                index_to_delete = [index_to_delete, i];
            end
        end
        
        list_values(index_to_delete, :) = [];
        mean_label = mean(list_values, 1);
        std_label = std(list_values, 0, 1);
        
        list_values = list_values - mean_label;
%         for i_dim = 1:nbr_dim
%             list_values(:, i_dim) = (list_values(:, i_dim) - mean_label(i_dim))/std_label(i_dim);
%         end
        
        centered_emb_matrix(index_to_upload, :) = list_values;
    end
end