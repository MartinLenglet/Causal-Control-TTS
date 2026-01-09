function zscore_vector = absolute_param_to_zscore(values, labels, labels_index_in_values, i_param, type)
    nbr_labels = length(labels(:,1));
    zscore_vector = values;
    
    for i_label = 1:nbr_labels
        current_label = labels{i_label, 1};
        list_current_label = labels{i_label, 2};
        length_list = length(list_current_label);
        
        list_values = zeros(1, length_list);
        index_to_upload = [];
        index_to_delete = [];
        for i = 1:length_list
            if ~isempty(find(labels_index_in_values == list_current_label(i)))
                index_in_values = find(labels_index_in_values == list_current_label(i));
                index_to_upload = [index_to_upload, index_in_values];
                list_values(i) = values(index_in_values);
            else
                index_to_delete = [index_to_delete, i];
            end
        end
        
        list_values(index_to_delete) = [];
        mean_param = nanmean(list_values);
        std_param = nanstd(list_values);
%         if i_param==11
%             fprintf('Char %s | mean: %0.2f | std: %0.2f\n', current_label, mean_param, std_param); 
%         end
%         std_param = 1;
        

%         fprintf('Mean after center: %0.2f\n', nanmean(list_values));

        if strcmp(type, 'delta')
            list_values = (list_values-mean_param);
        elseif strcmp(type, 'mean')
            list_values = mean_param * ones(1, length(list_values));
        elseif strcmp(type, 'zscore')
            list_values = (list_values-mean_param)/std_param;
        end
        
        zscore_vector(index_to_upload) = list_values;
    end
end