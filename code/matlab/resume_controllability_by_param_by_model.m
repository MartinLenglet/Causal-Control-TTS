function [resume_controllability, best_layer_indexes_array] = resume_controllability_by_param_by_model( ...
    r_square_by_param_by_layer, ...
    gain_by_param_by_layer, ...
    saturation_by_param_by_layer, ...
    list_models, ...
    name_layers, ...
    list_params, ...
    max_by_param_by_layer ...
    )

nbr_models = length(list_models(:,1));
nbr_params = length(list_params);
% nbr_layers = length(name_layers);

% Init results
resume_controllability = cell(nbr_models+1, nbr_params+1);
resume_controllability(2:end, 1) = list_models(:,1);
resume_controllability(1, 2:end) = list_params;
best_layer_indexes_array = zeros(nbr_params, nbr_models);

% Results are model dependant
for i_model = 1 :nbr_models
    % Find best layer to control (higher R²)
    for i_param = 1:nbr_params

        % Best layer metric
%         [~, index_best_layer] = max(r_square_by_param_by_layer(i_param, :, i_model));
%         [~, index_best_layer] = min(abs(gain_by_param_by_layer(i_param, :, i_model)-1));
        [~, index_best_layer] = min(abs(max_by_param_by_layer(i_param, :, i_model)-3));

        max_layer_name = name_layers{index_best_layer};
        best_layer_gain = gain_by_param_by_layer(i_param, index_best_layer, i_model);
        best_layer_saturation = saturation_by_param_by_layer(i_param, index_best_layer, i_model);
        best_r_square = r_square_by_param_by_layer(i_param, index_best_layer, i_model);

%         contrallability_metric = r_square_by_param_by_layer(i_param, :, i_model) .* gaussmf(gain_by_param_by_layer(i_param, :, i_model),[0.3 1]);

        % Update results cellarray
        best_layer_indexes_array(i_param, i_model) = index_best_layer;
        resume_controllability{i_model+1, i_param+1} = sprintf('%s(%0.2f)|G:%0.2f', max_layer_name, best_r_square, best_layer_gain);
    end
end

end