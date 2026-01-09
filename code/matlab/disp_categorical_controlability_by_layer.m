function figure_counter = disp_categorical_controlability_by_layer( ...
    r_square_by_param_by_layer, ...
    gain_by_param_by_layer, ...
    models, ...
    name_layer, ...
    params_to_control, ...
    figure_counter, ...
    model_type, ...
    array_best_layer_indexes, ...
    max_pred_by_layer, ...
    min_pred_by_layer, ...
    max_target_by_layer, ...
    min_target_by_layer, ...
    ref_proportion_of_silences_by_model ...
    )

% Avoid negative R² values
gain_by_param_by_layer = max(gain_by_param_by_layer, 0);
r_square_by_param_by_layer = max(r_square_by_param_by_layer, 0);

% if strcmp(model_type, 'fastspeech2')
%     r_square_by_param_by_layer([1, 8], 3:5, :) = nan(2, 3, size(r_square_by_param_by_layer, 3));
% end

% Hyperparameters graphs
% legend_by_param = {
%     '-', [193, 214, 167]/255;
%     '-', [127, 218, 244]/255;
%     '--', [253, 185, 36]/255;
%     '--', [134, 156, 152]/255;
%     '--', [131, 78, 86]/255;
%     '-', [255, 248, 91]/255;
%     '-', [255, 0, 76]/255;
%     ':', [0, 154, 205]/255;
%     '-.', [0, 250, 154]/255;
%     '-.', [123, 104, 238]/255;
%     '-.', [255, 110, 180]/255;
% };
legend_by_param = {
    '-', [0, 139, 139]/255;
    '-', [127, 218, 244]/255;
    '-', [253, 185, 36]/255;
    '-', [134, 156, 152]/255;
    '-', [131, 78, 86]/255;
    '-', [255, 248, 91]/255;
    '-', [255, 0, 76]/255;
    '-', [0, 154, 205]/255;
    '-', [0, 250, 154]/255;
    '-', [123, 104, 238]/255;
    '-', [255, 110, 180]/255;
};

% color_table = [
%    
%    ;
%    ;
%    ;
%    ;
%    ;
%    ;
%    ;
%    ;
%    ;
%    ;
%    ;
%    153, 50, 204;
%    188, 238, 104;
%    189, 183, 107;
%    0, 139, 139;
%    102, 205, 0;
%    139, 35, 35;
% ]/255;

nbr_params = size(r_square_by_param_by_layer, 1);
nbr_layers = size(r_square_by_param_by_layer, 2);
nbr_models = size(r_square_by_param_by_layer, 3);

list_params_supra_segmental = [1];
% list_params_supra_segmental = [1, 2, 6, 7, 8];
% list_params_segmental = [3, 4, 5, 9, 10];
% list_params_supra_segmental = [];
% list_params_segmental = 1;

% Separate encoder and decoder
if strcmp(model_type, 'tacotron2')
    limit_encoder = 3.5;
    x_text_encoder = 2;
    x_text_decoder = 3.6;
elseif strcmp(model_type, 'fastspeech2')
    limit_encoder = 2.5;
    x_text_encoder = 1.5;
    x_text_decoder = 3.5;
end
% ylimits = [0, 1.5];
ylimits = [-1, 23];
ylimits_targets = [0, 20];
xlimits = [1, nbr_layers+0.4];
y_text = ylimits(2)-2;

figure(figure_counter);
clf;
p = panel();
set(groot, "defaultAxesTickLabelInterpreter", 'latex');
set(groot, "defaultTextInterpreter", 'latex');
set(groot, "defaultLegendInterpreter", 'latex');
p.margin = [13 15 5 5];
p.pack(1, nbr_models);

for i_model = 1:nbr_models
    name_model = models{i_model,1};
    name_model_legend = models{i_model, 6};

    % Disp R² | supra-segmental
    p(1, i_model).select();
    for i_param = list_params_supra_segmental
        index_param = cell2mat(params_to_control(i_param, 2));


%         plot(1:nbr_layers, r_square_by_param_by_layer(i_param, :, i_model), 'LineStyle', legend_by_param{index_param, 1}, 'LineWidth', 1.5, 'Color', legend_by_param{index_param, 2});
%         plot(1:nbr_layers, gain_by_param_by_layer(i_param, :, i_model), 'LineStyle', legend_by_param{index_param, 1}, 'LineWidth', 1.5, 'Color', legend_by_param{index_param, 2});

        plot(xlimits, ref_proportion_of_silences_by_model(i_model)*ones(1, 2), 'k-.', 'LineWidth', 1);
        hold on;
        plot(1:nbr_layers, max_pred_by_layer(i_param, :, i_model), 'LineStyle', '--', 'LineWidth', 1.5, 'Color', legend_by_param{index_param, 2});
        plot(1:nbr_layers, min_pred_by_layer(i_param, :, i_model), 'LineStyle', ':', 'LineWidth', 1.5, 'Color', legend_by_param{index_param, 2});
        plot(xlimits, ylimits_targets(1)*ones(1, 2), 'r:', 'LineWidth', 1.5,'HandleVisibility','off');
        plot(xlimits, ylimits_targets(2)*ones(1, 2), 'r--', 'LineWidth', 1.5,'HandleVisibility','off');

%         scatter_points = scatter(1:nbr_layers, r_square_by_param_by_layer(i_param, :, i_model), 'o');
%         scatter_points = scatter(1:nbr_layers, gain_by_param_by_layer(i_param, :, i_model), 'o');
        scatter_points1 = scatter(1:nbr_layers, max_pred_by_layer(i_param, :, i_model), 'o');
        scatter_points2 = scatter(1:nbr_layers, min_pred_by_layer(i_param, :, i_model), 'o');

        scatter_points1.MarkerFaceColor = legend_by_param{index_param, 2};
        scatter_points1.MarkerEdgeColor = legend_by_param{index_param, 2};
        scatter_points1.SizeData = max(100*r_square_by_param_by_layer(i_param, :, i_model), 10^-6);
        scatter_points1.Annotation.LegendInformation.IconDisplayStyle = "off";

        scatter_points2.MarkerFaceColor = legend_by_param{index_param, 2};
        scatter_points2.MarkerEdgeColor = legend_by_param{index_param, 2};
        scatter_points2.SizeData = max(100*r_square_by_param_by_layer(i_param, :, i_model), 10^-6);
        scatter_points2.Annotation.LegendInformation.IconDisplayStyle = "off";
    end
%     plot(1:nbr_layers, zeros(1,nbr_layers), 'k--', 'LineWidth', 3);
%     plot(xlimits, ones(1,2), 'r-.', 'LineWidth', 1,'HandleVisibility','off');
%     plot(limit_encoder*ones(1, 2), ylimits, 'k--', 'LineWidth', 2,'HandleVisibility','off');
    text(x_text_encoder, y_text, 'Encoder');
%     text(x_text_decoder, y_text, 'Decoder');
    xticks(1:nbr_layers);
%     if i_model > 1
%         yticklabels([]);
%     end
    if i_model == 1
        ylabel('Control range (% of pauses)');
    end
    xlabel('\overrightarrow{Layers~Depth}', 'Interpreter','latex');
    xticklabels(name_layer);
%     xticklabels([]);
    ylim(ylimits);
    xlim(xlimits);
    if i_model == nbr_models
        % Add legend for point size
        size_point_legend = [0.1, 0.5, 1];
        for i_point_size_legend = 1:3
            scatter_points_legend = plot(i_point_size_legend, ylimits(2)+1, 'o', 'HandleVisibility','on', 'MarkerSize', sqrt(100*size_point_legend(i_point_size_legend)), 'MarkerEdgeColor','black', 'MarkerFaceColor','black');
        end

        legend({'unbiased'; 'max achievable'; 'min achievable'; 'R²: 0.1'; 'R²: 0.5'; 'R²: 1'}, 'Location','southeast');
    end
    grid on;
%     title(sprintf('Gain by layer (bias effect) | Model: %s | Supra-Segmental', name_model));
%     title(sprintf('Model: %s | Categorical Control', name_model));
    title(sprintf('%s', name_model_legend));

%     % Disp R² | segmental
%     p(2, i_model).select();
%     for i_param = list_params_segmental
%         index_param = cell2mat(params_to_control(i_param, 2));
% %         plot(1:nbr_layers, r_square_by_param_by_layer(i_param, :, i_model), 'LineStyle', legend_by_param{index_param, 1}, 'LineWidth', 1.5, 'Color', legend_by_param{index_param, 2});
%         plot(1:nbr_layers, gain_by_param_by_layer(i_param, :, i_model), 'LineStyle', legend_by_param{index_param, 1}, 'LineWidth', 1.5, 'Color', legend_by_param{index_param, 2});
%         hold on;
% %         scatter_points = scatter(1:nbr_layers, r_square_by_param_by_layer(i_param, :, i_model), 'o');
%         scatter_points = scatter(1:nbr_layers, gain_by_param_by_layer(i_param, :, i_model), 'o');
%         scatter_points.MarkerFaceColor = legend_by_param{index_param, 2};
%         scatter_points.MarkerEdgeColor = legend_by_param{index_param, 2};
% %         scatter_points.SizeData = max(100*gain_by_param_by_layer(i_param, :, i_model), 10^-6);
%         scatter_points.SizeData = max(100*r_square_by_param_by_layer(i_param, :, i_model), 10^-6);
%         scatter_points.Annotation.LegendInformation.IconDisplayStyle = "off";
% %         scatter_points.MarkerFaceColor=legend_by_param{index_param, 2};
%     end
% %     plot(1:nbr_layers, zeros(1,nbr_layers), 'k--', 'LineWidth', 3);
%     plot(xlimits, ones(1,2), 'r-.', 'LineWidth', 1,'HandleVisibility','off');
%     plot(limit_encoder*ones(1, 2), ylimits, 'k--', 'LineWidth', 2,'HandleVisibility','off');
%     text(x_text_encoder, y_text, 'Encoder');
%     text(x_text_decoder, y_text, 'Decoder');
%     xticks(1:nbr_layers);
%     if i_model > 1
%         yticklabels([]);
%     end
%     xticklabels(name_layer);
%     ylim(ylimits);
%     xlim(xlimits);
%     if i_model == nbr_models
%         % Add legend for point size
%         size_point_legend = [0.1, 0.5, 1];
%         for i_point_size_legend = 1:3
%             scatter_points_legend = plot(i_point_size_legend, ylimits(2)+1, 'o', 'HandleVisibility','on', 'MarkerSize', sqrt(100*size_point_legend(i_point_size_legend)), 'MarkerEdgeColor','black', 'MarkerFaceColor','black');
%         end
% 
%         legend([params_to_control(list_params_segmental,1); {'R²: 0.1'; 'R²: 0.5'; 'R²: 1'}], 'Location','southeast');
%     end
%     grid on;
% %     title(sprintf('Gain by layer (bias effect) | Model: %s | Segmental', name_model));
%     title(sprintf('Model: %s | Segmental', name_model));


%     % Disp saturation
%     subplot_counter= subplot_counter + 1;
%     subplot(nbr_models, 3, subplot_counter);
%     for i_param = 1:nbr_params
%         index_param = cell2mat(params_to_control(i_param, 2));
%         plot(1:nbr_layers, saturation_by_param_by_layer(i_param, :, i_model), legend_by_param{index_param});
%         hold on;
%     end
%     plot(1:nbr_layers, zeros(1,nbr_layers), 'k--', 'LineWidth', 3);
%     plot(1:nbr_layers, 3*ones(1,nbr_layers), 'r--', 'LineWidth', 3);
% %     ylim([0, 8]);
%     xticks(1:nbr_layers);
%     xticklabels(name_layer);
%     legend(params_to_control(:,1), 'Location','northwest');
%     grid on;
%     title(sprintf('Saturation by layer | Model: %s', name_model));

%     % Disp Gain around 0
%     subplot_counter = subplot_counter + 1;
%     subplot(nbr_models, 2, subplot_counter);
%     for i_param = 1:nbr_params
%         index_param = cell2mat(params_to_control(i_param, 2));
%         plot(1:nbr_layers, gain_by_param_by_layer(i_param, :, i_model), legend_by_param{index_param}, 'LineWidth', 2);
%         hold on;
%     end
%     plot(1:nbr_layers, zeros(1,nbr_layers), 'k--', 'LineWidth', 3);
%     plot(1:nbr_layers, ones(1,nbr_layers), 'r--', 'LineWidth', 3);
%     xticks(1:nbr_layers);
%     xticklabels(name_layer);
%     ylim([0, 1.3]);
%     legend(params_to_control(:,1), 'Location','northwest');
%     grid on;
%     title(sprintf('Gain by layer | Model: %s', name_model));

end

% Plot best layer to control each param
dilatation_coef = 0.15;
y_coord_stars = 22;
for i_model = 1:nbr_models
    for i_layer = 1:nbr_layers
        best_control_current_layer = array_best_layer_indexes(:, i_model) == i_layer;
%         nbr_best_control_current_layer_segmental = sum(best_control_current_layer(list_params_segmental));
        nbr_best_control_current_layer_supra_segmental = sum(best_control_current_layer(list_params_supra_segmental));
%         coord_stars_segmental = (i_layer - dilatation_coef*(nbr_best_control_current_layer_segmental-1)/2):dilatation_coef:(i_layer + dilatation_coef*(nbr_best_control_current_layer_segmental-1)/2);
        coord_stars_supra_segmental = (i_layer - dilatation_coef*(nbr_best_control_current_layer_supra_segmental-1)/2):dilatation_coef:(i_layer + dilatation_coef*(nbr_best_control_current_layer_supra_segmental-1)/2);

        index_param_segmental = 0;
        index_param_supra_segmental = 0;
        for i_param = 1:nbr_params
            index_param = cell2mat(params_to_control(i_param, 2));
            if best_control_current_layer(i_param)
%                 if ismember(i_param, list_params_supra_segmental)
                    p(1, i_model).select();
                    index_param_supra_segmental = index_param_supra_segmental + 1;
                    plot(coord_stars_supra_segmental(index_param_supra_segmental), y_coord_stars, '*', 'Color', legend_by_param{index_param, 2}, 'MarkerSize',10,'LineWidth',2,'HandleVisibility','off')
%                 else
%                     p(2, i_model).select();
%                     index_param_segmental = index_param_segmental + 1;
%                     plot(coord_stars_segmental(index_param_segmental), y_coord_stars, '*', 'Color', legend_by_param{index_param, 2}, 'MarkerSize',10,'LineWidth',2,'HandleVisibility','off')
%                 end

                
            end
        end
    end
end
figure_counter = figure_counter + 1;
end