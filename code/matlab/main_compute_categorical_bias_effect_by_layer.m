% Figure(10) in the code (in plot_categorical_controllability_by_layer) = Fig. 10 in the paper

clearvars
% model = 'fastspeech2';
model = 'tacotron2';

% =========== FASTSPEECH2 ===============
% --------- 1-9: ENCODER ---------------
% 1: raw char embeddings
% 2: raw char emb + positional_encoding
% 3-6: encoder layers
% 7: speaker embeddings
% 8: pitch embeddings
% 9: energy embeddings
% -------- 10-17: DECODER --------
% 10: upsampled encoder output
% 11: positional encoding
% 12-17: decoder layers
% -------- 18-19: MEL --------
% 18: mel before postnet
% 19: mel after postnet
name_layer_fastspeech = {
    'Phon Emb';
    'Pos Enc';
    'FFT1';
    'FFT2';
    'FFT3';
    'FFT4';
%     '2';
%     '3';
%     '4';
    'Spk Emb';
    'F0 Emb';
    'E Emb';
    'Upsampling';
    'Pos Enc';
    'FFT1';
    'FFT2';
    'FFT3';
    'FFT4';
    'FFT5';
    'FFT6';
%     '2';
%     '3';
%     '4';
%     '5';
%     '6';
    'Mel';
    'Mel+Post';
};

% =========== TACOTRON2 ===============
% --------- 1-8: ENCODER ---------------
% 1: raw char embeddings
% 2-4: convolutional layers
% 5: Bi-LSTM
% 6: speaker embeddings
% 7: pitch embeddings
% 8: energy embeddings
% -------- 9-11: DECODER --------
% 9: concat(128dim postnet+512dim previous context vector) = input att RNN
% 10: concat(1024dim att RNN+512dim current context vector) = input dec RNN
% 11: concat(1024dim dec RNN+512dim current context vector) = output dec RNN
% -------- 12-13: MEL --------
% 12: mel before postnet
% 13: mel after postnet
name_layer_tacotron = {
    'Phon Emb';
    'Conv1';
    'Conv2';
    'Conv3';
%     '2';
%     '3';
    'Bi-LSTM';
    'Spk Emb';
    'F0 Emb';
    'E Emb';
    'Context Vect';
    'LSTM1';
    'LSTM2';
%     '2';
    'Mel';
    'Mel+Post';
};

if strcmp(model,'fastspeech2')
    % FastSpeech2
    bias = [
        -2;
        -1;
        -0.5;
        -0.25;
        0;
        0.2;
        0.4;
        0.46;
        0.6;
        0.8;
        1;
        2;
        ];
    ref_index = 5;

    layers_to_test = [4, 6];

    %% Models/paths are configured centrally in config_paths.m
if ~exist('cfg', 'var')
    cfg = config_paths();
end
models = cfg.models_categorical;

else
    % Tacotron2
    bias = [
        -2;
        -1;
        -0.5;
        -0.25;
        0;
        0.2;
        0.4;
        0.45;
        0.5;
        0.55;
        0.6;
        0.8;
        1;
        2;
        ];
    ref_index = 5;

    layers_to_test = [2, 4, 5];

    %% Models/paths are configured centrally in config_paths.m
if ~exist('cfg', 'var')
    cfg = config_paths();
end
models = cfg.models_categorical;
end


frame_duration = 1000*256/22050;
nbr_bias = length(bias);

params_to_control = {
    'pause_control_bias', 1;
%     'liaison_control_bias', 2;
};
nbr_params_to_control = length(params_to_control(:,1));


nbr_layers = length(layers_to_test);

% %% Models/paths are configured centrally in config_paths.m
if ~exist('cfg', 'var')
    cfg = config_paths();
end
models = cfg.models_categorical;



nbr_models = length(models(:,1));

silence_threshold = 5;
range_duration_silence = 5:1:24;
index_silences = 1;
index_liaisons = 2;

figure(6);
clf;
x_nbr_graphs = nbr_layers;
y_nbr_graphs = nbr_models;
% create panel
p6 = panel();
p6.margin = [13 28 5 5];
set(groot, "defaultAxesTickLabelInterpreter", 'latex');
set(groot, "defaultTextInterpreter", 'latex');
set(groot, "defaultLegendInterpreter", 'latex');
p6.pack(x_nbr_graphs, y_nbr_graphs);

figure(11);
clf;

figure(7);
clf;
% % create panel
% p2 = panel();
% p2.pack(x_nbr_graphs, y_nbr_graphs);

figure(8);
clf;
% % create panel
p8 = panel();
p8.margin = [13 28 5 5];
% set(groot, "defaultAxesTickLabelInterpreter", 'latex');
% set(groot, "defaultTextInterpreter", 'latex');
% set(groot, "defaultLegendInterpreter", 'latex');
set(groot, "defaultAxesTickLabelInterpreter", 'default');
set(groot, "defaultTextInterpreter", 'default');
set(groot, "defaultLegendInterpreter", 'default');
p8.pack(x_nbr_graphs, y_nbr_graphs);

gain_by_layer = zeros(nbr_params_to_control, nbr_layers, nbr_models);

max_pred_by_layer = zeros(nbr_params_to_control, nbr_layers, nbr_models);
min_pred_by_layer = zeros(nbr_params_to_control, nbr_layers, nbr_models);

max_target_by_layer = zeros(nbr_params_to_control, nbr_layers, nbr_models);
min_target_by_layer = zeros(nbr_params_to_control, nbr_layers, nbr_models);

r_square_by_layer = zeros(nbr_params_to_control, nbr_layers, nbr_models);
silences_distrib_by_model = zeros(nbr_bias, length(range_duration_silence), nbr_layers, nbr_models);
ref_proportion_of_silences_by_model = zeros(1, nbr_models);
for i_model = 1:nbr_models
    model_name = models{i_model, 1};
    basename = models{i_model, 2};
    csv_name = models{i_model, 3};
    name_char_list = sprintf('data/char_list_test_%s', model_name);

    % Load Prediction of silences
    name_silences_pred_enc = sprintf('data/silences_pred_by_layer_enc_%s', model_name);
    name_categorical_bias = sprintf('data/categorical_bias_vector_by_layer_%s', model_name);
    load(name_silences_pred_enc);
    load(name_categorical_bias);

    % Read csv
    path_text_file = fullfile(cfg.csv_root, csv_name);
    fid = fopen(path_text_file, 'r');
    O = textscan(fid, "%s %s %s %s %s", 'Delimiter','|');
    fclose(fid);
    all_txt = O{3};
    nbr_utt = length(all_txt);

    % get all phones in corpus
    fprintf('Loading All phones in corpus | Model: %s\n', model_name);
    if exist([name_char_list '.mat'])
        load(name_char_list);
    else
        [list_char_in_corpus, nbr_char_in_corpus] = get_list_phones_in_corpus(all_txt, false);
        save(name_char_list, 'list_char_in_corpus', 'nbr_char_in_corpus');
    end

    [all_potential_silences_indexes, ~, all_phonation_indexes, ~] = select_silences_patterns(path_text_file);
    nbr_potential_silences = length(all_potential_silences_indexes);
    all_potential_liaisons_indexes = select_liaisons_patterns(path_text_file);
    nbr_potential_liaisons = length(all_potential_liaisons_indexes);

    % Init results
    silences_array = zeros(nbr_potential_silences, nbr_bias, nbr_layers);
    silences_distrib = zeros(nbr_bias, length(range_duration_silence), nbr_layers);
    liaisons_array = zeros(nbr_potential_liaisons, nbr_bias, nbr_layers);

    % Get duration by char (by bias and by layer)
    for i_layer = 1:nbr_layers
        % Load Prediction
        index_layer = layers_to_test(i_layer);
        Mdl = classifier_by_layer{index_layer};
        embeddings = coord_enc_emb_train_bias{index_layer};
        silence_bias_vector = bias_vector_by_layer{index_layer}(:, 1);
        target_proportion_of_silences = zeros(1, nbr_bias);

        for i_bias = 1:nbr_bias
            % Predict bias
            predicted_label = predict(Mdl,embeddings + bias(i_bias)*silence_bias_vector');
            nbr_silences = sum(matches(predicted_label, 'Silence'));
            proportion_silences = 100 * nbr_silences / length(predicted_label);
            target_proportion_of_silences(i_bias) = proportion_silences;

            if bias(i_bias) < 0
                label_bias = 'moins';
            else
                label_bias = 'plus';
            end
            absolute_bias = abs(bias(i_bias));

            fprintf("Model: %s | Layer: %d/%d | Param: %s | Bias %d/%d\n", model_name, i_layer, nbr_layers, params_to_control{index_silences, 1}, i_bias, nbr_bias);
            path_model_bias_by_layer = sprintf('%s/layer_%d/%s/%s_%s_%d/', basename, layers_to_test(i_layer), params_to_control{index_silences, 1}, params_to_control{index_silences, 1}, label_bias, round(100*absolute_bias));

            % get duration of all phones in corpus
            fprintf('Loading All phones duration in corpus | Model: %s\n', model_name);
            list_char_duration_in_corpus = get_list_duration_phones_in_corpus(path_model_bias_by_layer, nbr_char_in_corpus, nbr_utt);
            silences_array(:, i_bias, i_layer) = list_char_duration_in_corpus(all_potential_silences_indexes);

            index_range_silence = 0;
            for i_nbr_frames = range_duration_silence
                index_range_silence = index_range_silence + 1;
                silences_distrib(i_bias, index_range_silence, i_layer) = sum(silences_array(:, i_bias, i_layer) == i_nbr_frames);
            end
        end
        proportion_of_silences_by_layer = 100 * sum(silences_array(:, :, i_layer) >= silence_threshold) / nbr_potential_silences;

        ref_proportion_of_silences_by_model(i_model) = proportion_of_silences_by_layer(ref_index);

        p6(i_layer, i_model).select();
        plot(bias, proportion_of_silences_by_layer, '-');
        hold on;

        if i_layer == 2 && i_model == 1
            figure(11);
            plot(bias, proportion_of_silences_by_layer, '-', 'LineWidth',3, 'Color',[204, 147, 57]/256);
            hold on;
            plot(bias(2:8), proportion_of_silences_by_layer(2:8), ':', 'LineWidth',3, 'color', 'blue');
            xlabel('Pause Bias Magnitude');
            ylabel('Predicted Proportion of Pauses (#silences/# word boundaries)');
            %title('Proportion of silences by speaking rate (/potential silences) | GT');
            legend([{'Prediction'}, {'Range of Interest'}], 'Location', 'northwest');
            ylim([-5, 105]);
            grid on;
%             fontsize(gcf,scale=1.1);
        end

        figure(7);
%         p2(i_layer, i_model).select();
        subplot(x_nbr_graphs, y_nbr_graphs, nbr_models*(i_layer-1) + i_model);
        imagesc(log(silences_distrib(:, :, i_layer)));
%         imagesc(silences_distrib(:, :, i_layer));

        p8(i_layer, i_model).select();
        limit_points = 22;
%         limit_regression = boolean((target_proportion_of_silences< limit_points).* (proportion_of_silences_by_layer<limit_points));
        limit_regression = boolean(target_proportion_of_silences< limit_points);
%         limit_regression = 1:length(target_proportion_of_silences);
        plot(target_proportion_of_silences, proportion_of_silences_by_layer, '-*');
        hold on;
        plot([0, 100], [0, 100], '-');

%         p = polyfitZero(target_proportion_of_silences, proportion_of_silences_by_layer, 1);
        p = polyfit(target_proportion_of_silences(limit_regression), proportion_of_silences_by_layer(limit_regression), 1);
        regression_control = polyval(p, target_proportion_of_silences(limit_regression));

        % afine through 0 bias
        zero_bias_proportion = proportion_of_silences_by_layer(ref_index);
        ft = fittype(sprintf('a*(x) + %f*(1-a)', zero_bias_proportion),'indep','x');
        [mdl, gof] = fit(target_proportion_of_silences(limit_regression)', proportion_of_silences_by_layer(limit_regression)', ft, 'start', 1);

        slope = mdl.a;
        offset = zero_bias_proportion*(1-mdl.a);
        p8(i_layer, i_model).select();
%         plot(target_proportion_of_silences(limit_regression), regression_control, '-');
        plot(target_proportion_of_silences(limit_regression), slope*target_proportion_of_silences(limit_regression)+offset, '-');

%         gain_by_layer(1, i_layer, i_model) = p(1);
%         r_square_by_layer(1, i_layer, i_model) = 1 - sum((regression_control-proportion_of_silences_by_layer(limit_regression)).^2)/(var(proportion_of_silences_by_layer(limit_regression))*length(proportion_of_silences_by_layer(limit_regression)));
        gain_by_layer(1, i_layer, i_model) = slope;
        r_square_by_layer(1, i_layer, i_model) = gof.rsquare;

        max_pred_by_layer(1, i_layer, i_model) = max(proportion_of_silences_by_layer(limit_regression));
        min_pred_by_layer(1, i_layer, i_model) = proportion_of_silences_by_layer(1);
        max_target_by_layer(1, i_layer, i_model) = max(target_proportion_of_silences(limit_regression));
        min_target_by_layer(1, i_layer, i_model) = target_proportion_of_silences(1);
    end

    silences_distrib_by_model(:, :, :, i_model) = silences_distrib;

    % Save results
    save(sprintf('data/controllability_array_categorical_%s', model_name), 'silences_array', 'silences_distrib', 'liaisons_array');
end

min_value_color_map = max([0, min(min(min(min(log(silences_distrib_by_model)))))]);
max_value_color_map = max(max(max(max(log(silences_distrib_by_model)))));
% min_value_color_map = min(min(min(min(silences_distrib_by_model))));
% max_value_color_map = max(max(max(max(silences_distrib_by_model))));
custom_colormap = def_colormap(min_value_color_map, max_value_color_map);
for i_model = 1:nbr_models
    name_model = models{i_model, 1};
    model_type = models{i_model, 5};
    model_legend_label = models{i_model, 6};
    if strcmp(model_type, 'fastspeech2')
        name_layers = name_layer_fastspeech;
    elseif strcmp(model_type, 'tacotron2')
        name_layers = name_layer_tacotron;
    end
    for i_layer = 1:nbr_layers
        index_layer = layers_to_test(i_layer);

        p6(i_layer, i_model).select();
        title(sprintf('Controllability | %s | Layer %s', name_model, name_layers{index_layer}));
        grid on;
        ylim([0 100]);
        if (i_layer == nbr_layers)
            xlabel('Silence Bias');
        end
        if i_model == 1
            ylabel('Proportion of Silences');
        end

        figure(7);
%         p2(i_layer, i_model).select();
        subplot(x_nbr_graphs, y_nbr_graphs, nbr_models*(i_layer-1) + i_model);
        title(sprintf('Distr Sil | %s | Layer %s', name_model, name_layers{index_layer}));
        grid on;
        colormap(custom_colormap);
        if i_model == nbr_models
            c = colorbar;
            clim([min_value_color_map,max_value_color_map]);
        end
        if (i_layer == nbr_layers)
            xlabel('Silence Duration (ms)');
        end
        if i_model == 1
            ylabel('Bias');
        end
        yticks(1:nbr_bias);
        yticklabels(bias);
        set(gca,'YDir','normal');
        xticks([1 , 5:5:20]);
        xticklabels((5:5:25)*frame_duration);

        p8(i_layer, i_model).select();
        title(sprintf('Controllability | %s | Layer %s', name_model, name_layers{index_layer}));
        grid on;
        xlim([0 limit_points]);
        ylim([0 limit_points]);
        legend([{'Measure'}, {'Target'}, {'Regression'}], 'Location','northwest');
        if (i_layer == nbr_layers)
            xlabel('Target Proportion of Silences');
        end
        if i_model == 1
            ylabel('Measured Proportion of Silences');
        end
    end
end


%%

if strcmp(model,'fastspeech2')
    best_layer_indexes_array_FS = [2, 2, 2, 2];
    plot_categorical_controllability_by_layer( ...
        r_square_by_layer, ...
        gain_by_layer, ...
        models, ...
        name_layer_fastspeech(layers_to_test), ...
        params_to_control, ...
        10, ...
        model_type, ...
        best_layer_indexes_array_FS, ...
        max_pred_by_layer, ...
        min_pred_by_layer, ...
        max_target_by_layer, ...
        min_target_by_layer, ...
        ref_proportion_of_silences_by_model ...
        );

else
    best_layer_indexes_array_TC = [3, 3];
    plot_categorical_controllability_by_layer( ...
        r_square_by_layer, ...
        gain_by_layer, ...
        models, ...
        name_layer_tacotron(layers_to_test), ...
        params_to_control, ...
        10, ...
        model_type, ...
        best_layer_indexes_array_TC, ...
        max_pred_by_layer, ...
        min_pred_by_layer, ...
        max_target_by_layer, ...
        min_target_by_layer, ...
        ref_proportion_of_silences_by_model ...
        );
end