% Figure(1) and (2) in the code = Fig. 4 in the paper

clearvars;
plotSettings;
flag_write_residual = 1;
plot_supra_segmental = false;
plot_segmental = false;

tic

% Fix rng
rng(1234);
list_vowels = {'a'; 'u'; 'i'; 'e'; 'e^'; 'y'; 'x'; 'x^'; 'o'; 'o^'; 'x~'; 'e~'; 'o~'; 'a~';};
% list_vowels = {'i'};
nbr_vowels = length(list_vowels);
correlation_to_disp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
% correlation_to_disp = [1];
selected_acoustic_params = [1, 8, 11, 15, 18, 20, 21, 26, 34, 22, 35];
selected_acoustic_params = selected_acoustic_params(correlation_to_disp);
label_acoustic_plot = {'D_{ }'; 'f_o'; 'F_1'; 'F_2'; 'F_3'; 'ST_{ }'; 'E_{ }'; 'RP_{ }'; 'Pfit'; 'CoG_{ }'; 'SB_{1kHz}'};
label_acoustic = {'D'; 'fo'; 'F1'; 'F2'; 'F3'; 'ST'; 'E'; 'RP'; 'Pfit'; 'CoG'; 'SB'};
nbr_acoustic_params = length(selected_acoustic_params);
selected_dim = [1, 2];
% legend_by_param = {
%     '-*', [193, 214, 167]/255;
%     '-*', [127, 218, 244]/255;
%     '--+', [253, 185, 36]/255;
%     '--+', [134, 156, 152]/255;
%     '--+', [131, 78, 86]/255;
%     '--^', [255, 248, 91]/255;
%     '-*', [255, 0, 76]/255;
%     ':x', [0, 154, 205]/255;
%     '-.x', [0, 250, 154]/255;
%     '--^', [123, 104, 238]/255;
%     '--^', [255, 110, 180]/255;
%     };
legend_by_param = {
    '-*', pcol.gipsaPPC;
    '-*', pcol.gipsaGAIA;
    '--+', [253, 185, 36]/255;
    '--+', [134, 156, 152]/255;
    '--+', [131, 78, 86]/255;
    '--^', pcol.gipsaPSD;
    '-*', pcol.gipsaPAD;
    ':x', pcol.gipsaBlue;
    '-.x', [0, 250, 154]/255;
    '--^', [123, 104, 238]/255;
    '--^', [255, 110, 180]/255;
    };
% list_params_supra_segmental = [1, 2, 6, 7, 8, 9];
list_params_supra_segmental = [1, 2, 6, 7, 8];
list_params_segmental = [3, 4, 5, 10, 11];

% Control Params
% MDS
use_MDS = true;
select_best_dim = true;
MDS_threshold = false;
part_random_phon_for_mds = 0.5;
% part_random_phon_for_mds = 1;
use_only_vowels = true;

% Regression
reduce_model_regression = false;%true;
% center_params = false;
center_emb = true;

% Reload
reload_corpus = false;
reload_acoustic_params = false;
reload_encoder_emb = false;
reload_encoder_emb_reduced = false;
reload_decoder_emb = false;
reload_decoder_emb_reduced = false;
reload_mel = false;

% Name Generator form options
if use_MDS
    if select_best_dim
        reduced_name = 'mds90';
    else
        reduced_name = 'mds100';
    end
else
    reduced_name = 'noMds';
end

if center_emb
    center_emb_name = 'center';
else
    center_emb_name = 'noCenter';
end

if use_only_vowels
    phon_type_name = 'vowels';
else
    phon_type_name = 'all';
end

if reduce_model_regression
    reduced_regression = 'reducedRegression';
else
    reduced_regression = 'noReducedRegression';
end

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
    'Phon. Emb.';
    'Pos. Enc.';
    'FFT1';
    %     'FFT2';
    %     'FFT3';
    %     'FFT4';
    '2';
    '3';
    '4';
    'Spk Emb.';
    'fo Emb.';
    'E Emb.';
    'Upsampling';
    'Pos. Enc.';
    'FFT1';
    %     'FFT2';
    %     'FFT3';
    %     'FFT4';
    %     'FFT5';
    %     'FFT6';
    '2';
    '3';
    '4';
    '5';
    '6';
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
    'Phon. Emb.';
    'Conv1';
    %     'Conv2';
    %     'Conv3';
    '2';
    '3';
    'Bi-LSTM';
    'Spk Emb.';
    'fo Emb.';
    'E Emb.';
    'Context Vect.';
    'LSTM1';
    %     'LSTM2';
    '2';
    'Mel';
    'Mel+Post';
    };
% tacotron_residual_dim = [128, 1024, 1024];
% tacotron_context_vec_dim = [512, 512, 512];


corr_coef_by_layer_by_model = [];

%% Models/paths are configured centrally in config_paths.m
if ~exist('cfg', 'var')
    cfg = config_paths();
end
models = cfg.models_corr_op;

nbr_models = length(models(:,1));




if flag_write_residual == 1
    %%
    res_path = cell(1,length(label_acoustic));
    for i_param = 1:length(label_acoustic)
        res_path{i_param} = fullfile(cfg.results_root,'residuals',strcat('residuals_',label_acoustic{i_param},'.csv'));

        % Define column names and data types
        res_col_names = {'obs', 'param', 'calc', 'model', 'encdec', 'layer',  'corr', 'error'};
        res_var_types = {'double', 'string', 'string', 'string', 'string', 'double', 'double', 'double'};

        % Initialize an empty table with the correct structure
        T = table('Size', [0, numel(res_col_names)], ...
            'VariableTypes', res_var_types, ...
            'VariableNames', res_col_names);

        writetable(T, res_path{i_param});
    end
end


plotSettings;


% Plot Correlation by layer for all models
fig1 = figure(1);
clf;
fp = get(gcf,'Position');
if nbr_models > 4
    fp(3:4) = [1900 460];
    set(gcf,'PaperSize',[67 16])
else
    fp(3:4) = [1250 460];
    set(gcf,'PaperSize',[44 16])
end
set(gcf,'Position',fp)
set(gcf,'Renderer', 'painters');

set(groot, "defaultAxesTickLabelInterpreter", 'latex');
set(groot, "defaultTextInterpreter", 'latex');
set(groot, "defaultLegendInterpreter", 'latex');
% p.title('R² by layer (bias computation)');
% sgtitle('R² by layer (bias computation)');


fig2 = figure(2);
clf;
fp = get(gcf,'Position');
if nbr_models > 4
    fp(3:4) = [1900 460];
    set(gcf,'PaperSize',[66 16])
else
    fp(3:4) = [1250 460];
    set(gcf,'PaperSize',[44 16])
end
set(gcf,'Position',fp)
set(gcf,'Renderer', 'painters');

set(groot, "defaultAxesTickLabelInterpreter", 'latex');
set(groot, "defaultTextInterpreter", 'latex');
set(groot, "defaultLegendInterpreter", 'latex');
% p.title('R² by layer (bias computation)');
% sgtitle('R² by layer (bias computation)');

if nbr_models > 4
    posPlot = plotPosition(1,nbr_models,[0.02 0.0 0.08 0.37],[0.015 0.01 0.00 0.01]);
else
    posPlot = plotPosition(1,4,[0.035 0.0 0.08 0.37],[0.02 0.01 0.00 0.01]);
end

p = panel();
p.margin = [13 28 5 5];
if plot_supra_segmental && plot_segmental
    p.pack(2, nbr_models);
else
    p.pack(1, nbr_models);
end

ptxt.title.size = 26;
ptxt.label.size = 22;
ptxt.axis.size = 17;
ptxt.text.size = 18;


for i_model = 1:nbr_models
    name_model = models{i_model, 1};
    path_model = models{i_model, 2};
    path_csv = models{i_model, 3};
    model_type = models{i_model, 4};
    legend_model = models{i_model, 5};

    if strcmp(model_type, 'fastspeech')
        ptxt.legend.size = 22;
    else
        ptxt.legend.size = 18;
    end

    if strcmp(model_type, 'tacotron')
        % Tacotron2
        max_dim_by_param = [
            11;
            11;
            11;
            11;
            11;
            11;
            11;
            11;
            11;
            11;
            11;
            ];
        %         max_dim_by_param = 11*ones(nbr_acoustic_params, 1);
    elseif strcmp(model_type, 'fastspeech')
        max_dim_by_param = [
            7;
            17;
            17;
            17;
            17;
            17;
            17;
            17;
            7;
            17;
            17;
            ];
        %         max_dim_by_param = 17*ones(nbr_acoustic_params, 1);
    end

    name_phone_list_mat = sprintf('data/phones_list_%s', name_model);
    name_acoustic_mat = sprintf('data/acoustic_measures_%s', name_model);
    name_enc_emb_mat = sprintf('data/enc_emb_by_layer_%s', name_model);
    name_dec_emb_mat = sprintf('data/dec_emb_by_layer_%s', name_model);
    name_postnet_emb_mat = sprintf('data/postnet_emb_by_layer_%s', name_model);
    name_mel_mat = sprintf('data/mel_by_layer_%s', name_model);

    name_enc_transfer_mat = sprintf('data/enc_transfer_emb_by_layer_%s', name_model);
    name_enc_predictor_mat = sprintf('data/enc_predictor_emb_by_layer_%s', name_model);
    name_dec_transfer_mat = sprintf('data/dec_transfer_emb_by_layer_%s', name_model);
    name_dec_predictor_mat = sprintf('data/dec_predictor_emb_by_layer_%s', name_model);

    fid = fopen(path_csv, 'r');
    F = textscan(fid, "%s %s %s %s %s", 'Delimiter','|');
    fclose(fid);

    nbr_utt = length(F{1});

    % get all phones in corpus
    fprintf('Loading All phones in corpus | Model: %s\n', name_model);
    if exist([name_phone_list_mat '.mat']) && ~reload_corpus
        load(name_phone_list_mat);
    else
        [list_phones_in_corpus, nbr_char_in_corpus] = get_list_phones_in_corpus(F{3}, false);
        save(name_phone_list_mat, 'list_phones_in_corpus', 'nbr_char_in_corpus');
    end

    % load acoustic params
    fprintf('Loading All acoustic measurements | Model: %s\n', name_model);
    if exist([name_acoustic_mat '.mat']) && ~reload_acoustic_params
        load(name_acoustic_mat);
    else
        acoustic_params = load_acoustic_param_from_model(path_csv, path_model, nbr_utt, model_type)';
        save(name_acoustic_mat, 'acoustic_params');
    end

    % Keep only vowels params
    considered_phon_indexes = [];
    if use_only_vowels
        for i_vowel = 1:nbr_vowels
            index_in_phones_list = find(cellfun(@(subc) strcmp(list_vowels{i_vowel}, subc), list_phones_in_corpus(:, 1)));
            considered_phon_indexes = [considered_phon_indexes; list_phones_in_corpus{index_in_phones_list, 2}];
        end
        considered_phon_indexes = sort(considered_phon_indexes);
    else
        considered_phon_indexes = 1:nbr_char_in_corpus;
    end
    considered_acoustic_params = acoustic_params(considered_phon_indexes, selected_acoustic_params);
    considered_acoustic_params_centered = considered_acoustic_params;
    considered_acoustic_params_mean = considered_acoustic_params;
    % Center acoustic params
    %     if center_params
    for index_param = 1:nbr_acoustic_params
        considered_acoustic_params_centered(:, index_param) = absolute_param_to_zscore(considered_acoustic_params_centered(:, index_param), list_phones_in_corpus, considered_phon_indexes, index_param, 'delta');
        considered_acoustic_params_mean(:, index_param) = absolute_param_to_zscore(considered_acoustic_params_mean(:, index_param), list_phones_in_corpus, considered_phon_indexes, index_param, 'mean');
    end
    %     end

    % ---------------------- ENCODER ---------------------------
    % Load embeddings by layer
    fprintf('Loading All embeddings in ENCODER | Model: %s\n', name_model);
    if exist([name_enc_emb_mat '.mat']) && ~reload_encoder_emb
        load(name_enc_emb_mat);
        nbr_layers = size(all_enc_emb_mat, 3);

        if reload_encoder_emb_reduced
            % Reduce Dimension (MDS on vowels only)
            random_considered_phon_indexes = randperm(length(considered_phon_indexes), round(part_random_phon_for_mds*length(considered_phon_indexes)));
            all_vowel_emb_enc = all_enc_emb_mat(considered_phon_indexes(random_considered_phon_indexes), :, :);
            nbr_dim = size(all_vowel_emb_enc, 2);

            all_enc_emb_mat_reduced = cell(nbr_layers, 1);
            all_enc_emb_mat_reduced_centered = cell(nbr_layers, 1);
            enc_transfer_mat_reduced2init = cell(nbr_layers, 1);
            enc_transfer_mat_reduced2init_centered = cell(nbr_layers, 1);
            enc_transfer_mat_init2reduced = cell(nbr_layers, 1);
            enc_transfer_mat_init_centered2reduced = cell(nbr_layers, 1);

            for i_layer = 1:nbr_layers
                %                 if (i_layer == 1)
                %                     W = zeros(nbr_dim, 2);
                %                     X = zeros(2, nbr_dim);
                %                 else
                [~, W, ~, X] = compute_coef_mds(all_vowel_emb_enc(:, :, i_layer)', MDS_threshold, select_best_dim);
                %                 end
                all_enc_emb_mat_reduced{i_layer} = all_enc_emb_mat(:, :, i_layer)*W;
                enc_transfer_mat_reduced2init{i_layer} = X;
                enc_transfer_mat_init2reduced{i_layer} = W;

                % Same for centered embeddings
                all_enc_emb_mat_centered = center_emb_by_phon_label(all_enc_emb_mat(:, :, i_layer), list_phones_in_corpus, 1:size(all_enc_emb_mat, 1));

                all_vowel_emb_enc_centered = all_enc_emb_mat_centered(considered_phon_indexes(random_considered_phon_indexes), :);
                if (i_layer == 1)
                    W = zeros(nbr_dim, 2);
                    X = zeros(2, nbr_dim);
                else
                    [~, W, ~, X] = compute_coef_mds(all_vowel_emb_enc_centered', MDS_threshold, select_best_dim);
                end
                all_enc_emb_mat_reduced_centered{i_layer} = all_enc_emb_mat_centered*W;
                enc_transfer_mat_reduced2init_centered{i_layer} = X;
                enc_transfer_mat_init_centered2reduced{i_layer} = W;
            end
            save(name_enc_emb_mat, 'all_enc_emb_mat', 'all_enc_emb_mat_reduced', 'enc_transfer_mat_reduced2init', 'all_enc_emb_mat_reduced_centered', 'enc_transfer_mat_reduced2init_centered');
            save(name_enc_transfer_mat, 'enc_transfer_mat_init2reduced', 'enc_transfer_mat_init_centered2reduced');
        end
    else
        all_enc_emb_mat = load_encoder_embeddings_by_layer(path_model, nbr_utt, nbr_char_in_corpus);
        nbr_layers = size(all_enc_emb_mat, 3);

        % Reduce Dimension (MDS on vowels only)
        random_considered_phon_indexes = randperm(length(considered_phon_indexes), round(part_random_phon_for_mds*length(considered_phon_indexes)));
        all_vowel_emb_enc = all_enc_emb_mat(considered_phon_indexes(random_considered_phon_indexes), :, :);

        all_enc_emb_mat_reduced = cell(nbr_layers, 1);
        enc_transfer_mat_reduced2init = cell(nbr_layers, 1);
        all_enc_emb_mat_reduced_centered = cell(nbr_layers, 1);
        enc_transfer_mat_reduced2init_centered = cell(nbr_layers, 1);
        enc_transfer_mat_init2reduced = cell(nbr_layers, 1);
        enc_transfer_mat_init_centered2reduced = cell(nbr_layers, 1);
        for i_layer = 1:nbr_layers
            dim_emb = length(all_vowel_emb_enc(1, :, i_layer));
            [~, W, ~, X] = compute_coef_mds(all_vowel_emb_enc(:, :, i_layer)', MDS_threshold, select_best_dim);
            all_enc_emb_mat_reduced{i_layer} = all_enc_emb_mat(:, :, i_layer)*W;
            enc_transfer_mat_reduced2init{i_layer} = X;
            enc_transfer_mat_init2reduced{i_layer} = W;

            % Same for centered embeddings
            all_enc_emb_mat_centered = center_emb_by_phon_label(all_enc_emb_mat(:, :, i_layer), list_phones_in_corpus, 1:size(all_enc_emb_mat, 1));

            all_vowel_emb_enc_centered = all_enc_emb_mat_centered(considered_phon_indexes(random_considered_phon_indexes), :);
            if (i_layer == 1)
                W = zeros(dim_emb, 2);
                X = zeros(2, dim_emb);
            else
                [~, W, ~, X] = compute_coef_mds(all_vowel_emb_enc_centered', MDS_threshold, select_best_dim);
            end
            all_enc_emb_mat_reduced_centered{i_layer} = all_enc_emb_mat_centered*W;
            enc_transfer_mat_reduced2init_centered{i_layer} = X;
            enc_transfer_mat_init_centered2reduced{i_layer} = W;
        end

        save(name_enc_emb_mat, 'all_enc_emb_mat', 'all_enc_emb_mat_reduced', 'enc_transfer_mat_reduced2init', 'all_enc_emb_mat_reduced_centered', 'enc_transfer_mat_reduced2init_centered');
        save(name_enc_transfer_mat, 'enc_transfer_mat_init2reduced', 'enc_transfer_mat_init_centered2reduced');
    end

    % Center embeddings
    corr_coef_by_layer_enc = zeros(nbr_acoustic_params, nbr_layers);
    corr_coef_by_layer_enc_centered = zeros(nbr_acoustic_params, nbr_layers);
    corr_coef_by_layer_enc_mean = zeros(nbr_acoustic_params, nbr_layers);

    enc_predicted_acoustic_params_mean_by_layer = zeros(length(considered_phon_indexes), nbr_acoustic_params, nbr_layers);
    enc_predicted_acoustic_params_centered_by_layer = zeros(length(considered_phon_indexes), nbr_acoustic_params, nbr_layers);

    bias_vector_by_layer_enc = cell(nbr_layers, 1);
    pred_coef_by_layer_enc = cell(nbr_layers, 1);
    constant_coef_by_layer_enc = cell(nbr_layers, 1);
    pred_coef_by_layer_enc_centered = cell(nbr_layers, 1);
    constant_coef_by_layer_enc_centered = cell(nbr_layers, 1);
    pred_coef_by_layer_enc_mean = cell(nbr_layers, 1);
    constant_coef_by_layer_enc_mean = cell(nbr_layers, 1);
    for i_layer = 1:nbr_layers
        fprintf('Compute Correlation for model %s, layer %d | ENCODER\n', name_model, i_layer);

        %         if use_MDS
        %             considered_emb = all_enc_emb_mat_reduced{i_layer}(considered_phon_indexes, :);
        %         else
        %             considered_emb = all_enc_emb_mat(considered_phon_indexes, :, i_layer);
        %         end
        %
        %         if center_emb
        %             centered_emb_by_layer = center_emb_by_phon_label(considered_emb, list_phones_in_corpus, considered_phon_indexes);
        %         else
        %             centered_emb_by_layer = double(considered_emb);
        %         end
        %
        %         emb_by_layer = double(considered_emb);
        %         if use_MDS && center_emb
        %             emb_by_layer = all_enc_emb_mat_reduced_centered{i_layer}(considered_phon_indexes, :);
        %         elseif use_MDS && ~center_emb
        %             emb_by_layer = all_enc_emb_mat_reduced{i_layer}(considered_phon_indexes, :);
        %         elseif ~use_MDS && center_emb
        %             emb_by_layer = all_enc_emb_mat(considered_phon_indexes, :, i_layer);
        %             emb_by_layer = center_emb_by_phon_label(emb_by_layer, list_phones_in_corpus, considered_phon_indexes);
        %         elseif ~use_MDS && ~center_emb
        %             emb_by_layer = all_enc_emb_mat(considered_phon_indexes, :, i_layer);
        if use_MDS
            emb_by_layer = all_enc_emb_mat_reduced{i_layer}(considered_phon_indexes, :);
            emb_by_layer_centered = all_enc_emb_mat_reduced_centered{i_layer}(considered_phon_indexes, :);
        elseif ~use_MDS
            emb_by_layer = all_enc_emb_mat(considered_phon_indexes, :, i_layer);
            emb_by_layer_centered = center_emb_by_phon_label(emb_by_layer, list_phones_in_corpus, considered_phon_indexes);
        else
            error('Conditions calcul embeddings impossible');
        end
        emb_by_layer = double(emb_by_layer);
        emb_by_layer_centered = double(emb_by_layer_centered);

        %         if center_emb
        %             acoustic_params_for_regression = considered_acoustic_params_centered;
        %         else
        %             acoustic_params_for_regression = considered_acoustic_params;
        %         end

        % Compute regression by layer
        [corr_coef, predicted_acoustic_params, all_beta, all_beta_normalized, all_constants, ~, ~, ~] = acoustic_regression_in_latent_space(emb_by_layer, considered_acoustic_params, selected_dim, reduce_model_regression);
        corr_coef_by_layer_enc(:, i_layer) = corr_coef';

        if flag_write_residual == 1
            %% Write regression residual
            for i_param = list_params_supra_segmental
                fprintf('\t> %s (R^2 = %0.2f) - error: %0.2f\n',label_acoustic{i_param},corr_coef(i_param),rmse(predicted_acoustic_params(~isnan(considered_acoustic_params(:,i_param)),i_param),considered_acoustic_params(~isnan(considered_acoustic_params(:,i_param)),i_param)))
                write_residual_OP(res_path{i_param},predicted_acoustic_params(:,i_param),considered_acoustic_params(:,i_param),'none',label_acoustic{i_param},i_layer,name_model,'enc',corr_coef(i_param))
            end
        end


        % Compute regression by layer centered
        [corr_coef_centered, predicted_acoustic_params_centered, all_beta_centered, all_beta_normalized_centered, all_constants_centered, ~, ~, ~] = acoustic_regression_in_latent_space(emb_by_layer_centered, considered_acoustic_params_centered, selected_dim, reduce_model_regression);
        corr_coef_by_layer_enc_centered(:, i_layer) = corr_coef_centered';

        if flag_write_residual == 1
            %% Write regression residual
            for i_param = list_params_segmental
                fprintf('\t> %s (R^2 = %0.2f) - error: %0.2f\n',label_acoustic{i_param},corr_coef_centered(i_param),rmse(predicted_acoustic_params_centered(~isnan(considered_acoustic_params_centered(:,i_param)),i_param),considered_acoustic_params_centered(~isnan(considered_acoustic_params_centered(:,i_param)),i_param)))
                write_residual_OP(res_path{i_param},predicted_acoustic_params_centered(:,i_param),considered_acoustic_params_centered(:,i_param),'centred',label_acoustic{i_param},i_layer,name_model,'enc',corr_coef_centered(i_param))
            end
        end


        % Compute regression by layer mean
        [corr_coef_mean, predicted_acoustic_params_mean, all_beta_mean, all_beta_normalized_mean, all_constants_mean, ~, ~, ~] = acoustic_regression_in_latent_space(emb_by_layer, considered_acoustic_params_mean, selected_dim, reduce_model_regression);
        corr_coef_by_layer_enc_mean(:, i_layer) = corr_coef_mean';

        if flag_write_residual == 1
            %% Write regression residual
            for i_param = list_params_segmental
                fprintf('\t> %s (R^2 = %0.2f) - error: %0.2f\n',label_acoustic{i_param},corr_coef_mean(i_param),rmse(predicted_acoustic_params_mean(~isnan(considered_acoustic_params_mean(:,i_param)),i_param),considered_acoustic_params_mean(~isnan(considered_acoustic_params_mean(:,i_param)),i_param)))
                write_residual_OP(res_path{i_param},predicted_acoustic_params_mean(:,i_param),considered_acoustic_params_mean(:,i_param),'mean',label_acoustic{i_param},i_layer,name_model,'enc',corr_coef_mean(i_param))
            end
        end


        enc_predicted_acoustic_params_mean_by_layer(:, :, i_layer) = predicted_acoustic_params_mean;
        enc_predicted_acoustic_params_centered_by_layer(:, :, i_layer) = predicted_acoustic_params_centered;

        % Save predictor coef
        pred_coef_by_layer_enc{i_layer} = all_beta;
        constant_coef_by_layer_enc{i_layer} = all_constants;
        pred_coef_by_layer_enc_centered{i_layer} = all_beta_centered;
        constant_coef_by_layer_enc_centered{i_layer} = all_constants_centered;
        pred_coef_by_layer_enc_mean{i_layer} = all_beta_mean;
        constant_coef_by_layer_enc_mean{i_layer} = all_constants_mean;

        % Save bias vector by param by layer
        if use_MDS && center_emb
            bias_vector_current_layer = (all_beta_normalized_centered'*enc_transfer_mat_reduced2init_centered{i_layer})';
        elseif use_MDS && ~center_emb
            bias_vector_current_layer = (all_beta_normalized'*enc_transfer_mat_reduced2init{i_layer})';
        elseif ~use_MDS && center_emb
            bias_vector_current_layer = all_beta_normalized_centered;
        elseif ~use_MDS && ~center_emb
            bias_vector_current_layer = all_beta_normalized;
        end
        bias_vector_by_layer_enc{i_layer} = bias_vector_current_layer;
    end
    % ---------------------- FIN ENCODER ---------------------------

    % save predictors from latent spaces
    save(name_enc_predictor_mat, 'pred_coef_by_layer_enc', 'constant_coef_by_layer_enc', 'pred_coef_by_layer_enc_centered', 'constant_coef_by_layer_enc_centered', 'pred_coef_by_layer_enc_mean', 'constant_coef_by_layer_enc_mean');

    % ---------------------- DECODER ---------------------------
    % Load embeddings by layer
    fprintf('Loading All embeddings in DECODER | Model: %s\n', name_model);
    if exist([name_dec_emb_mat '.mat']) && ~reload_decoder_emb
        load(name_dec_emb_mat);
        nbr_layers = length(all_dec_emb_mat);

        if reload_decoder_emb_reduced
            % Reduce Dimension (MDS on vowels only)
            random_considered_phon_indexes = randperm(length(considered_phon_indexes), round(part_random_phon_for_mds*length(considered_phon_indexes)));

            all_dec_emb_mat_reduced = cell(nbr_layers, 1);
            all_dec_emb_mat_residual_reduced = cell(nbr_layers, 1);
            all_dec_emb_mat_context_vector_reduced = cell(nbr_layers, 1);
            all_dec_emb_mat_reduced_centered = cell(nbr_layers, 1);
            all_dec_emb_mat_residual_reduced_centered = cell(nbr_layers, 1);
            all_dec_emb_mat_context_vector_reduced_centered = cell(nbr_layers, 1);

            dec_transfer_mat_reduced2init = cell(nbr_layers, 1);
            dec_transfer_mat_residual_reduced2init = cell(nbr_layers, 1);
            dec_transfer_mat_context_vector_reduced2init = cell(nbr_layers, 1);
            dec_transfer_mat_reduced2init_centered = cell(nbr_layers, 1);
            dec_transfer_mat_residual_reduced2init_centered = cell(nbr_layers, 1);
            dec_transfer_mat_context_vector_reduced2init_centered = cell(nbr_layers, 1);

            dec_transfer_mat_init2reduced = cell(nbr_layers, 1);
            dec_transfer_mat_init_centered2reduced = cell(nbr_layers, 1);

            for i_layer = 1:nbr_layers
                % Complete embeddings
                all_vowel_emb_dec = all_dec_emb_mat{i_layer}(considered_phon_indexes(random_considered_phon_indexes), :);
                [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec', MDS_threshold, select_best_dim) ;
                all_dec_emb_mat_reduced{i_layer} = all_dec_emb_mat{i_layer}*W;
                dec_transfer_mat_reduced2init{i_layer} = X;
                dec_transfer_mat_init2reduced{i_layer} = W;

                % Same for centered embeddings
                all_dec_emb_mat_centered = center_emb_by_phon_label(all_dec_emb_mat{i_layer}, list_phones_in_corpus, 1:size(all_dec_emb_mat{i_layer}, 1));

                all_vowel_emb_dec_centered = all_dec_emb_mat_centered(considered_phon_indexes(random_considered_phon_indexes), :);
                [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec_centered', MDS_threshold, select_best_dim);
                all_dec_emb_mat_reduced_centered{i_layer} = all_dec_emb_mat_centered*W;
                dec_transfer_mat_reduced2init_centered{i_layer} = X;
                dec_transfer_mat_init_centered2reduced{i_layer} = W;

                if strcmp(model_type, 'tacotron')
                    % residual embeddings
                    all_vowel_emb_dec = all_dec_emb_mat_residual{i_layer}(considered_phon_indexes(random_considered_phon_indexes), :);
                    [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec', MDS_threshold, select_best_dim) ;
                    all_dec_emb_mat_residual_reduced{i_layer} = all_dec_emb_mat_residual{i_layer}*W;
                    dec_transfer_mat_residual_reduced2init{i_layer} = X;

                    % Same for centered Embeddings
                    all_dec_emb_mat_residual_centered = center_emb_by_phon_label(all_dec_emb_mat_residual{i_layer}, list_phones_in_corpus, 1:size(all_dec_emb_mat_residual{i_layer}, 1));
                    all_vowel_emb_dec_residual_centered = all_dec_emb_mat_residual_centered(considered_phon_indexes(random_considered_phon_indexes), :);
                    [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec_residual_centered', MDS_threshold, select_best_dim);
                    all_dec_emb_mat_residual_reduced_centered{i_layer} = all_dec_emb_mat_residual_centered*W;
                    dec_transfer_mat_residual_reduced2init_centered{i_layer} = X;

                    % context vector embeddings only
                    all_vowel_emb_dec = all_dec_emb_mat_context_vector{i_layer}(considered_phon_indexes(random_considered_phon_indexes), :);
                    [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec', MDS_threshold, select_best_dim) ;
                    all_dec_emb_mat_context_vector_reduced{i_layer} = all_dec_emb_mat_context_vector{i_layer}*W;
                    dec_transfer_mat_context_vector_reduced2init{i_layer} = X;

                    % Same for centered Embeddings
                    all_dec_emb_mat_context_vector_centered = center_emb_by_phon_label(all_dec_emb_mat_context_vector{i_layer}, list_phones_in_corpus, 1:size(all_dec_emb_mat_context_vector{i_layer}, 1));
                    all_vowel_emb_dec_context_vector_centered = all_dec_emb_mat_context_vector_centered(considered_phon_indexes(random_considered_phon_indexes), :);
                    [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec_context_vector_centered', MDS_threshold, select_best_dim);
                    all_dec_emb_mat_context_vector_reduced_centered{i_layer} = all_dec_emb_mat_context_vector_centered*W;
                    dec_transfer_mat_context_vector_reduced2init_centered{i_layer} = X;
                end
            end

            save( ...
                name_dec_emb_mat, ...
                'all_dec_emb_mat', ...
                'all_dec_emb_mat_residual', ...
                'all_dec_emb_mat_context_vector', ...
                'all_dec_emb_mat_reduced', ...
                'all_dec_emb_mat_residual_reduced', ...
                'all_dec_emb_mat_context_vector_reduced', ...
                'dec_transfer_mat_reduced2init', ...
                'dec_transfer_mat_residual_reduced2init', ...
                'dec_transfer_mat_context_vector_reduced2init', ...
                'all_dec_emb_mat_reduced_centered', ...
                'all_dec_emb_mat_residual_reduced_centered', ...
                'all_dec_emb_mat_context_vector_reduced_centered', ...
                'dec_transfer_mat_reduced2init_centered', ...
                'dec_transfer_mat_residual_reduced2init_centered', ...
                'dec_transfer_mat_context_vector_reduced2init_centered' ...
                );

            save(name_dec_transfer_mat, 'dec_transfer_mat_init2reduced', 'dec_transfer_mat_init_centered2reduced');
        end
    else
        %         name_emb_mat = sprintf('%sTEST%05d_syn_dec_emb_by_layer', path_model, 1);
        %         load(name_emb_mat); % variable name = dec_output_by_layer_mat
        %
        %         if strcmp(model_type, 'fastspeech')
        %             %dim_emb = size(dec_output_by_layer_mat, 1);
        %             nbr_layers = size(dec_output_by_layer_mat, 3);
        %         elseif strcmp(model_type, 'tacotron')
        %             %dim_emb = 512;
        %             nbr_layers  = length(dec_output_by_layer_mat);
        %         end
        %
        %         %all_dec_emb_mat = single(zeros(nbr_char_in_corpus, dim_emb, nbr_layers));
        %         all_dec_emb_mat = cell(nbr_layers, 1);
        %         all_dec_emb_mat_residual = cell(nbr_layers, 1);
        %         all_dec_emb_mat_context_vector = cell(nbr_layers, 1);
        % %         for i_layer = 1:nbr_layers
        % %             dim_emb = size(dec_output_by_layer_mat{i_layer}, 1);
        % %             all_dec_emb_mat{i_layer} = single(zeros(nbr_char_in_corpus, dim_emb));
        % %
        % % %             all_dec_emb_mat_residual{i_layer} = single(zeros(nbr_char_in_corpus, tacotron_residual_dim(i_layer)));
        % % %             all_dec_emb_mat_context_vector{i_layer} = single(zeros(nbr_char_in_corpus, tacotron_context_vec_dim(i_layer)));
        % %         end
        %
        %         for i_layer = 1:nbr_layers
        %             if strcmp(model_type, 'fastspeech')
        %                 dim_emb = size(dec_output_by_layer_mat, 1);
        %                 all_dec_emb_mat_residual_current_layer = single(zeros(nbr_char_in_corpus, 256));
        %                 all_dec_emb_mat_context_vector_current_layer = single(zeros(nbr_char_in_corpus, 256));
        %             elseif strcmp(model_type, 'tacotron')
        %                 dim_emb = size(dec_output_by_layer_mat{i_layer}, 1);
        %                 all_dec_emb_mat_residual_current_layer = single(zeros(nbr_char_in_corpus, tacotron_residual_dim(i_layer)));
        %                 all_dec_emb_mat_context_vector_current_layer = single(zeros(nbr_char_in_corpus, tacotron_context_vec_dim(i_layer)));
        %             end
        %
        %             all_dec_emb_mat_current_layer = single(zeros(nbr_char_in_corpus, dim_emb));
        %
        %             index_char = 0;
        %             for i_utt = 1:nbr_utt
        %                 fprintf('Decoder Embeddings | Layer: %d/%d | Utt %d/%d\n', i_layer, nbr_layers, i_utt, nbr_utt);
        %
        %                 if strcmp(model_type, 'fastspeech')
        %                     name_emb_mat = sprintf('%sTEST%05d_syn_dec_emb_by_layer', path_model, i_utt);
        %                     load(name_emb_mat); % variable name = dec_output_by_layer_mat
        %
        % %                     temp_dec_output_by_layer_mat = cell(nbr_layers, 1);
        % %                     for i_layer = 1:nbr_layers
        % %                         temp_dec_output_by_layer_mat{i_layer} = dec_output_by_layer_mat(:, :, i_layer);
        % %                     end
        % %                     dec_output_by_layer_mat = temp_dec_output_by_layer_mat;
        %
        %                     dec_output_current_layer = dec_output_by_layer_mat(:, :, i_layer);
        %                 elseif strcmp(model_type, 'tacotron')
        %                     % Load last encoder stat
        %                     name_emb_mat = sprintf('%sTEST%05d_syn_dec_emb_by_layer', path_model, i_utt);
        %                     load(name_emb_mat); % variable name = dec_output_by_layer_mat
        %
        %     %                 dec_output_by_layer = cell(nbr_layers, 1);
        % %                     for i_layer = 1:nbr_layers
        % %                         dec_output_current_layer = kron(dec_output_by_layer_mat{i_layer}, ones(1,2));
        % %                         dec_output_by_layer_mat{i_layer} = dec_output_current_layer;
        % %                     end
        %                     dec_output_current_layer = kron(dec_output_by_layer_mat{i_layer}, ones(1,2));
        %                 end
        %     %             nbr_frames = size(dec_output_by_layer_mat, 2);
        %                 nbr_frames = size(dec_output_current_layer, 2);
        %
        %                 % Get mean_emb of frames by char
        %                 name_seg_file = sprintf('%sTEST%05d_seg.csv', path_model, i_utt);
        %                 fid = fopen(name_seg_file, 'r');
        %                 S = textscan(fid, "%s %f %f %f %f %f", 'Delimiter','\t', 'headerlines', 1);
        %                 fclose(fid);
        %
        %                 nbr_char_in_utt = length(S{1});
        %                 for i_char = 1:nbr_char_in_utt
        %                     index_char = index_char + 1;
        %
        %                     first_frame = max(round((22050/256)*S{2}(i_char)/1000)+1, 1);
        %                     last_frame = min(round((22050/256)*S{3}(i_char)/1000), nbr_frames);
        %
        %                     first_third = min(first_frame + floor((last_frame - first_frame + 1)/3), nbr_frames);
        %                     second_third = min(first_frame + ceil((last_frame - first_frame + 1)*2/3), nbr_frames);
        %
        %                     if second_third < first_third
        %                         continue;
        %                     end
        %
        %                     permute_dec_output_by_layer_mat = dec_output_current_layer';
        %                     frames_current_char = permute_dec_output_by_layer_mat(first_third:second_third, :);
        %                     mean_frames_current_char = mean(frames_current_char, 1);
        %
        %                     all_dec_emb_mat_current_layer(index_char, :) = mean_frames_current_char;
        % %                     all_dec_emb_mat{i_layer} = all_dec_emb_mat_current_layer;
        %
        %                     if strcmp(model_type, 'tacotron')
        % %                         all_dec_emb_mat_residual_current_layer = all_dec_emb_mat_residual{i_layer};
        % %                         all_dec_emb_mat_context_vector_current_layer = all_dec_emb_mat_context_vector{i_layer};
        % %
        %                         all_dec_emb_mat_residual_current_layer(index_char, :) = mean_frames_current_char(1:tacotron_residual_dim(i_layer));
        % %                         all_dec_emb_mat_residual{i_layer} = all_dec_emb_mat_residual_current_layer;
        % %
        %                         all_dec_emb_mat_context_vector_current_layer(index_char, :) = mean_frames_current_char(tacotron_residual_dim(i_layer)+1:end);
        % %                         all_dec_emb_mat_context_vector{i_layer} = all_dec_emb_mat_context_vector_current_layer;
        %                     end
        %                 end
        %             end
        %             all_dec_emb_mat{i_layer} = all_dec_emb_mat_current_layer;
        %             all_dec_emb_mat_residual{i_layer} = all_dec_emb_mat_residual_current_layer;
        %             all_dec_emb_mat_context_vector{i_layer} = all_dec_emb_mat_context_vector_current_layer;
        %         end
        [all_dec_emb_mat, all_dec_emb_mat_residual, all_dec_emb_mat_context_vector] = load_decoder_embeddings_by_layer(path_model, model_type, nbr_utt, nbr_char_in_corpus);
        nbr_layers = length(all_dec_emb_mat);

        % Reduce Dimension (MDS on vowels only)
        random_considered_phon_indexes = randperm(length(considered_phon_indexes), round(part_random_phon_for_mds*length(considered_phon_indexes)));
        %         all_vowel_emb_dec = all_dec_emb_mat(vowel_indexes(random_vowel_indexes), :, :);

        all_dec_emb_mat_reduced = cell(nbr_layers, 1);
        all_dec_emb_mat_residual_reduced = cell(nbr_layers, 1);
        all_dec_emb_mat_context_vector_reduced = cell(nbr_layers, 1);
        all_dec_emb_mat_reduced_centered = cell(nbr_layers, 1);
        all_dec_emb_mat_residual_reduced_centered = cell(nbr_layers, 1);
        all_dec_emb_mat_context_vector_reduced_centered = cell(nbr_layers, 1);

        dec_transfer_mat_reduced2init = cell(nbr_layers, 1);
        dec_transfer_mat_residual_reduced2init = cell(nbr_layers, 1);
        dec_transfer_mat_context_vector_reduced2init = cell(nbr_layers, 1);
        dec_transfer_mat_reduced2init_centered = cell(nbr_layers, 1);
        dec_transfer_mat_residual_reduced2init_centered = cell(nbr_layers, 1);
        dec_transfer_mat_context_vector_reduced2init_centered = cell(nbr_layers, 1);

        dec_transfer_mat_init2reduced = cell(nbr_layers, 1);
        dec_transfer_mat_init_centered2reduced = cell(nbr_layers, 1);
        for i_layer = 1:nbr_layers
            % Complete embeddings
            all_vowel_emb_dec = all_dec_emb_mat{i_layer}(considered_phon_indexes(random_considered_phon_indexes), :);
            [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec', MDS_threshold, select_best_dim) ;
            all_dec_emb_mat_reduced{i_layer} = all_dec_emb_mat{i_layer}*W;
            dec_transfer_mat_reduced2init{i_layer} = X;
            dec_transfer_mat_init2reduced{i_layer} = W;

            % Same for centered embeddings
            all_dec_emb_mat_centered = center_emb_by_phon_label(all_dec_emb_mat{i_layer}, list_phones_in_corpus, 1:size(all_dec_emb_mat{i_layer}, 1));

            all_vowel_emb_dec_centered = all_dec_emb_mat_centered(considered_phon_indexes(random_considered_phon_indexes), :);
            [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec_centered', MDS_threshold, select_best_dim);
            all_dec_emb_mat_reduced_centered{i_layer} = all_dec_emb_mat_centered*W;
            dec_transfer_mat_reduced2init_centered{i_layer} = X;
            dec_transfer_mat_init_centered2reduced{i_layer} = W;

            if strcmp(model_type, 'tacotron')
                % residual embeddings
                all_vowel_emb_dec = all_dec_emb_mat_residual{i_layer}(considered_phon_indexes(random_considered_phon_indexes), :);
                [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec', MDS_threshold, select_best_dim) ;
                all_dec_emb_mat_residual_reduced{i_layer} = all_dec_emb_mat_residual{i_layer}*W;
                dec_transfer_mat_residual_reduced2init{i_layer} = X;

                % Same for centered Embeddings
                all_dec_emb_mat_residual_centered = center_emb_by_phon_label(all_dec_emb_mat_residual{i_layer}, list_phones_in_corpus, 1:size(all_dec_emb_mat_residual{i_layer}, 1));
                all_vowel_emb_dec_residual_centered = all_dec_emb_mat_residual_centered(considered_phon_indexes(random_considered_phon_indexes), :);
                [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec_residual_centered', MDS_threshold, select_best_dim);
                all_dec_emb_mat_residual_reduced_centered{i_layer} = all_dec_emb_mat_residual_centered*W;
                dec_transfer_mat_residual_reduced2init_centered{i_layer} = X;

                % context vector embeddings only
                all_vowel_emb_dec = all_dec_emb_mat_context_vector{i_layer}(considered_phon_indexes(random_considered_phon_indexes), :);
                [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec', MDS_threshold, select_best_dim) ;
                all_dec_emb_mat_context_vector_reduced{i_layer} = all_dec_emb_mat_context_vector{i_layer}*W;
                dec_transfer_mat_context_vector_reduced2init{i_layer} = X;

                % Same for centered Embeddings
                all_dec_emb_mat_context_vector_centered = center_emb_by_phon_label(all_dec_emb_mat_context_vector{i_layer}, list_phones_in_corpus, 1:size(all_dec_emb_mat_context_vector{i_layer}, 1));
                all_vowel_emb_dec_context_vector_centered = all_dec_emb_mat_context_vector_centered(considered_phon_indexes(random_considered_phon_indexes), :);
                [~, W, ~, X] = compute_coef_mds(all_vowel_emb_dec_context_vector_centered', MDS_threshold, select_best_dim);
                all_dec_emb_mat_context_vector_reduced_centered{i_layer} = all_dec_emb_mat_context_vector_centered*W;
                dec_transfer_mat_context_vector_reduced2init_centered{i_layer} = X;
            end
        end

        save( ...
            name_dec_emb_mat, ...
            'all_dec_emb_mat', ...
            'all_dec_emb_mat_residual', ...
            'all_dec_emb_mat_context_vector', ...
            'all_dec_emb_mat_reduced', ...
            'all_dec_emb_mat_residual_reduced', ...
            'all_dec_emb_mat_context_vector_reduced', ...
            'dec_transfer_mat_reduced2init', ...
            'dec_transfer_mat_residual_reduced2init', ...
            'dec_transfer_mat_context_vector_reduced2init', ...
            'all_dec_emb_mat_reduced_centered', ...
            'all_dec_emb_mat_residual_reduced_centered', ...
            'all_dec_emb_mat_context_vector_reduced_centered', ...
            'dec_transfer_mat_reduced2init_centered', ...
            'dec_transfer_mat_residual_reduced2init_centered', ...
            'dec_transfer_mat_context_vector_reduced2init_centered' ...
            );

        save(name_dec_transfer_mat, 'dec_transfer_mat_init2reduced', 'dec_transfer_mat_init_centered2reduced');
    end

    % Center embeddings
    corr_coef_by_layer_dec = zeros(nbr_acoustic_params, nbr_layers);
    corr_coef_by_layer_dec_centered = zeros(nbr_acoustic_params, nbr_layers);
    corr_coef_by_layer_dec_mean = zeros(nbr_acoustic_params, nbr_layers);
    %     corr_coef_by_layer_dec_residual = zeros(nbr_acoustic_params, nbr_layers);
    corr_coef_by_layer_dec_context_vector = zeros(nbr_acoustic_params, nbr_layers);
    corr_coef_by_layer_dec_context_vector_centered = zeros(nbr_acoustic_params, nbr_layers);
    corr_coef_by_layer_dec_context_vector_mean = zeros(nbr_acoustic_params, nbr_layers);

    dec_predicted_acoustic_params_mean_by_layer = zeros(length(considered_phon_indexes), nbr_acoustic_params, nbr_layers);
    dec_predicted_acoustic_params_centered_by_layer = zeros(length(considered_phon_indexes), nbr_acoustic_params, nbr_layers);
    dec_context_vector_predicted_acoustic_params_mean_by_layer = zeros(length(considered_phon_indexes), nbr_acoustic_params, nbr_layers);
    dec_context_vector_predicted_acoustic_params_centered_by_layer = zeros(length(considered_phon_indexes), nbr_acoustic_params, nbr_layers);

    bias_vector_by_layer_dec = cell(nbr_layers, 1);
    %     bias_vector_by_layer_dec_residual = cell(nbr_layers, 1);
    bias_vector_by_layer_dec_context_vector = cell(nbr_layers, 1);

    pred_coef_by_layer_dec = cell(nbr_layers, 1);
    constant_coef_by_layer_dec = cell(nbr_layers, 1);
    pred_coef_by_layer_dec_centered = cell(nbr_layers, 1);
    constant_coef_by_layer_dec_centered = cell(nbr_layers, 1);
    pred_coef_by_layer_dec_mean = cell(nbr_layers, 1);
    constant_coef_by_layer_dec_mean = cell(nbr_layers, 1);

    for i_layer = 1:nbr_layers
        fprintf('Compute Correlation for model %s, layer %d | DECODER\n', name_model, i_layer);

        % Complete embeddings
        %         if use_MDS
        %             considered_emb = all_dec_emb_mat_reduced{i_layer}(considered_phon_indexes, :);
        %         else
        %             considered_emb = all_dec_emb_mat{i_layer}(considered_phon_indexes, :);
        %         end
        %
        %         if center_emb
        %             centered_emb_by_layer = center_emb_by_phon_label(considered_emb, list_phones_in_corpus, considered_phon_indexes);
        %         else
        %             centered_emb_by_layer = double(considered_emb);
        %         end
        %         if use_MDS && center_emb
        %             emb_by_layer = all_dec_emb_mat_reduced_centered{i_layer}(considered_phon_indexes, :);
        %         elseif use_MDS && ~center_emb
        %             emb_by_layer = all_dec_emb_mat_reduced{i_layer}(considered_phon_indexes, :);
        %         elseif ~use_MDS && center_emb
        %             emb_by_layer = all_dec_emb_mat{i_layer}(considered_phon_indexes, :);
        %             emb_by_layer = center_emb_by_phon_label(emb_by_layer, list_phones_in_corpus, considered_phon_indexes);
        %         elseif ~use_MDS && ~center_emb
        %             emb_by_layer = all_dec_emb_mat{i_layer}(considered_phon_indexes, :);
        if use_MDS
            emb_by_layer = all_dec_emb_mat_reduced{i_layer}(considered_phon_indexes, :);
            emb_by_layer_centered = all_dec_emb_mat_reduced_centered{i_layer}(considered_phon_indexes, :);
        elseif ~use_MDS
            emb_by_layer = all_dec_emb_mat{i_layer}(considered_phon_indexes, :);
            emb_by_layer_centered = center_emb_by_phon_label(emb_by_layer, list_phones_in_corpus, considered_phon_indexes);
        else
            error('Conditions calcul embeddings impossible');
        end
        emb_by_layer = double(emb_by_layer);
        emb_by_layer_centered = double(emb_by_layer_centered);

        %         if center_emb
        %             acoustic_params_for_regression = considered_acoustic_params_centered;
        %         else
        %             acoustic_params_for_regression = considered_acoustic_params;
        %         end

        % Compute regression by layer
        [corr_coef, predicted_acoustic_params, all_beta, all_beta_normalized, all_constants, ~, ~, ~] = acoustic_regression_in_latent_space(emb_by_layer, considered_acoustic_params, selected_dim, reduce_model_regression);
        corr_coef_by_layer_dec(:, i_layer) = corr_coef';

        if flag_write_residual == 1 && ~(strcmp(model_type, 'tacotron') && i_layer == 1) % If not context vector of Tacotron2
            %% Write regression residual
            for i_param = list_params_supra_segmental
                fprintf('\t> %s (R^2 = %0.2f) - error: %0.2f\n',label_acoustic{i_param},corr_coef(i_param),rmse(predicted_acoustic_params(~isnan(considered_acoustic_params(:,i_param)),i_param),considered_acoustic_params(~isnan(considered_acoustic_params(:,i_param)),i_param)))
                write_residual_OP(res_path{i_param},predicted_acoustic_params(:,i_param),considered_acoustic_params(:,i_param),'none',label_acoustic{i_param},i_layer,name_model,'dec',corr_coef(i_param))
            end
        end


        % Compute regression by layer centered
        [corr_coef_centered, predicted_acoustic_params_centered, all_beta_centered, all_beta_normalized_centered, all_constants_centered, ~, ~, ~] = acoustic_regression_in_latent_space(emb_by_layer_centered, considered_acoustic_params_centered, selected_dim, reduce_model_regression);
        corr_coef_by_layer_dec_centered(:, i_layer) = corr_coef_centered';

        if flag_write_residual == 1 && ~(strcmp(model_type, 'tacotron') && i_layer == 1)
            %% Write regression residual
            for i_param = list_params_segmental
                fprintf('\t> %s (R^2 = %0.2f) - error: %0.2f\n',label_acoustic{i_param},corr_coef_centered(i_param),rmse(predicted_acoustic_params_centered(~isnan(considered_acoustic_params_centered(:,i_param)),i_param),considered_acoustic_params_centered(~isnan(considered_acoustic_params_centered(:,i_param)),i_param)))
                write_residual_OP(res_path{i_param},predicted_acoustic_params_centered(:,i_param),considered_acoustic_params_centered(:,i_param),'centred',label_acoustic{i_param},i_layer,name_model,'dec',corr_coef_centered(i_param))
            end
        end


        % Compute regression by layer mean
        [corr_coef_mean, predicted_acoustic_params_mean, all_beta_mean, all_beta_normalized_mean, all_constants_mean, ~, ~, ~] = acoustic_regression_in_latent_space(emb_by_layer, considered_acoustic_params_mean, selected_dim, reduce_model_regression);
        corr_coef_by_layer_dec_mean(:, i_layer) = corr_coef_mean';

        if flag_write_residual == 1 && ~(strcmp(model_type, 'tacotron') && i_layer == 1)
            %% Write regression residual
            for i_param = list_params_segmental
                fprintf('\t> %s (R^2 = %0.2f) - error: %0.2f\n',label_acoustic{i_param},corr_coef_mean(i_param),rmse(predicted_acoustic_params_mean(~isnan(considered_acoustic_params_mean(:,i_param)),i_param),considered_acoustic_params_mean(~isnan(considered_acoustic_params_mean(:,i_param)),i_param)))
                write_residual_OP(res_path{i_param},predicted_acoustic_params_mean(:,i_param),considered_acoustic_params_mean(:,i_param),'mean',label_acoustic{i_param},i_layer,name_model,'dec',corr_coef_mean(i_param))
            end
        end


        dec_predicted_acoustic_params_mean_by_layer(:, :, i_layer) = predicted_acoustic_params_mean;
        dec_predicted_acoustic_params_centered_by_layer(:, :, i_layer) = predicted_acoustic_params_centered;

        % Save predictors
        pred_coef_by_layer_dec{i_layer} = all_beta;
        constant_coef_by_layer_dec{i_layer} = all_constants;
        pred_coef_by_layer_dec_centered{i_layer} = all_beta_centered;
        constant_coef_by_layer_dec_centered{i_layer} = all_constants_centered;
        pred_coef_by_layer_dec_mean{i_layer} = all_beta_mean;
        constant_coef_by_layer_dec_mean{i_layer} = all_constants_mean;

        % Save bias vector by param by layer
        if use_MDS && center_emb
            bias_vector_current_layer = (all_beta_normalized_centered'*dec_transfer_mat_reduced2init_centered{i_layer})';
        elseif use_MDS && ~center_emb
            bias_vector_current_layer = (all_beta_normalized'*dec_transfer_mat_reduced2init{i_layer})';
        elseif ~use_MDS && center_emb
            bias_vector_current_layer = all_beta_normalized_centered;
        elseif ~use_MDS && ~center_emb
            bias_vector_current_layer = all_beta_normalized;
        end
        bias_vector_by_layer_dec{i_layer} = bias_vector_current_layer;

        if strcmp(model_type, 'tacotron')
            %             % residual embeddings
            %             if use_MDS
            %                 considered_emb = all_dec_emb_mat_residual_reduced{i_layer}(considered_phon_indexes, :);
            %             else
            %                 considered_emb = all_dec_emb_mat_residual{i_layer}(considered_phon_indexes, :);
            %             end
            %
            %             if center_emb
            %                 centered_emb_by_layer = center_emb_by_phon_label(considered_emb, list_phones_in_corpus, considered_phon_indexes);
            %             else
            %                 centered_emb_by_layer = double(considered_emb);
            %             end
            %
            %             % Compute regression by layer
            %             [corr_coef, ~, ~, all_beta_normalized, ~, ~, ~, ~] = acoustic_regression_in_latent_space(centered_emb_by_layer, considered_acoustic_params, selected_dim, reduce_model_regression);
            %             corr_coef_by_layer_dec_residual(:, i_layer) = corr_coef';
            %
            %             % Save bias vector by param by layer
            %             if use_MDS
            %                 bias_vector_current_layer = (all_beta_normalized'*dec_transfer_mat_residual_reduced2init{i_layer})';
            %                 bias_vector_by_layer_dec_residual{i_layer} = bias_vector_current_layer;
            %             else
            %                 bias_vector_by_layer_dec_residual{i_layer} = all_beta_normalized;
            %             end

            % context_vector embeddings only
            %             if use_MDS
            %                 considered_emb = all_dec_emb_mat_context_vector_reduced{i_layer}(considered_phon_indexes, :);
            %             else
            %                 considered_emb = all_dec_emb_mat_context_vector{i_layer}(considered_phon_indexes, :);
            %             end
            %
            %             if center_emb
            %                 centered_emb_by_layer = center_emb_by_phon_label(considered_emb, list_phones_in_corpus, considered_phon_indexes);
            %             else
            %                 centered_emb_by_layer = double(considered_emb);
            %             end
            %             if use_MDS && center_emb
            %                 emb_by_layer = all_dec_emb_mat_context_vector_reduced_centered{i_layer}(considered_phon_indexes, :);
            %             elseif use_MDS && ~center_emb
            %                 emb_by_layer = all_dec_emb_mat_context_vector_reduced{i_layer}(considered_phon_indexes, :);
            %             elseif ~use_MDS && center_emb
            %                 emb_by_layer = all_dec_emb_mat_context_vector{i_layer}(considered_phon_indexes, :);
            %                 emb_by_layer = center_emb_by_phon_label(emb_by_layer, list_phones_in_corpus, considered_phon_indexes);
            %             elseif ~use_MDS && ~center_emb
            %                 emb_by_layer = all_dec_emb_mat_context_vector{i_layer}(considered_phon_indexes, :);
            if use_MDS
                emb_by_layer = all_dec_emb_mat_context_vector_reduced{i_layer}(considered_phon_indexes, :);
                emb_by_layer_centered = all_dec_emb_mat_context_vector_reduced_centered{i_layer}(considered_phon_indexes, :);
            elseif ~use_MDS
                emb_by_layer = all_dec_emb_mat_context_vector{i_layer}(considered_phon_indexes, :);
                emb_by_layer_centered = center_emb_by_phon_label(emb_by_layer, list_phones_in_corpus, considered_phon_indexes);
            else
                error('Conditions calcul embeddings impossible');
            end
            emb_by_layer = double(emb_by_layer);
            emb_by_layer_centered = double(emb_by_layer_centered);

            %             if center_emb
            %                 acoustic_params_for_regression = considered_acoustic_params_centered;
            %             else
            %                 acoustic_params_for_regression = considered_acoustic_params;
            %             end

            % Compute regression by layer
            [corr_coef, predicted_acoustic_params, ~, all_beta_normalized, ~, ~, ~, ~] = acoustic_regression_in_latent_space(emb_by_layer, considered_acoustic_params, selected_dim, reduce_model_regression);
            corr_coef_by_layer_dec_context_vector(:, i_layer) = corr_coef';

            if flag_write_residual == 1 && i_layer == 1
                %% Write regression residual
                for i_param = list_params_supra_segmental
                    fprintf('\t> %s (R^2 = %0.2f) - error: %0.2f\n',label_acoustic{i_param},corr_coef(i_param),rmse(predicted_acoustic_params(~isnan(considered_acoustic_params(:,i_param)),i_param),considered_acoustic_params(~isnan(considered_acoustic_params(:,i_param)),i_param)))
                    write_residual_OP(res_path{i_param},predicted_acoustic_params(:,i_param),considered_acoustic_params(:,i_param),'none',label_acoustic{i_param},i_layer,name_model,'dec',corr_coef(i_param))
                end
            end


            % Compute regression by layer centered
            [corr_coef_centered, predicted_acoustic_params_centered, ~, all_beta_normalized_centered, ~, ~, ~, ~] = acoustic_regression_in_latent_space(emb_by_layer_centered, considered_acoustic_params_centered, selected_dim, reduce_model_regression);
            corr_coef_by_layer_dec_context_vector_centered(:, i_layer) = corr_coef_centered';

            if flag_write_residual == 1 && i_layer == 1
                %% Write regression residual
                for i_param = list_params_segmental
                    fprintf('\t> %s (R^2 = %0.2f) - error: %0.2f\n',label_acoustic{i_param},corr_coef_centered(i_param),rmse(predicted_acoustic_params_centered(~isnan(considered_acoustic_params_centered(:,i_param)),i_param),considered_acoustic_params_centered(~isnan(considered_acoustic_params_centered(:,i_param)),i_param)))
                    write_residual_OP(res_path{i_param},predicted_acoustic_params_centered(:,i_param),considered_acoustic_params_centered(:,i_param),'centred',label_acoustic{i_param},i_layer,name_model,'dec',corr_coef_centered(i_param))
                end
            end


            % Compute regression by layer mean
            [corr_coef_mean, predicted_acoustic_params_mean, ~, all_beta_normalized_mean, ~, ~, ~, ~] = acoustic_regression_in_latent_space(emb_by_layer, considered_acoustic_params_mean, selected_dim, reduce_model_regression);
            corr_coef_by_layer_dec_context_vector_mean(:, i_layer) = corr_coef_mean';

            if flag_write_residual == 1 && i_layer == 1
                %% Write regression residual
                for i_param = list_params_segmental
                    fprintf('\t> %s (R^2 = %0.2f) - error: %0.2f\n',label_acoustic{i_param},corr_coef_mean(i_param),rmse(predicted_acoustic_params_mean(~isnan(considered_acoustic_params_mean(:,i_param)),i_param),considered_acoustic_params_mean(~isnan(considered_acoustic_params_mean(:,i_param)),i_param)))
                    write_residual_OP(res_path{i_param},predicted_acoustic_params_mean(:,i_param),considered_acoustic_params_mean(:,i_param),'mean',label_acoustic{i_param},i_layer,name_model,'dec',corr_coef_mean(i_param))
                end
            end


            dec_context_vector_predicted_acoustic_params_mean_by_layer(:, :, i_layer) = predicted_acoustic_params_mean;
            dec_context_vector_predicted_acoustic_params_centered_by_layer(:, :, i_layer) = predicted_acoustic_params_centered;

            % Save bias vector by param by layer
            if use_MDS && center_emb
                bias_vector_current_layer = (all_beta_normalized_centered'*dec_transfer_mat_context_vector_reduced2init_centered{i_layer})';
            elseif use_MDS && ~center_emb
                bias_vector_current_layer = (all_beta_normalized'*dec_transfer_mat_context_vector_reduced2init{i_layer})';
            elseif ~use_MDS && center_emb
                bias_vector_current_layer = all_beta_normalized_centered;
            elseif ~use_MDS && ~center_emb
                bias_vector_current_layer = all_beta_normalized;
            end
            bias_vector_by_layer_dec_context_vector{i_layer} = bias_vector_current_layer;
        end
    end

    % If Tacotron, use context vector as first decoder layer (prenet
    % unusable)
    if strcmp(model_type, 'tacotron')
        corr_coef_by_layer_dec(:,1) = corr_coef_by_layer_dec_context_vector(:, 2);
        corr_coef_by_layer_dec_centered(:,1) = corr_coef_by_layer_dec_context_vector_centered(:, 2);
        corr_coef_by_layer_dec_mean(:,1) = corr_coef_by_layer_dec_context_vector_mean(:, 2);
        bias_vector_by_layer_dec{1} = bias_vector_by_layer_dec_context_vector{2};

        dec_predicted_acoustic_params_mean_by_layer(:, :, 1) = dec_context_vector_predicted_acoustic_params_mean_by_layer(:, :, 2);
        dec_predicted_acoustic_params_centered_by_layer(:, :, 1) = dec_context_vector_predicted_acoustic_params_centered_by_layer(:, :, 2);
    end
    % ---------------------- FIN DECODER ---------------------------

    save(name_dec_predictor_mat, 'pred_coef_by_layer_dec', 'constant_coef_by_layer_dec', 'pred_coef_by_layer_dec_centered', 'constant_coef_by_layer_dec_centered', 'pred_coef_by_layer_dec_mean', 'constant_coef_by_layer_dec_mean');

    %     % ---------------------- MEL ---------------------------
    %     % Load embeddings by layer
    %     fprintf('Loading Mel | Model: %s\n', name_model);
    %     if exist([name_mel_mat '.mat']) && ~reload_mel
    %         load(name_mel_mat);
    %
    %         dim_emb = size(all_mel_mat, 1);
    %         nbr_layers = size(all_mel_mat, 3);
    %     else
    %         name_emb_mat = sprintf('%sTEST%05d_syn_mel_by_layer', path_model, 1);
    %         load(name_emb_mat); % variable name = mel_output_by_layer_mat
    %
    %         dim_emb = size(mel_output_by_layer_mat, 1);
    %         nbr_layers = size(mel_output_by_layer_mat, 3);
    %
    %         all_mel_mat = single(zeros(nbr_char_in_corpus, dim_emb, nbr_layers));
    %         index_char = 0;
    %         for i_utt = 1:nbr_utt
    %             fprintf('Mel Frames | Utt %d/%d\n', i_utt, nbr_utt);
    %             name_emb_mat = sprintf('%sTEST%05d_syn_mel_by_layer', path_model, i_utt);
    %             load(name_emb_mat); % variable name = mel_output_by_layer_mat
    %             nbr_frames = size(mel_output_by_layer_mat, 2);
    %
    %             % Get mean_mel of frames by char
    %             name_seg_file = sprintf('%sTEST%05d_seg.csv', path_model, i_utt);
    %             fid = fopen(name_seg_file, 'r');
    %             S = textscan(fid, "%s %f %f %f %f %f", 'Delimiter','\t', 'headerlines', 1);
    %             fclose(fid);
    %
    %             nbr_char_in_utt = length(S{1});
    %             for i_char = 1:nbr_char_in_utt
    %                 index_char = index_char + 1;
    %
    %                 first_frame = max(round((22050/256)*S{2}(i_char)/1000)+1, 1);
    %                 last_frame = min(round((22050/256)*S{3}(i_char)/1000), nbr_frames);
    %
    %                 first_third = min(first_frame + floor((last_frame - first_frame + 1)/3), nbr_frames);
    %                 second_third = min(first_frame + ceil((last_frame - first_frame + 1)*2/3), nbr_frames);
    %
    %                 if second_third < first_third
    %                     continue;
    %                 end
    %
    %                 permute_mel_output_by_layer_mat = permute(mel_output_by_layer_mat, [2, 1, 3]);
    %                 frames_current_char = permute_mel_output_by_layer_mat(first_third:second_third, :, :);
    %                 mean_frames_current_char = mean(frames_current_char, 1);
    %
    %                 all_mel_mat(index_char, :, :) = mean_frames_current_char;
    %             end
    %         end
    %         save(name_mel_mat, 'all_mel_mat');
    %     end
    %
    %     % Center embeddings
    nbr_layers = 2;
    corr_coef_by_layer_mel = zeros(nbr_acoustic_params, nbr_layers);
    corr_coef_by_layer_mel_centered = zeros(nbr_acoustic_params, nbr_layers);
    corr_coef_by_layer_mel_mean = zeros(nbr_acoustic_params, nbr_layers);
    bias_vector_by_layer_mel = cell(nbr_layers, 1);
    %     considered_emb = all_mel_mat(considered_phon_indexes, :, :);
    %     for i_layer = 1:nbr_layers
    %         fprintf('Compute Correlation for model %s, layer %d | MEL\n', name_model, i_layer);
    %
    %         if center_emb
    %             centered_emb_by_layer = center_emb_by_phon_label(considered_emb(:, :, i_layer), list_phones_in_corpus, considered_phon_indexes);
    %         else
    %             centered_emb_by_layer = double(considered_emb(:, :, i_layer));
    %         end
    %
    %         % Compute regression by layer
    %         [corr_coef, ~, ~, all_beta_normalized, ~, ~, ~, ~] = acoustic_regression_in_latent_space(centered_emb_by_layer, considered_acoustic_params, selected_dim, reduce_model_regression);
    %         corr_coef_by_layer_mel(:, i_layer) = corr_coef';
    %
    %         % Save bias vector by param by layer
    %         bias_vector_by_layer_mel{i_layer} = all_beta_normalized;
    %     end
    %     % ---------------------- FIN MEL ---------------------------

    % Concatenate bias vector by layer
    bias_vector_by_layer = [bias_vector_by_layer_enc; bias_vector_by_layer_dec; bias_vector_by_layer_mel];
    save(sprintf('bias_vector_by_layer_%s_%s_%s_%s_%s', name_model, reduced_name, center_emb_name, phon_type_name, reduced_regression), 'bias_vector_by_layer');

    % Concatenate results by models
    %     corr_coef_by_layer_by_model = cat(3, corr_coef_by_layer_by_model, corr_coef_by_layer);

    % Plot Correlation by layer for all models
    if strcmp(model_type, 'fastspeech')
        %         subplot(2, 2, i_model);
        dim_to_disp = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17];
    elseif strcmp(model_type, 'tacotron')
        %         subplot(1, 2, i_model);
        dim_to_disp = [1, 2, 3, 4, 5, 9, 10, 11];
    end
    %     p(1, i_model).select();
    total_corr_coef_by_layer = [corr_coef_by_layer_enc, corr_coef_by_layer_dec, corr_coef_by_layer_mel];
    total_corr_coef_by_layer_centered = [corr_coef_by_layer_enc_centered, corr_coef_by_layer_dec_centered, corr_coef_by_layer_mel_centered];
    total_corr_coef_by_layer_mean = [corr_coef_by_layer_enc_mean, corr_coef_by_layer_dec_mean, corr_coef_by_layer_mel_mean];
    % R² clearer
    %     total_corr_coef_by_layer = total_corr_coef_by_layer.^2;
    %     total_corr_coef_by_layer_centered = total_corr_coef_by_layer_centered.^2;
    %     total_corr_coef_by_layer_mean = total_corr_coef_by_layer_mean.^2;
    %     total_nbr_layers = size(total_corr_coef_by_layer, 2);

    predicted_params_mean_by_layer = cat(3, enc_predicted_acoustic_params_mean_by_layer, dec_predicted_acoustic_params_mean_by_layer);
    predicted_params_centered_by_layer = cat(3, enc_predicted_acoustic_params_centered_by_layer, dec_predicted_acoustic_params_centered_by_layer);

    %%

    % posPlot = plotPosition(1,4,[0.035 0.0 0.08 0.37],[0.02 0.01 0.00 0.01]);

    figure(2);

    % Supra-Segmental
    if plot_supra_segmental
        subplotMargin(posPlot,i_model);
        for index_param = list_params_supra_segmental
            current_x_axis = dim_to_disp(1:sum(dim_to_disp <= max_dim_by_param(index_param)));
            plot(2:4, 2:4, '-', 'LineWidth', 8, 'Color', legend_by_param{index_param, 2});
            hold on;
            plot(1:length(current_x_axis), total_corr_coef_by_layer(index_param, current_x_axis), '-', 'LineWidth', 2, 'Color', legend_by_param{index_param, 2},'HandleVisibility','off');
            hold on;
            %         plot(1:length(current_x_axis), total_corr_coef_by_layer_centered(index_param, current_x_axis), ':', 'LineWidth', 2, 'Color', legend_by_param{index_param, 2}, 'HandleVisibility','off');
            %         plot(1:length(current_x_axis), total_corr_coef_by_layer_mean(index_param, current_x_axis), '--', 'LineWidth', 2, 'Color', legend_by_param{index_param, 2}, 'HandleVisibility','off');
        end

        % Add events
        if strcmp(model_type, 'fastspeech')
            lim_x = [1 15];
            plot([8.5, 8.5], [0, 1.2], '--', 'color', 'black', 'HandleVisibility','off','LineWidth',2);
            text((8.5-lim_x(1))/2+lim_x(1), 1.1, 'Encoder', 'HorizontalAlignment','center');
            text((lim_x(2)-8.5)/2+8.5, 1.1, 'Decoder', 'HorizontalAlignment','center');
            xticks(1:length(dim_to_disp));
            xticklabels(name_layer_fastspeech(dim_to_disp));
            xtickangle(50)
            xlabel('\overrightarrow{Layers~Depth}', 'Interpreter','latex');
        elseif strcmp(model_type, 'tacotron')
            plot([5.5, 5.5], [0, 1.2], '--', 'color', 'black', 'HandleVisibility','off','LineWidth',2);
            text(3, 1.1, 'Encoder', 'HorizontalAlignment','center');
            text(7, 1.1, 'Decoder', 'HorizontalAlignment','center');
            xticks(1:length(dim_to_disp));
            xticklabels(name_layer_tacotron(dim_to_disp));
            xtickangle(40)
            lim_x = [1 8];
            xlabel('\overrightarrow{Layers~Depth}', 'Interpreter','latex');
        end

        xlim(lim_x)
        ylim([0, 1.2]);
        grid on;
        title(sprintf('%s', legend_model),'Interpreter','latex');
        yticks(0:0.2:1);
        if i_model == 1
            ylabel('Goodness of fit $R^2$', 'Interpreter','latex');
            yticklabels(0:0.2:1);
        end

        if i_model == 1
            if strcmp(model_type, 'fastspeech')
                % lgnd2 = legend([label_acoustic_plot(correlation_to_disp(list_params_supra_segmental)); {'delta'; 'mean'}], 'Location','southwest','interpreter','tex','Orientation','horizontal');
                lgnd2 = legend(label_acoustic_plot(correlation_to_disp(list_params_supra_segmental)), 'Location','southwest','interpreter','tex','Orientation','horizontal');
                %%
                lgnd2.Position(1) = 0.42;
                lgnd2.Position(2) = 0.02;
            else
                %%
                % lgnd2 = legend([label_acoustic(correlation_to_disp(list_params_supra_segmental)); {'delta'; 'mean'}], 'Location','southwest','interpreter','tex','Orientation','vertical');
                lgnd2 = legend([label_acoustic_plot(correlation_to_disp(list_params_supra_segmental))], 'Location','southwest','interpreter','tex','Orientation','vertical');
                lgnd2.Position(1) = 0.55;
                lgnd2.Position(2) = 0.47;
            end
        end

    end

    setPlotFonts(ptxt);
    %%
    figure(1);

    if plot_segmental
        subplotMargin(posPlot,i_model);

        for index_param = list_params_segmental
            current_x_axis = dim_to_disp(1:sum(dim_to_disp <= max_dim_by_param(index_param)));
            %         plot(1:length(current_x_axis), total_corr_coef_by_layer(index_param, current_x_axis), '-', 'LineWidth', 2, 'Color', legend_by_param{index_param, 2});
            plot(2:4, 2:4, '-', 'LineWidth', 8, 'Color', legend_by_param{index_param, 2});
            hold on;
            plot(1:length(current_x_axis), total_corr_coef_by_layer_centered(index_param, current_x_axis), ':', 'LineWidth', 2, 'Color', legend_by_param{index_param, 2}, 'HandleVisibility','off');
            plot(1:length(current_x_axis), total_corr_coef_by_layer_mean(index_param, current_x_axis), '--', 'LineWidth', 2, 'Color', legend_by_param{index_param, 2}, 'HandleVisibility','off');
        end
        plot(2, 2, 'k:', 'LineWidth', 2);
        plot(2, 2, 'k--', 'LineWidth', 2);


        % Add events
        if strcmp(model_type, 'fastspeech')
            plot([8.5, 8.5], [0, 1.2], '--', 'color', 'black', 'HandleVisibility','off','LineWidth',2);
            text((8.5-lim_x(1))/2+lim_x(1), 1.1, 'Encoder', 'HorizontalAlignment','center');
            text((lim_x(2)-8.5)/2+8.5, 1.1, 'Decoder', 'HorizontalAlignment','center');
            xticks(1:length(dim_to_disp));
            xticklabels(name_layer_fastspeech(dim_to_disp));
            xtickangle(50)
            lim_x = [1 15];
            xlabel('\overrightarrow{Layers~Depth}', 'Interpreter','latex');
        elseif strcmp(model_type, 'tacotron')
            plot([5.5, 5.5], [0, 1.2], '--', 'color', 'black', 'HandleVisibility','off','LineWidth',2);
            text(3, 1.1, 'Encoder', 'HorizontalAlignment','center');
            text(7, 1.1, 'Decoder', 'HorizontalAlignment','center');
            xticks(1:length(dim_to_disp));
            xticklabels(name_layer_tacotron(dim_to_disp));
            xtickangle(40)
            lim_x = [1 8];
            xlabel('\overrightarrow{Layers~Depth}', 'Interpreter','latex');
        end

        xlim(lim_x)
        ylim([0, 1.2]);
        grid on;
        title(sprintf('%s', legend_model),'Interpreter','latex');
        yticks(0:0.2:1);
        if i_model == 1
            ylabel('Goodness of fit $R^2$', 'Interpreter','latex');
            yticklabels(0:0.2:1);
        end

        % if i_model == 1
        %     if strcmp(model_type, 'fastspeech')
        %         %%
        %         lgnd1 = legend([label_acoustic_plot(correlation_to_disp(list_params_segmental)); {'delta'; 'mean'}], 'Location','southwest','interpreter','tex','Orientation','horizontal');
        %         lgnd1.Position(1) = 0.35;
        %         lgnd1.Position(2) = 0.02;
        %     else
        %         lgnd1 = legend([label_acoustic_plot(correlation_to_disp(list_params_segmental)); {'delta'; 'mean'}], 'Location','westoutside','interpreter','tex','Orientation','vertical');
        %         lgnd1.Position(1) = 0.55;
        %         lgnd1.Position(2) = 0.43;
        %     end
        %
        % end
        if i_model == nbr_models && strcmp(model_type, 'fastspeech')
            %% Multispeaker legend

            % Fake plots to add in legend
            index_param = [correlation_to_disp(list_params_segmental) correlation_to_disp(list_params_supra_segmental)];
            for i_param = correlation_to_disp(list_params_supra_segmental)
                plot(1:15, 10*ones(1,15),'LineWidth', 8, 'Color', legend_by_param{i_param, 2});
                hold on;
            end

            lgnd = legend([label_acoustic_plot(correlation_to_disp(list_params_segmental)); {'delta'; 'mean              |    '}; label_acoustic_plot(correlation_to_disp(list_params_supra_segmental))] , 'Location','southeast','interpreter','tex','Orientation','horizontal');
            lgnd.Position(1) = 0.245;
            lgnd.Position(2) = 0.02;

            % Patch to cover the | in the legend (that are used to impose space
            % between groups of legend)
            annotation('rectangle',[0.59,0.03,0.01,0.055],'FaceColor','white','LineStyle','none')
        end


    end

    setPlotFonts(ptxt);

    %     plot_predicted_acoustic_mean_VS_delta_in_embeddings(predicted_params_mean_by_layer, predicted_params_centered_by_layer, list_phones_in_corpus, considered_phon_indexes, layer_index, acoustic_params_index, label_acoustic, name_layer_fastspeech, name_model);
end

setPlotFonts(ptxt);

%%

if plot_segmental
    figure(1);
    if strcmp(model_type, 'tacotron')
    elseif strcmp(model_type, 'fastspeech')
    end
end

if plot_supra_segmental
    figure(2);
    if strcmp(model_type, 'tacotron')
    elseif strcmp(model_type, 'fastspeech')
    end
end

toc