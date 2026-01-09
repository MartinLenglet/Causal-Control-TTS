% Figure(1) in the code = Fig. 3 in the paper

clearvars;
% Fix rng
rng(1234);
list_vowels = {'a'; 'u'; 'i'; 'e'; 'e^'; 'y'; 'x'; 'x^'; 'o'; 'o^'; 'x~'; 'e~'; 'o~'; 'a~';};
% list_vowels = {'i'};
nbr_vowels = length(list_vowels);
correlation_to_disp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
% correlation_to_disp = [1];
% selected_acoustic_params = [1, 8, 11, 15, 18, 20, 21, 26, 34, 22, 35];
% selected_acoustic_params = selected_acoustic_params(correlation_to_disp);
% label_acoustic = {'Log(D)'; 'F0'; 'F1'; 'F2'; 'F3'; 'ST'; 'E'; 'RP'; 'Pfit'; 'CoG'; 'SB1k'};
% nbr_acoustic_params = length(selected_acoustic_params);
% selected_dim = [1, 2];
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
% };
% list_params_supra_segmental = [1, 2, 6, 7, 8, 9];
% list_params_segmental = [3, 4, 5, 10, 11];

% Reload
reload_corpus = false;
reload_encoder_emb = false;
reload_decoder_emb = false;
reload_duration_corpus = false;

limit_number_of_samples = 0;

reload_phonetic_prediction_enc = false;
reload_silences_prediction_enc = false;
reload_liaisons_prediction_enc = false;

reload_phonetic_prediction_dec = false;
reload_silences_prediction_dec = false;

reload_categorical_bias_vector = false;


silence_threshold = 5; % 5 for FS

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
    'Phon Emb';
    'Conv1';
%     'Conv2';
%     'Conv3';
    '2';
    '3';
    'Bi-LSTM';
    'Spk Emb';
    'fo Emb';
    'E Emb';
    'Context Vect';
    'LSTM1';
%     'LSTM2';
    '2';
    'Mel';
    'Mel+Post';
};

% Models/paths are configured centrally in config_paths.m
if ~exist('cfg', 'var')
    cfg = config_paths();
end
models = cfg.models_phonetic_pred_op;


nbr_models = length(models(:,1));


colors = [0    0.4470    0.7410; ...
    0.8500    0.3250    0.0980; ...
    0.9290    0.6940    0.1250; ...
    0.4940    0.1840    0.5560; ...
    0.4660    0.6740    0.1880; ...
    0.3010    0.7450    0.9330; ...
    0.6350    0.0780    0.1840];

plotSettings;

% Plot Correlation by layer for all models
figure(1);
clf;
fp = get(gcf,'Position');
fp(3:4) = [1250 460];
set(gcf,'PaperSize',[44 16])
set(gcf,'Position',fp)
set(gcf,'Renderer', 'painters');

posPlot = plotPosition(1,4,[0.035 0.0 0.1 0.28],[0.02 0.01 0.00 0.07]);

p = panel();
p.margin = [13 28 5 5];
p.pack(1, nbr_models);

set(groot, "defaultAxesTickLabelInterpreter", 'latex');
set(groot, "defaultTextInterpreter", 'latex');
set(groot, "defaultLegendInterpreter", 'latex');

ptxt.title.size = 26;
ptxt.label.size = 22;
if strcmp(model_type, 'fastspeech')
    ptxt.legend.size = 22;
else
    ptxt.legend.size = 18;
end
ptxt.axis.size = 17;
ptxt.text.size = 18;


for i_model = 1:nbr_models
    name_model = models{i_model, 1};
    path_model_train_bias = models{i_model, 2};
    path_csv_train_bias = models{i_model, 3};
    path_model_val = models{i_model, 4};
    path_csv_val = models{i_model, 5};
    model_type = models{i_model, 6};
    model_legend_name = models{i_model, 7};

    name_char_list_train_bias = sprintf('data/char_list_train_bias_%s', name_model);
    name_char_list_val = sprintf('data/char_list_val_%s', name_model);
    name_target_phone_list_train_bias = sprintf('data/target_phones_list_ortho_inputs_train_bias_%s', name_model);
    name_target_phone_list_val = sprintf('data/target_phones_list_ortho_inputs_val_%s', name_model);
    name_enc_emb_mat = sprintf('data/enc_emb_by_layer_%s_ortho', name_model);
    name_dec_emb_mat = sprintf('data/dec_emb_by_layer_%s_ortho', name_model);
    name_phonetic_pred_enc = sprintf('data/phonetic_pred_by_layer_enc_%s', name_model);
    name_phonetic_pred_dec = sprintf('data/phonetic_pred_by_layer_dec_%s', name_model);

    name_char_duration_list_train_bias = sprintf('data/char_duration_list_train_bias_%s', name_model);
    name_char_duration_list_val = sprintf('data/char_duration_list_val_%s', name_model);
    name_silences_pred_enc = sprintf('data/silences_pred_by_layer_enc_%s', name_model);
    name_categorical_bias = sprintf('data/categorical_bias_vector_by_layer_%s', name_model);
    name_silences_pred_dec = sprintf('data/silences_pred_by_layer_dec_%s', name_model);
    name_liaisons_pred_enc = sprintf('data/liaisons_pred_by_layer_enc_%s', name_model);
    name_liaisons_pred_dec = sprintf('data/liaisons_pred_by_layer_dec_%s', name_model);

    fid = fopen(path_csv_train_bias, 'r');
    T = textscan(fid, "%s %s %s %s %s", 'Delimiter','|');
    fclose(fid);
    fid = fopen(path_csv_val, 'r');
    V = textscan(fid, "%s %s %s %s %s", 'Delimiter','|');
    fclose(fid);

    nbr_utt_train_bias = length(T{1});
    nbr_utt_val = length(V{1});

    % get all phones in corpus
    fprintf('Loading All phones in corpus | Model: %s\n', name_model);
    if exist([name_char_list_train_bias '.mat']) && ...
        exist([name_char_list_val '.mat']) && ...
        exist([name_target_phone_list_train_bias '.mat']) && ...
        exist([name_target_phone_list_val '.mat']) && ...
        ~reload_corpus

        load(name_char_list_train_bias);
        load(name_char_list_val);
        load(name_target_phone_list_train_bias);
        load(name_target_phone_list_val);
    else
        [list_char_in_corpus_train_bias, nbr_char_in_corpus_train_bias] = get_list_phones_in_corpus(T{3}, false);
        save(name_char_list_train_bias, 'list_char_in_corpus_train_bias', 'nbr_char_in_corpus_train_bias');
        [list_char_in_corpus_val, nbr_char_in_corpus_val] = get_list_phones_in_corpus(V{3}, false);
        save(name_char_list_val, 'list_char_in_corpus_val', 'nbr_char_in_corpus_val');

        [list_phone_target_in_corpus_train_bias, nbr_phones_in_corpus_train_bias] = get_list_phones_in_corpus(T{5}, true);
        save(name_target_phone_list_train_bias, 'list_phone_target_in_corpus_train_bias', 'nbr_phones_in_corpus_train_bias');
        [list_phone_target_in_corpus_val, nbr_phones_in_corpus_val] = get_list_phones_in_corpus(V{5}, true);
        save(name_target_phone_list_val, 'list_phone_target_in_corpus_val', 'nbr_phones_in_corpus_val');
    end

    % get duration of all phones in corpus
    fprintf('Loading All phones duration in corpus | Model: %s\n', name_model);
    if exist([name_char_duration_list_train_bias '.mat']) && ...
        exist([name_char_duration_list_val '.mat']) && ...
        ~reload_duration_corpus

        load(name_char_duration_list_train_bias);
        load(name_char_duration_list_val);
    else
        list_char_duration_in_corpus_train_bias = get_list_duration_phones_in_corpus(path_model_train_bias, nbr_char_in_corpus_train_bias, nbr_utt_train_bias);
        save(name_char_duration_list_train_bias, 'list_char_duration_in_corpus_train_bias');
        list_char_duration_in_corpus_val = get_list_duration_phones_in_corpus(path_model_val, nbr_char_in_corpus_val, nbr_utt_val);
        save(name_char_duration_list_val, 'list_char_duration_in_corpus_val');
    end

    % Create full target phonetic cell array to train the classifier
    all_phonetic_labels_train_bias = order_phonetic_targets(list_phone_target_in_corpus_train_bias);
    all_phonetic_labels_val = order_phonetic_targets(list_phone_target_in_corpus_val);

    % Select potential silences patterns
    [all_potential_silences_indexes_train_bias, ~] = select_silences_patterns(path_csv_train_bias);
    [all_potential_silences_indexes_val, ~] = select_silences_patterns(path_csv_val);
    % Create silences labels
    all_silences_labels_train_bias = get_silences_labels(list_char_duration_in_corpus_train_bias(all_potential_silences_indexes_train_bias), silence_threshold);
    all_silences_labels_val = get_silences_labels(list_char_duration_in_corpus_val(all_potential_silences_indexes_val), silence_threshold);
    
    % Select potential liaisons patterns
    all_potential_liaisons_indexes_train_bias = select_liaisons_patterns(path_csv_train_bias);
    all_potential_liaisons_indexes_val = select_liaisons_patterns(path_csv_val);
    % Create liaisons labels
    all_liaisons_labels_train_bias = get_liaisons_labels(list_char_duration_in_corpus_train_bias(all_potential_liaisons_indexes_train_bias), silence_threshold);
    all_liaisons_labels_val = get_liaisons_labels(list_char_duration_in_corpus_val(all_potential_liaisons_indexes_val), silence_threshold);
    
    % Get indexes of non-null duration chars
    all_indexes_non_null_duration_char_train_bias = find(list_char_duration_in_corpus_train_bias >= silence_threshold);
    all_indexes_non_null_duration_char_val = find(list_char_duration_in_corpus_val >= silence_threshold);

    % Load embeddings by layer: ENCODER
    fprintf('Loading All embeddings in ENCODER | Model: %s\n', name_model);
    if exist([name_enc_emb_mat '.mat']) && ~reload_encoder_emb
        load(name_enc_emb_mat);
    else
        all_enc_emb_mat_train_bias = load_encoder_embeddings_by_layer(path_model_train_bias, nbr_utt_train_bias, nbr_char_in_corpus_train_bias);
        all_enc_emb_mat_val = load_encoder_embeddings_by_layer(path_model_val, nbr_utt_val, nbr_char_in_corpus_val);

        save(name_enc_emb_mat, 'all_enc_emb_mat_train_bias', 'all_enc_emb_mat_val');
    end
    nbr_layers = size(all_enc_emb_mat_train_bias, 3);

    % Compute Phonetic prediction with SVM
    if exist([name_phonetic_pred_enc '.mat']) && ~reload_phonetic_prediction_enc
        load(name_phonetic_pred_enc);
    else
        if limit_number_of_samples > 0
            phonetic_pred_by_layer_enc = cell(limit_number_of_samples, nbr_layers);
        else
            phonetic_pred_by_layer_enc = cell(nbr_char_in_corpus_val, nbr_layers);
        end
        for i_layer = 1:nbr_layers
            fprintf('Compute Phonetic Prediction for model %s, layer %d/%d | ENCODER\n', name_model, i_layer, nbr_layers);
    
            if limit_number_of_samples > 0
                considered_enc_emb_train_bias = double(all_enc_emb_mat_train_bias(1:limit_number_of_samples, :, i_layer));
                considered_enc_emb_val = double(all_enc_emb_mat_val(1:limit_number_of_samples, :, i_layer));
                considered_phonetic_labels_train_bias = all_phonetic_labels_train_bias(1:limit_number_of_samples);
            else
                considered_enc_emb_train_bias = double(all_enc_emb_mat_train_bias(:, :, i_layer));
                considered_enc_emb_val = double(all_enc_emb_mat_val(:, :, i_layer));
                considered_phonetic_labels_train_bias = all_phonetic_labels_train_bias;
            end
            Mdl = fitcdiscr(considered_enc_emb_train_bias, considered_phonetic_labels_train_bias);
            predicted_label = predict(Mdl,considered_enc_emb_val);
            phonetic_pred_by_layer_enc(:, i_layer) = predicted_label;
        end
        save(name_phonetic_pred_enc, 'phonetic_pred_by_layer_enc');
    end

    % Compute Phonetic prediction accuracy by layer
    all_accuracy_by_layer_enc = zeros(1, nbr_layers);
    phon_accuracy_by_layer_enc = zeros(1, nbr_layers);
    vowels_accuracy_by_layer_enc = zeros(1, nbr_layers);
    consonants_accuracy_by_layer_enc = zeros(1, nbr_layers);
    for i_layer = 1:nbr_layers
        if limit_number_of_samples > 0
            [all_accuracy, phon_accuracy, weighted_all_f1, weighted_phone_f1, weighted_vowels_f1, weighted_consonants_f1, f1_by_class] = compute_classification_prediction_accuracy(all_phonetic_labels_val(1:limit_number_of_samples), phonetic_pred_by_layer_enc(:, i_layer), false);
        else
            [all_accuracy, phon_accuracy, weighted_all_f1, weighted_phone_f1, weighted_vowels_f1, weighted_consonants_f1, f1_by_class] = compute_classification_prediction_accuracy(all_phonetic_labels_val, phonetic_pred_by_layer_enc(:, i_layer), false);
        end
%         all_accuracy_by_layer_enc(i_layer) = all_accuracy;
%         phon_accuracy_by_layer_enc(i_layer) = phon_accuracy;
        all_accuracy_by_layer_enc(i_layer) = weighted_all_f1;
        phon_accuracy_by_layer_enc(i_layer) = weighted_phone_f1;
        vowels_accuracy_by_layer_enc(i_layer) = weighted_vowels_f1;
        consonants_accuracy_by_layer_enc(i_layer) = weighted_consonants_f1;
    end

    % Compute Silences prediction with SVM
    bias_vector_by_layer = cell(nbr_layers, 1);
    if exist([name_silences_pred_enc '.mat']) && exist([name_categorical_bias '.mat']) && ~reload_silences_prediction_enc && ~reload_categorical_bias_vector
        load(name_silences_pred_enc);
        load(name_categorical_bias);
    else
        if limit_number_of_samples > 0
            silences_pred_by_layer_enc = cell(limit_number_of_samples, nbr_layers);
        else
            silences_pred_by_layer_enc = cell(length(all_potential_silences_indexes_val), nbr_layers);
        end
        coord_enc_emb_train_bias = cell(nbr_layers, 1);
        classifier_by_layer = cell(nbr_layers, 1);
        for i_layer = 1:nbr_layers
            fprintf('Compute Silences Prediction for model %s, layer %d/%d | ENCODER\n', name_model, i_layer, nbr_layers);
    
            if limit_number_of_samples > 0
                considered_enc_emb_train_bias = double(all_enc_emb_mat_train_bias(all_potential_silences_indexes_train_bias(1:limit_number_of_samples), :, i_layer));
                considered_enc_emb_val = double(all_enc_emb_mat_val(all_potential_silences_indexes_val(1:limit_number_of_samples), :, i_layer));
                considered_silences_labels_train_bias = all_silences_labels_train_bias(1:limit_number_of_samples);
            else
                considered_enc_emb_train_bias = double(all_enc_emb_mat_train_bias(all_potential_silences_indexes_train_bias, :, i_layer));
                considered_enc_emb_val = double(all_enc_emb_mat_val(all_potential_silences_indexes_val, :, i_layer));
                considered_silences_labels_train_bias = all_silences_labels_train_bias;
            end

%             Mdl = fitcdiscr(considered_enc_emb_train_bias, considered_silences_labels_train_bias);
            Mdl = fitcdiscr(considered_enc_emb_train_bias, considered_silences_labels_train_bias, 'discrimType', 'pseudoLinear');

            classifier_by_layer{i_layer} = Mdl;

            predicted_label = predict(Mdl,considered_enc_emb_val);
            silences_pred_by_layer_enc(:, i_layer) = predicted_label;
            [lambda, W, Z] = get_reduced_dimensions_from_lda(Mdl, considered_enc_emb_train_bias);

            % Bias Control
            considered_enc_emb_train_bias_reduced = considered_enc_emb_train_bias*W(:,1);
            center_non_silence = mean(considered_enc_emb_train_bias_reduced(matches(considered_silences_labels_train_bias,'Non-Silence')));
            center_silence = mean(considered_enc_emb_train_bias_reduced(matches(considered_silences_labels_train_bias,'Silence')));

            bias_vector_silence = (center_silence-center_non_silence) * (Z(1,:)');

            bias_vector_by_layer{i_layer} = bias_vector_silence;
            coord_enc_emb_train_bias{i_layer} = considered_enc_emb_train_bias;

            if i_layer == 6
                figure(10);
                clf;
                histogram(considered_enc_emb_train_bias_reduced(matches(considered_silences_labels_train_bias,'Non-Silence')), -23678:0.2:-23661, 'FaceColor', 'blue');
                hold on;
                histogram(considered_enc_emb_train_bias_reduced(matches(considered_silences_labels_train_bias,'Silence')), -23678:1:-23661, 'FaceColor', 'red');
                xlabel('$c_{1}$');
                ylabel('\# of embeddings');
                grid on;
                legend({'muted character'; 'silence'});
            end
        end
        save(name_silences_pred_enc, 'silences_pred_by_layer_enc', 'coord_enc_emb_train_bias', 'classifier_by_layer');
    end

    % Compute Silence prediction accuracy by layer
    silence_accuracy_by_layer_enc = zeros(1, nbr_layers);
    for i_layer = 1:nbr_layers
        if limit_number_of_samples > 0
            [~, ~, silence_accuracy, ~, ~, ~, f1_by_class] = compute_classification_prediction_accuracy(all_silences_labels_val(1:limit_number_of_samples), silences_pred_by_layer_enc(:, i_layer), true);
        else
            [~, ~, silence_accuracy, ~, ~, ~, f1_by_class] = compute_classification_prediction_accuracy(all_silences_labels_val, silences_pred_by_layer_enc(:, i_layer), true);
        end
%         silence_accuracy_by_layer_enc(i_layer) = silence_accuracy;
        silence_accuracy_by_layer_enc(i_layer) = f1_by_class(2, 3);
    end

    % Compute Liaisons prediction with SVM
    if exist([name_liaisons_pred_enc '.mat']) && ~reload_liaisons_prediction_enc && ~reload_categorical_bias_vector
        load(name_liaisons_pred_enc);
    else
        if limit_number_of_samples > 0
            liaisons_pred_by_layer_enc = cell(limit_number_of_samples, nbr_layers);
        else
            liaisons_pred_by_layer_enc = cell(length(all_potential_liaisons_indexes_val), nbr_layers);
        end
        coord_enc_emb_train_bias = cell(nbr_layers, 1);
        classifier_by_layer = cell(nbr_layers, 1);
        for i_layer = 1:nbr_layers
            fprintf('Compute Liaisons Prediction for model %s, layer %d/%d | ENCODER\n', name_model, i_layer, nbr_layers);
    
            if limit_number_of_samples > 0
                considered_enc_emb_train_bias = double(all_enc_emb_mat_train_bias(all_potential_liaisons_indexes_train_bias(1:limit_number_of_samples), :, i_layer));
                considered_enc_emb_val = double(all_enc_emb_mat_val(all_potential_liaisons_indexes_val(1:limit_number_of_samples), :, i_layer));
                considered_liaisons_labels_train_bias = all_liaisons_labels_train_bias(1:limit_number_of_samples);
            else
                considered_enc_emb_train_bias = double(all_enc_emb_mat_train_bias(all_potential_liaisons_indexes_train_bias, :, i_layer));
                considered_enc_emb_val = double(all_enc_emb_mat_val(all_potential_liaisons_indexes_val, :, i_layer));
                considered_liaisons_labels_train_bias = all_liaisons_labels_train_bias;
            end
%             Mdl = fitcdiscr(considered_enc_emb_train_bias, considered_liaisons_labels_train_bias);
            Mdl = fitcdiscr(considered_enc_emb_train_bias, considered_liaisons_labels_train_bias, 'discrimType', 'pseudoLinear');
            
            classifier_by_layer{i_layer} = Mdl;

            predicted_label = predict(Mdl, considered_enc_emb_val);
            [lambda, W, Z] = get_reduced_dimensions_from_lda(Mdl, considered_enc_emb_train_bias);
            liaisons_pred_by_layer_enc(:, i_layer) = predicted_label;

            % Bias Control
            considered_enc_emb_train_bias_reduced = considered_enc_emb_train_bias*W(:,1);
            center_non_liaisons = mean(considered_enc_emb_train_bias_reduced(matches(considered_liaisons_labels_train_bias,'Non-Liaison')));
            center_liaisons = mean(considered_enc_emb_train_bias_reduced(matches(considered_liaisons_labels_train_bias,'Liaison')));

            bias_vector_silence = (center_silence-center_non_silence) * (Z(1,:)');

            % Bias vector is normalized so that +1 = barycentre non-liaisons -> barycentre liaisons
            bias_vector_by_layer{i_layer} = [bias_vector_by_layer{i_layer}, bias_vector_silence];

            coord_enc_emb_train_bias{i_layer} = considered_enc_emb_train_bias;
        end
        save(name_liaisons_pred_enc, 'liaisons_pred_by_layer_enc', 'coord_enc_emb_train_bias', 'classifier_by_layer');
    end

    if ~exist([name_categorical_bias '.mat']) || reload_categorical_bias_vector
        save(name_categorical_bias, 'bias_vector_by_layer');
    end

    % Compute Liaisons prediction accuracy by layer
    liaisons_accuracy_by_layer_enc = zeros(1, nbr_layers);
    for i_layer = 1:nbr_layers
        if limit_number_of_samples > 0
            [~, ~, ~, ~, ~, ~, f1_by_class] = compute_classification_prediction_accuracy(all_liaisons_labels_val(1:limit_number_of_samples), liaisons_pred_by_layer_enc(:, i_layer), true);
        else
            [~, ~, ~, ~, ~, ~, f1_by_class] = compute_classification_prediction_accuracy(all_liaisons_labels_val, liaisons_pred_by_layer_enc(:, i_layer), true);
        end
        liaisons_accuracy_by_layer_enc(i_layer) = f1_by_class(1, 3);
    end

    % Load embeddings by layer: Decoder
    fprintf('Loading All embeddings in DECODER | Model: %s\n', name_model);
    if exist([name_dec_emb_mat '.mat']) && ~reload_decoder_emb
        load(name_dec_emb_mat);
    else
        [all_dec_emb_mat_train_bias, all_dec_emb_mat_residual_train_bias, all_dec_emb_mat_context_vector_train_bias] = load_decoder_embeddings_by_layer(path_model_train_bias, model_type, nbr_utt_train_bias, nbr_char_in_corpus_train_bias);
        [all_dec_emb_mat_val, all_dec_emb_mat_residual_val, all_dec_emb_mat_context_vector_val] = load_decoder_embeddings_by_layer(path_model_val, model_type, nbr_utt_val, nbr_char_in_corpus_val);

        if strcmp(model_type, 'tacotron')
            all_dec_emb_mat_train_bias{1} = all_dec_emb_mat_context_vector_train_bias{2};
            all_dec_emb_mat_val{1} = all_dec_emb_mat_context_vector_val{2};
        end

        save(name_dec_emb_mat, 'all_dec_emb_mat_train_bias', 'all_dec_emb_mat_val');
    end
    nbr_layers = length(all_dec_emb_mat_train_bias);

    % Compute Phonetic prediction with SVM
    if exist([name_phonetic_pred_dec '.mat']) && ~reload_phonetic_prediction_dec
        load(name_phonetic_pred_dec);
    else
        if limit_number_of_samples > 0
            phonetic_pred_by_layer_dec = cell(limit_number_of_samples, nbr_layers);
        else
            phonetic_pred_by_layer_dec = cell(length(all_indexes_non_null_duration_char_val), nbr_layers);
        end
        for i_layer = 1:nbr_layers
            fprintf('Compute Phonetic Prediction for model %s, layer %d/%d | DECODER\n', name_model, i_layer, nbr_layers);
    
            if limit_number_of_samples > 0
                considered_dec_emb_train_bias = double(all_dec_emb_mat_train_bias{i_layer}(all_indexes_non_null_duration_char_train_bias(1:limit_number_of_samples), :));
                considered_dec_emb_val = double(all_dec_emb_mat_val{i_layer}(all_indexes_non_null_duration_char_val(1:limit_number_of_samples), :));
                considered_phonetic_labels_train_bias = all_phonetic_labels_train_bias(all_indexes_non_null_duration_char_train_bias(1:limit_number_of_samples));
            else
                considered_dec_emb_train_bias = double(all_dec_emb_mat_train_bias{i_layer}(all_indexes_non_null_duration_char_train_bias, :));
                considered_dec_emb_val = double(all_dec_emb_mat_val{i_layer}(all_indexes_non_null_duration_char_val, :));
                considered_phonetic_labels_train_bias = all_phonetic_labels_train_bias(all_indexes_non_null_duration_char_train_bias);
            end
            Mdl = fitcdiscr(considered_dec_emb_train_bias, considered_phonetic_labels_train_bias);
            predicted_label = predict(Mdl,considered_dec_emb_val);
            
            phonetic_pred_by_layer_dec(:, i_layer) = predicted_label;
        end
        save(name_phonetic_pred_dec, 'phonetic_pred_by_layer_dec');
    end

    % Compute Phonetic prediction accuracy by layer
    all_accuracy_by_layer_dec = zeros(1, nbr_layers);
    phon_accuracy_by_layer_dec = zeros(1, nbr_layers);
    vowels_accuracy_by_layer_dec = zeros(1, nbr_layers);
    consonants_accuracy_by_layer_dec = zeros(1, nbr_layers);
    for i_layer = 1:nbr_layers
        if limit_number_of_samples > 0
            [all_accuracy, phon_accuracy, weighted_all_f1, weighted_phone_f1, weighted_vowels_f1, weighted_consonants_f1, f1_by_class] = compute_classification_prediction_accuracy(all_phonetic_labels_val(all_indexes_non_null_duration_char_val(1:limit_number_of_samples)), phonetic_pred_by_layer_dec(:, i_layer), false);
        else
            [all_accuracy, phon_accuracy, weighted_all_f1, weighted_phone_f1, weighted_vowels_f1, weighted_consonants_f1, f1_by_class] = compute_classification_prediction_accuracy(all_phonetic_labels_val(all_indexes_non_null_duration_char_val), phonetic_pred_by_layer_dec(:, i_layer), false);
        end
%         all_accuracy_by_layer_dec(i_layer) = all_accuracy;
%         phon_accuracy_by_layer_dec(i_layer) = phon_accuracy;
        all_accuracy_by_layer_dec(i_layer) = weighted_all_f1;
        phon_accuracy_by_layer_dec(i_layer) = weighted_phone_f1;
        vowels_accuracy_by_layer_dec(i_layer) = weighted_vowels_f1;
        consonants_accuracy_by_layer_dec(i_layer) = weighted_consonants_f1;
    end

%     % Compute Silences prediction with SVM
%     if exist([name_silences_pred_dec '.mat']) && ~reload_silences_prediction_dec
%         load(name_silences_pred_dec);
%     else
%         if limit_number_of_samples > 0
%             silences_pred_by_layer_dec = cell(limit_number_of_samples, nbr_layers);
%         else
%             silences_pred_by_layer_dec = cell(nbr_char_in_corpus_val, nbr_layers);
%         end
%         for i_layer = 1:nbr_layers
%             fprintf('Compute Silences Prediction for model %s, layer %d/%d | DECODER\n', name_model, i_layer, nbr_layers);
%     
%             if limit_number_of_samples > 0
%                 Mdl = fitcsvm(double(all_dec_emb_mat_train_bias{i_layer}(all_potential_silences_indexes_train_bias(1:limit_number_of_samples), :)), all_silences_labels_train_bias(1:limit_number_of_samples));
%                 predicted_label = predict(Mdl,double(all_enc_emb_mat_val(all_potential_silences_indexes_val(1:limit_number_of_samples), :, i_layer)));
%             else
%                 Mdl = fitcsvm(double(all_dec_emb_mat_train_bias{i_layer}(all_potential_silences_indexes_train_bias, :)), all_silences_labels_train_bias);
%                 predicted_label = predict(Mdl,double(all_enc_emb_mat_val(all_silences_labels_val, :, i_layer)));
%             end
%             silences_pred_by_layer_dec(:, i_layer) = predicted_label;
%         end
%         save(name_silences_pred_dec, 'silences_pred_by_layer_dec');
%     end

    % Compute Silence prediction accuracy by layer
    silence_accuracy_by_layer_dec = zeros(1, nbr_layers);
%     for i_layer = 1:nbr_layers
%         if limit_number_of_samples > 0
%             [~, ~, silence_accuracy, ~, ~, ~, f1_by_class] = compute_classification_prediction_accuracy(all_silences_labels_val(1:limit_number_of_samples), silences_pred_by_layer_dec(:, i_layer), true);
%         else
%             [~, ~, silence_accuracy, ~, ~, ~, f1_by_class] = compute_classification_prediction_accuracy(all_silences_labels_val, silences_pred_by_layer_dec(:, i_layer), true);
%         end
% %         silence_accuracy_by_layer_dec(i_layer) = silence_accuracy;
%         silence_accuracy_by_layer_dec(i_layer) = f1_by_class(2, 3);
%     end

    % Compute Liaisons prediction accuracy by layer
    liaisons_accuracy_by_layer_dec = zeros(1, nbr_layers);

    total_all_accuracy = [all_accuracy_by_layer_enc, all_accuracy_by_layer_dec];
    total_phon_accuracy = [phon_accuracy_by_layer_enc, phon_accuracy_by_layer_dec];
    total_vowels_accuracy = [vowels_accuracy_by_layer_enc, vowels_accuracy_by_layer_dec];
    total_consonants_accuracy = [consonants_accuracy_by_layer_enc, consonants_accuracy_by_layer_dec];
    total_silences_accuracy = [silence_accuracy_by_layer_enc, silence_accuracy_by_layer_dec];
    total_liaisons_accuracy = [liaisons_accuracy_by_layer_enc, liaisons_accuracy_by_layer_dec];
%%
    if strcmp(model_type, 'fastspeech')
        dim_to_disp = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17];
        max_layer_silence_pred = 8;
    elseif strcmp(model_type, 'tacotron')
        dim_to_disp = [1, 2, 3, 4, 5, 9, 10, 11];
        max_layer_silence_pred = 5;
    end

    % p(1, i_model).select();
    subplotMargin(posPlot,i_model);
    for i = 1:5
        plot(2:4, 2:4, '-', 'LineWidth', 8, 'Color',colors(i,:));
    end
    hold on;
%     plot(1:length(dim_to_disp), total_all_accuracy(dim_to_disp), '-', 'LineWidth', 2);
%     hold on;
    plot(1:length(dim_to_disp), total_phon_accuracy(dim_to_disp), '-', 'LineWidth', 2,'Color',colors(1,:),'HandleVisibility','off');
    plot(1:length(dim_to_disp), total_vowels_accuracy(dim_to_disp), '-', 'LineWidth', 2,'Color',colors(2,:),'HandleVisibility','off');
    plot(1:length(dim_to_disp), total_consonants_accuracy(dim_to_disp), '-', 'LineWidth', 2,'Color',colors(3,:),'HandleVisibility','off');
    plot(1:max_layer_silence_pred, total_silences_accuracy(dim_to_disp(1:max_layer_silence_pred)), '-', 'LineWidth', 2,'Color',colors(4,:),'HandleVisibility','off');
    plot(1:max_layer_silence_pred, total_liaisons_accuracy(dim_to_disp(1:max_layer_silence_pred)), '-', 'LineWidth', 2,'Color',colors(5,:),'HandleVisibility','off');

    ylim([0, 1.2]);
    grid on;
    box on;

%     title(sprintf('Model: %s', name_model));
    title(sprintf('%s', model_legend_name), 'Interpreter','latex');
    
    if i_model == nbr_models
        if strcmp(model_type, 'fastspeech')
    %         legend({'Phones + /\_/';'Phones';'Vowels';'Stop-Consonants';'Silences';'Liaisons';}, 'Location','northwest');
            lgnd = legend({'Phones';'Vowels';'Stop-Consonants';'Silences';'Liaisons';}, 'Location','northwest','interpreter','tex','Orientation','horizontal');
            lgnd.Position(1) = 0.25;
            lgnd.Position(2) = 0.02;
        else
            lgnd = legend({'Phones';'Vowels';'Stop-Consonants';'Silences';'Liaisons';}, 'Location','northwest','interpreter','tex','Orientation','vertical');
            lgnd.Position(1) = 0.55;
            lgnd.Position(2) = 0.47;
        end
    end
    

    % Add events
    if strcmp(model_type, 'fastspeech')
        lim_x = [1 15];
        plot([8.5, 8.5], [0, 1.2], '--', 'color', 'black', 'HandleVisibility','off','LineWidth',2);
        text((8.5-lim_x(1))/2+lim_x(1), 1.1, 'Encoder', 'HorizontalAlignment','center');
        text((lim_x(2)-8.5)/2+8.5, 1.1, 'Decoder', 'HorizontalAlignment','center');
        xticks(1:length(dim_to_disp));
        xticklabels(name_layer_fastspeech(dim_to_disp));
        xtickangle(50);
        xlim(lim_x)
        if i_model == 1
            ylabel('F1-score');
        end
        yticks(0:0.2:1);
        xlabel('\overrightarrow{Layers~Depth}', 'Interpreter','latex');

    elseif strcmp(model_type, 'tacotron')
        lim_x = [1 8];
        plot([5.5, 5.5], [0, 1.2], '--', 'color', 'black', 'HandleVisibility','off','LineWidth',2);
        text((5.5-lim_x(1))/2+lim_x(1), 1.1, 'Encoder', 'HorizontalAlignment','center');
        text((lim_x(2)-5.5)/2+5.5, 1.1, 'Decoder', 'HorizontalAlignment','center');
        xticks(1:length(dim_to_disp));
        xticklabels(name_layer_tacotron(dim_to_disp));
        xtickangle(40);
        xlim(lim_x)
        if i_model == 1
            ylabel('F1-score');
        end
        yticks(0:0.2:1);
%         xlabel('Layers');
        xlabel('\overrightarrow{Layers~Depth}', 'Interpreter','latex');
    end

    setPlotFonts(ptxt);

end


%%
if strcmp(model_type, 'tacotron')
elseif strcmp(model_type, 'fastspeech')
end