% Figure(1000) in the code = Fig. 5 in the paper
% Figure(30) in the code = Fig. 6 in the paper


clearvars;

% Load configuration (paths, model lists)
if ~exist('cfg','var'); cfg = config_paths(); end
models_tacotron = {
    'TCN', '\textit{\textbf{TC}}';
    'TCP', '\textit{\textbf{TC$_{\it{\mathbf{P}}}$}}';
};
% models_tacotron = {
%     'TCN', '$TC$';
%     'TCP', '$TC_{P}$';
% };
nbr_models_tacotron = length(models_tacotron(:,1));
% nbr_models_tacotron = 0;
layers_to_test_tacotron = [2, 4, 5, 9];
nbr_layers_tacotron = length(layers_to_test_tacotron);

name_layer_tacotron = {
    'Phon Emb';
    'Conv1';
    'Conv2';
    'Conv3';
    'Bi-LSTM';
    'Spk Emb';
    'F0 Emb';
    'E Emb';
    'CV';
    'LSTM1';
    'LSTM2';
    'Mel';
    'Mel+Post';
};

% models_fastspeech = {
%     'FSE', '$\mathbf{FS}$';
%     'FS', '$FS_{\backslash phon}$';
%     'FSP', '$FS_{\backslash E}$';
%     'FSN', '$FS_{\backslash P \backslash E}$';
% };

models_fastspeech = {
    'FSE', '\textit{\textbf{FS}}';
    'FS', '\textit{\textbf{FS$_{\backslash \it{\mathbf{phon}}}$}}';
    'FSP', '\textit{\textbf{FS$_{\backslash \it{\mathbf{E}}}$}}';
    'FSN', '\textit{\textbf{FS$_{\backslash \it{\mathbf{P}} \backslash \it{\mathbf{E}}}$}}';
};

% nbr_models_fastspeech = length(models_fastspeech(:,1));
nbr_models_fastspeech = 1;

layers_to_test_fastspeech = [4, 6, 14, 16, 17];
% layers_to_test_fastspeech = [4];
nbr_layers_fastspeech = length(layers_to_test_fastspeech);

name_layer_fastspeech = {
    'Phon Emb';
    'Pos Enc';
    'FFT1';
    'FFT2';
    'FFT3';
    'FFT4';
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
    'Mel';
    'Mel+Post';
};

bias = [
%   -9;
%   -6;
  -3;
  -2;
  -1.5;
  -1;
  -0.5;
  0;
  0.5;
  1;
  1.5;
  2;
  3;
%   6;
%   9;
];
ref_index = 6;
nbr_bias = length(bias);

% params_to_control = {
%     'F0', 2;
% };

% params_to_control = {
%     'log(D)', 1;
%     'F0', 2;
%     'F1', 3;
%     'F2', 4;
%     'F3', 5;
%     'Spectral Tilt', 6;
%     'Energy', 7;
% %     'relative_pos', 8;
%     'pfitzinger', 9;
%     'CoG', 10;
%     'SB1k', 11;
% };
params_to_control = {
    'D_{ }', 1;
    'f_o', 2;
    'F_1   ', 3;
    'F_2   ', 4;
    'F_3', 5;
    'ST_{ }', 6;
    'E_{ }      |', 7;
%     'relative_pos', 8;
    'pfitzinger', 9;
    'CoG_{ }', 10;
    'SB_{1kHz}      |    ', 11;
};
nbr_params_to_control = length(params_to_control(:,1));

load(fullfile(cfg.results_root,"data/all_utt_acoustic_params_GT")); % all_utt_acoustic_params
all_acoustic_params = all_utt_acoustic_params(:, 1:2:21);
mean_by_param = nanmean(all_acoustic_params);
std_by_param = nanstd(all_acoustic_params);

figure_counter = 10;
disp_all_graphs = false;%true;


%%
% Contrallability Tacotron2
disp('Tacotron 2');
[gain_by_param_by_layer_tacotron, r_square_by_param_by_layer_tacotron, saturation_by_param_by_layer_tacotron, figure_counter, max_by_param_by_layer_tacotron] = compute_controllability_by_layer( ...
    models_tacotron, ...
    params_to_control, ...
    std_by_param, ...
    name_layer_tacotron(layers_to_test_tacotron), ...
    bias, ...
    ref_index, ...
    disp_all_graphs, ...
    figure_counter ...
    );
%%
% Contrallability FastSpeech2
disp('FastSpeech 2');
[gain_by_param_by_layer_fastspeech, r_square_by_param_by_layer_fastspeech, saturation_by_param_by_layer_fastspeech, figure_counter, max_by_param_by_layer_fastspeech] = compute_controllability_by_layer( ...
    models_fastspeech, ...
    params_to_control, ...
    std_by_param, ...
    name_layer_fastspeech(layers_to_test_fastspeech), ...
    bias, ...
    ref_index, ...
    disp_all_graphs, ...
    figure_counter ...
    );
%%
% Resume results in cell array
[resume_controllability_tacotron, best_layer_indexes_array_TC] = resume_controllability_by_param_by_model( ...
    r_square_by_param_by_layer_tacotron, ...
    gain_by_param_by_layer_tacotron,  ...
    saturation_by_param_by_layer_tacotron, ... 
    models_tacotron, ...
    name_layer_tacotron(layers_to_test_tacotron), ...
    params_to_control(:,1), ...
    max_by_param_by_layer_tacotron ...
    );

% Resume results in cell array
[resume_controllability_fastspeech, best_layer_indexes_array_FS] = resume_controllability_by_param_by_model( ...
    r_square_by_param_by_layer_fastspeech, ...
    gain_by_param_by_layer_fastspeech,  ...
    saturation_by_param_by_layer_fastspeech, ... 
    models_fastspeech, ...
    name_layer_fastspeech(layers_to_test_fastspeech), ...
    params_to_control(:,1), ...
    max_by_param_by_layer_fastspeech ...
    );
%%

close all

% Plot controls by layer by model (Tacotron2)
figure_counter = plot_controllability_by_layer( ...
    r_square_by_param_by_layer_tacotron, ...
    gain_by_param_by_layer_tacotron, ...
    saturation_by_param_by_layer_tacotron, ...
    models_tacotron, ...
    name_layer_tacotron(layers_to_test_tacotron), ...
    params_to_control, ...
    figure_counter, ...
    'tacotron2', ...
    best_layer_indexes_array_TC, ...
    max_by_param_by_layer_tacotron);

%%
close all


% Plot controls by layer by model (FastSpeech2)
figure_counter = plot_controllability_by_layer( ...
    r_square_by_param_by_layer_fastspeech, ...
    gain_by_param_by_layer_fastspeech, ...
    saturation_by_param_by_layer_fastspeech, ...
    models_fastspeech, ...
    name_layer_fastspeech(layers_to_test_fastspeech), ...
    params_to_control, ...
    figure_counter, ...
    'fastspeech2', ...
    best_layer_indexes_array_FS, ...
    max_by_param_by_layer_fastspeech);

resume_controllability_tacotron = [resume_controllability_tacotron; resume_controllability_fastspeech];

pretty_print_2D_cell_array(resume_controllability_tacotron);
pretty_print_2D_cell_array(resume_controllability_fastspeech);