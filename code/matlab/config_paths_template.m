function cfg = config_paths()
%CONFIG_PATHS User-editable configuration for locating model outputs.
%
% Copy this file to config_paths.m and edit the paths below.
%
% Goal: make the whole MATLAB pipeline configurable from a *single* place.
% The analysis scripts (SPECOM_*.m) now read their model lists from this cfg.
%
% A "model output folder" is expected to contain (per utterance):
%   TEST00001_syn.wav
%   TEST00001_acoustic_params.csv
%   TEST00001_syn_enc_emb_by_layer.mat   (variable: enc_output_by_layer_mat)
%   TEST00001_syn_dec_emb_by_layer.mat   (variable: dec_output_by_layer_mat)
%
% See ../REPRODUCING.md for the full end-to-end procedure.

cfg = struct();

here = fileparts(mfilename('fullpath'));
cfg.repo_root = fullfile(here, '..');

% Where the analysis scripts read list/txt files (bias_ortho.txt, etc.)
% If you keep the repo layout unchanged, this default is fine.
cfg.csv_root = fullfile(cfg.repo_root, 'data', 'lists');

% Root directory where you store model output folders (EDIT ME)
cfg.models_root = fullfile(cfg.repo_root, 'data', 'models');

% Where to write figures/results (EDIT ME if desired)
cfg.results_root = fullfile(cfg.repo_root, 'results');

% Model output folders used to generate *_seg.csv from alignment exports.
% Each row: {key, model_output_dir, model_type}
% model_output_dir must contain TEST%05d_syn.wav and the alignment CSV (cfg.alignment_csv_filename).
cfg.models_seg = {
%   'FSE', fullfile(cfg.repo_root,'results_step_1'), 'fastspeech';
};


% -------------------------------------------------------------------------
% Model registry (EDIT THESE ENTRIES)
% -------------------------------------------------------------------------
%
% Each entry is a struct with these fields:
%   key                short id used in filenames inside MATLAB (e.g., 'FSE')
%   type               'fastspeech' | 'fastspeech2' | 'tacotron' | ...
%   legend             LaTeX label used in figures
%
%   phon_bias_dir       folder containing the *biased* synthesis outputs
%   phon_val_dir        folder containing the *validation* synthesis outputs
%   phon_bias_list      path to the list/txt file used by phonetic pred scripts
%   phon_val_list       path to the list/txt file used by phonetic pred scripts
%
%   corr_dir            folder containing outputs used in correlation scripts
%   corr_list           path to the list/txt file used in correlation scripts
%
%   cat_control_root    folder containing categorical-control outputs
%   cat_csv_name        filename of the categorical list (located under cfg.csv_root)
%   cat_emb_root        folder containing embeddings used for categorical analysis
%
% You can leave unused fields empty.

cfg.model_specs = [ ...
    struct(...
        'key','FSE',...
        'type','fastspeech',...
        'legend','$FS$',...
        'phon_bias_dir', fullfile(cfg.models_root, 'FSE_bias_ortho_emb_by_layer'),...
        'phon_val_dir',  fullfile(cfg.models_root, 'FSE_val_ortho_emb_by_layer'),...
        'phon_bias_list', fullfile(cfg.csv_root, 'bias_ortho.txt'),...
        'phon_val_list',  fullfile(cfg.csv_root, 'val_ortho.txt'),...
        'corr_dir',  fullfile(cfg.models_root, 'FSE_bias_mean_distrib'),...
        'corr_list', fullfile(cfg.csv_root, 'bias_phon_mean_distrib_FSE_bias_by_layer.txt'),...
        'cat_control_root', fullfile(cfg.models_root, 'FSE_test_control_categorical'),...
        'cat_csv_name', 'val_ortho_mean_distrib_calib_FSE_test_by_layer.txt',...
        'cat_emb_root', fullfile(cfg.models_root, 'FSE_test_emb_by_layer_mean_distrib_calib')...
    ),...
    struct(...
        'key','FS',...
        'type','fastspeech',...
        'legend','$FS_{\\backslash phon}$',...
        'phon_bias_dir', fullfile(cfg.models_root, 'FS_bias_ortho_emb_by_layer'),...
        'phon_val_dir',  fullfile(cfg.models_root, 'FS_val_ortho_emb_by_layer'),...
        'phon_bias_list', fullfile(cfg.csv_root, 'bias_ortho.txt'),...
        'phon_val_list',  fullfile(cfg.csv_root, 'val_ortho.txt'),...
        'corr_dir',  fullfile(cfg.models_root, 'FS_bias_mean_distrib'),...
        'corr_list', fullfile(cfg.csv_root, 'bias_phon_mean_distrib_FSE_noPhon_bias_by_layer.txt'),...
        'cat_control_root', fullfile(cfg.models_root, 'FS_test_control_categorical'),...
        'cat_csv_name', 'val_ortho_mean_distrib_calib_FSE_noPhon_test_by_layer.txt',...
        'cat_emb_root', fullfile(cfg.models_root, 'FS_test_emb_by_layer_mean_distrib_calib')...
    )...
];

% -------------------------------------------------------------------------
% Derived tables used by the analysis scripts (do not edit)
% -------------------------------------------------------------------------

% SPECOM_do_phonetic_pred_by_emb_layer.m expects 7 columns:
%   key | phon_bias_dir | phon_bias_list | phon_val_dir | phon_val_list | type | legend
n = numel(cfg.model_specs);
cfg.models_phonetic_pred = cell(n, 7);
cfg.models_phonetic_pred_op = cell(n, 7);
for i = 1:n
    m = cfg.model_specs(i);
    cfg.models_phonetic_pred(i,:) = {m.key, m.phon_bias_dir, m.phon_bias_list, m.phon_val_dir, m.phon_val_list, m.type, m.legend};

    % OP scripts use a different legend formatting in your originals.
    % If you want to preserve that exact styling, override legends here.
    cfg.models_phonetic_pred_op(i,:) = cfg.models_phonetic_pred(i,:);
end

% SPECOM_do_correlation_by_emb_layer*.m expects 5 columns:
%   key | corr_dir | corr_list | type | legend
cfg.models_corr = cell(n, 5);
cfg.models_corr_op = cell(n, 5);
for i = 1:n
    m = cfg.model_specs(i);
    cfg.models_corr(i,:) = {m.key, m.corr_dir, m.corr_list, m.type, m.legend};
    cfg.models_corr_op(i,:) = cfg.models_corr(i,:);
end

% SPECOM_compute_categorical_bias_effect_by_layer*.m expects 6 columns:
%   key | cat_control_root | cat_csv_name | cat_emb_root | type | legend
cfg.models_categorical = cell(n, 6);
for i = 1:n
    m = cfg.model_specs(i);
    cfg.models_categorical(i,:) = {m.key, m.cat_control_root, m.cat_csv_name, m.cat_emb_root, m.type, m.legend};
end


% Alignment file produced by your FastSpeech2 inference run (used to build *_seg.csv)
cfg.alignment_csv_filename = 'alignment.csv';  % edit to match your export

% Path to stats_phon_corpus.mat (needed to compute z-scores in *_seg.csv)
% If empty or missing, z-scores will be set to 0.
cfg.stats_phon_corpus_path = fullfile(cfg.repo_root, 'data', 'stats_phon_corpus.mat');

end