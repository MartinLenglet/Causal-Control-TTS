function run_all()
%RUN_ALL Reproduce the main MATLAB analyses from the paper.
%
% Usage:
%   cd matlab
%   setup
%   run_all
%
% This runs the "entry scripts" in the recommended order. Each entry script
% reads model paths and experiment definitions from config_paths.m.
%
% Edit config_paths.m (copy from config_paths_template.m) before running.

if ~exist('cfg','var')
    if exist('config_paths','file')
        cfg = config_paths(); %#ok<NASGU>
        assignin('base','cfg',cfg);
    else
        error('config_paths.m not found. Copy config_paths_template.m to config_paths.m and edit paths.');
    end
end

fprintf('Running MATLAB analyses...\n');

try
    main_correlate_embeddings_with_acoustics_by_layer;
catch ME
    warning('Correlation-by-layer analysis failed: %s', ME.message);
end

try
    main_predict_phonetic_categories_by_layer;
catch ME
    warning('Phonetic prediction-by-layer analysis failed: %s', ME.message);
end

fprintf('Done.\n');
end
