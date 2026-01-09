% compute_seg_files_from_alignment.m
% ------------------------------------------------------------
% STEP 2 of the reproduction pipeline (see ../README.md):
% Build per-utterance segmentation files (TEST%05d_seg.csv) from the
% alignment/duration export produced during FastSpeech2 inference.
%
% Why: The Praat acoustic extraction script (Step 3) needs *_seg.csv to know
% the start/end time of each character/phoneme in the synthesized waveform.
%
% Expected inputs (per model output folder):
%   - TEST%05d_syn.wav               (synthesized audio, optional for this step)
%   - <cfg.alignment_csv_filename>   (one file for the whole set; delimiter '|')
%
% Output (per utterance, written into the same model folder):
%   - TEST%05d_seg.csv   (TAB-separated, header required by Praat)
%
% Configure which model folders to process in matlab/config_paths.m:
%   cfg.models_seg = { 'KEY', '/path/to/model_outputs', 'fastspeech'; ... };
%
% Notes:
% - This script optionally computes z-scores from stats_phon_corpus.mat.
%   If cfg.stats_phon_corpus_path does not exist, z-scores are set to 0.
% - The parsing of the alignment CSV is codebase-specific (FastSpeech2 fork).
%   The defaults (phon_col=3, align_col=5) match the original script.

clearvars;

if ~exist('cfg','var'); cfg = config_paths(); end

list_pct = {' ','?','!',':',';','.','§','~','[',']','(',')','-','"','¬', ',', '«', '»'};

% FastSpeech2 parameters (edit only if your hop size / sample rate differ)
n_frames_per_step = 2;
frame_duration = 1000*256/22050; % ms per frame (hop=256, sr=22050)

% Alignment CSV format (5 string columns separated by '|')
phon_colomn  = 3;
align_colomn = 5;

% Load corpus phoneme duration statistics if available
stats_loaded = false;
stats_phon_corpus = [];
if isfield(cfg,'stats_phon_corpus_path') && exist(cfg.stats_phon_corpus_path,'file')
    S = load(cfg.stats_phon_corpus_path); % expects stats_phon_corpus
    if isfield(S,'stats_phon_corpus')
        stats_phon_corpus = S.stats_phon_corpus;
        stats_loaded = true;
    end
end
if ~stats_loaded
    warning('stats_phon_corpus.mat not found; z-scores will be set to 0. Set cfg.stats_phon_corpus_path to enable z-scores.');
end

models = cfg.models_seg;
if isempty(models)
    error('cfg.models_seg is empty. Edit matlab/config_paths.m to specify model output folders.');
end
nbr_models = size(models,1);

for i_model = 1:nbr_models
    name_model = models{i_model,1};
    path_model = models{i_model,2};
    model_type = models{i_model,3};

    if ~endsWith(path_model, filesep)
        path_model = [path_model filesep]; %#ok<AGROW>
    end

    % In this repo we only document FastSpeech2, but keep the flag for clarity
    use_log_duration = strcmp(model_type,'fastspeech');

    csv_file = fullfile(path_model, cfg.alignment_csv_filename);
    if ~exist(csv_file,'file')
        error('Alignment CSV not found for model %s: %s', name_model, csv_file);
    end

    fprintf('Model: %s\n  folder: %s\n  alignment: %s\n', name_model, path_model, csv_file);

    fid = fopen(csv_file, 'r');
    F = textscan(fid, "%s %s %s %s %s", 'Delimiter','|');
    fclose(fid);
    limit = length(F{1});

    for i_utt = 1:limit
        % Each row is expected to contain a space-separated sequence for:
        %  - phon_colomn: per-character "ground truth" duration or a placeholder
        %  - align_colomn: per-character predicted duration (frames) and/or tokens
        %
        % The original script assumes column 4 holds duration_mat and column 2
        % holds log_duration_mat (adjust here if your export differs).
        %
        % IMPORTANT: adapt these two lines to your FastSpeech2 export.
        duration_mat     = str2num(F{4}{i_utt}); %#ok<ST2NM>
        log_duration_mat = str2num(F{2}{i_utt}); %#ok<ST2NM>

        split_current_phon_align = split(F{align_colomn}{i_utt}, ' ');
        nbr_char = length(split_current_phon_align);

        start_char = 0;
        seg_out = fullfile(path_model, sprintf("TEST%05d_seg.csv", i_utt));
        fid_out = fopen(seg_out, "w");
        fprintf(fid_out,"character\tstart\tend\tGTduration\tZScoreGT\tZScoreAlign\n");

        for i_char = 1:nbr_char
            gt_duration = NaN;

            if use_log_duration
                predicted_duration = exp(log_duration_mat(i_char)) - 1;
                char_duration_frames = max([predicted_duration, 0]);
            else
                char_duration_frames = floor(duration_mat(i_char));
            end

            char_duration_s = frame_duration*char_duration_frames;
            end_char = start_char + char_duration_s;

            current_phon = split_current_phon_align{i_char};
            if strcmp(current_phon, '_') || strcmp(current_phon, '__') || strcmp(current_phon, '#')
                current_phon = ' ';
            end

            % default z-scores
            current_z_score_gt = 0;
            current_z_score_align = 0;

            if stats_loaded
                if ismember(current_phon, list_pct)
                    cmp_phon = '_';
                else
                    cmp_phon = current_phon;
                end
                index_phon = find(cellfun(@(subc) strcmp(cmp_phon, subc), stats_phon_corpus(1, :)));
                if ~isempty(index_phon)
                    current_z_score_gt = (gt_duration - stats_phon_corpus{4, index_phon}) / stats_phon_corpus{5, index_phon};
                    current_z_score_align = (char_duration_s - stats_phon_corpus{4, index_phon}) / stats_phon_corpus{5, index_phon};
                end
            end

            fprintf(fid_out, "%s\t%.2f\t%.2f\t%d\t%0.2f\t%0.2f\n", ...
                current_phon, start_char, end_char, gt_duration, current_z_score_gt, current_z_score_align);

            start_char = end_char;
        end
        fclose(fid_out);
    end
end

fprintf('Segmentation files generated.\n');
