%% load_accoustic_param_from_model.m
% Load all acoustic parameters from each character of the utterances generated for this model.
%
% Return the concatenation of all parameters.
%
function acoustic_params = load_acoustic_param_from_model(path_syn_file, path_model, limit, model_type)     
    % Model Type
    if strcmp(model_type,'tacotron')
%         S = read_csv(path_syn_file, 6);
        fid = fopen(path_syn_file, 'r');
        S = textscan(fid, "%s %s %s %s %s", 'Delimiter','|');
        fclose(fid);
        
        index_utt = 3;
    elseif strcmp(model_type,'fastspeech')
        fid = fopen(path_syn_file, 'r');
        S = textscan(fid, "%s %s %s %s %s", 'Delimiter','|');
        fclose(fid);
        
        index_utt = 3;
    end
    
    % Load all char in corpus
    all_char_index = [];
    for i_utt = 1:limit
        all_char_index = [all_char_index; phon_input_to_cell_array(S{index_utt}{i_utt})];
    end
    total_nbr_char = length(all_char_index);
    
    % Global parameters
    nbr_pitch_cwt_coef = 0;
    nbr_acc_params = 35+nbr_pitch_cwt_coef;
    acoustic_params = zeros(nbr_acc_params, total_nbr_char);
    index_start_char = 1;
    index_end_char = 0;
        
    % Incrementatly update results
    for i_utt = 1:limit
        fprintf('Acoustic mat "%s": utt %d\n', path_model, i_utt);
        A = load_acoustic_params(i_utt, path_model);
        len_utt = size(A, 1);
        
        index_end_char = index_end_char + len_utt;

        acoustic_params(:,index_start_char:index_end_char) = A';
        
        index_start_char = index_start_char + len_utt;
    end
end