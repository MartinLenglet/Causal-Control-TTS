%% load_pattern_index_model.m
% Load indexes corresponding to the specified pattern from the model.
%
% Return index list of the first char of the pattern.
%
function all_pattern_index = load_pattern_index_from_corpus(path_csv, pattern_to_find)
    fid = fopen(path_csv, 'r');
    S = textscan(fid, "%s %s %s %s %s", 'Delimiter','|');
    fclose(fid);
    
    nbr_utt = length(S{1});
    
    nbr_char_in_pattern = length(pattern_to_find);
    all_pattern_index = [];
        
    % Incrementatly update results
    index_char_in_corpus = 0;
    for i_utt = 1:nbr_utt
        current_utt = phon_input_to_cell_array(S{3}{i_utt});

        nbr_char = length(current_utt);
        
        for i_char=1:(nbr_char-(nbr_char_in_pattern-1))
            found_pattern = 0;
            for i_char_pattern=1:nbr_char_in_pattern
                if (current_utt{i_char+i_char_pattern-1}==pattern_to_find(i_char_pattern))
                    found_pattern = found_pattern + 1;
                end
            end
            if found_pattern == nbr_char_in_pattern
                all_pattern_index = [all_pattern_index; [i_utt, i_char, (index_char_in_corpus+i_char)]];
            end
        end
        index_char_in_corpus = index_char_in_corpus + nbr_char;
    end
end