function [all_potential_silences_indexes_in_corpus, all_potential_silences_indexes_of_utt, all_phonation_indexes, all_phonation_indexes_of_utt] = select_silences_patterns(path_corpus)
    list_pct = {'?','!',':',';','.','§','~','[',']','(',')','-','"','¬', ',', '«', '»', ' '};

    all_potential_silences_indexes = [];
    for i_pct = 1:length(list_pct)
        fprintf('Search for Silences pattern | Pattern: "%s" %d/%d\n', list_pct{i_pct}, i_pct, length(list_pct));
        all_pattern_indexes = load_pattern_index_from_corpus(path_corpus, list_pct{i_pct});
        all_potential_silences_indexes = [all_potential_silences_indexes; all_pattern_indexes];
    end

    % Save all phonation indexes
    all_phonation_indexes = (1:max(all_potential_silences_indexes(:,3)))';
    all_phonation_indexes(all_potential_silences_indexes(:, 3)) = [];
    % Find index of utterance
    all_phonation_indexes_of_utt = ones(length(all_phonation_indexes), 1);
    counter = 0;
    for i_phon = all_phonation_indexes'
        counter = counter + 1;
        i_phon_prev = i_phon-1;
        while i_phon_prev > 0
            index_phon_prev_in_corpus = find(all_potential_silences_indexes(:,3) == i_phon_prev);
            if ~isempty(index_phon_prev_in_corpus)
                all_phonation_indexes_of_utt(counter) = all_potential_silences_indexes(index_phon_prev_in_corpus,1);
                break;
            else
                i_phon_prev = i_phon_prev - 1;
            end
        end
    end

    % Exclude first char of utterances
    all_potential_silences_indexes(all_potential_silences_indexes(:,2) == 1, :) = [];

    % Exclude last chars of utterances 
    fid = fopen(path_corpus, 'r');
    C = textscan(fid, "%s %s %s %s %s", 'Delimiter','|');
    fclose(fid);
    nbr_utt = length(C{3});
    for i_utt = 1:nbr_utt
        fprintf('Exclude ending pct | Utt %d/%d\n', i_utt, nbr_utt);
        utt_text = phon_input_to_cell_array(C{3}{i_utt});
        nbr_char = length(utt_text);
        cursor_end_utt = nbr_char;
        indexes_patterns_current_utt = find(all_potential_silences_indexes(:,1) == i_utt);
        data_patterns_current_utt = all_potential_silences_indexes(indexes_patterns_current_utt, :);
        ignored_indexes = [];
        while ismember(utt_text{cursor_end_utt}, list_pct)
            ignored_indexes = [ignored_indexes; find(data_patterns_current_utt(:,2) == cursor_end_utt)];
            cursor_end_utt = cursor_end_utt - 1;
        end
        all_potential_silences_indexes(indexes_patterns_current_utt(ignored_indexes), :) = [];
    end

    % Sort by ascent indexes in corpus
    [~, order_list] = sort(all_potential_silences_indexes(:, 3));
    all_potential_silences_indexes_in_corpus = all_potential_silences_indexes(order_list, 3);
    all_potential_silences_indexes_of_utt = all_potential_silences_indexes(order_list, 1);
end
