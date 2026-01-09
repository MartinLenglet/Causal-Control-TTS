function [list_phones_in_corpus, char_index] = get_list_phones_in_corpus(list_utterances, use_phonetic_targets)
    nbr_utt = length(list_utterances);

    list_phones_in_corpus = [];
    char_index = 0;

    for i_utt = 1:nbr_utt
        fprintf('Utt: %d/%d\n', i_utt, nbr_utt);
        if use_phonetic_targets
            current_utt = split(list_utterances{i_utt}, ' ')';
        else
            current_utt = phon_input_to_cell_array(list_utterances{i_utt});
        end
        nbr_char = length(current_utt);
        for i_char = 1:nbr_char
            current_char = current_utt{i_char};
            char_index = char_index + 1;

            if ~isempty(list_phones_in_corpus) && ismember(current_char, list_phones_in_corpus(:,1))
                index_in_phones_list = find(cellfun(@(subc) strcmp(current_char, subc), list_phones_in_corpus(:,1)));
                list_phones_in_corpus{index_in_phones_list, 2} = [list_phones_in_corpus{index_in_phones_list, 2}; char_index];
            else
                list_phones_in_corpus = [list_phones_in_corpus; [{current_char}, char_index]];
            end
        end
    end

    [~, ia] = sort(list_phones_in_corpus(:, 1));
    list_phones_in_corpus = list_phones_in_corpus(ia, :);
    
%     char_index = char_index -1;
end