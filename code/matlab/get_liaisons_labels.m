function all_liaisons_labels = get_liaisons_labels(list_char_duration_in_corpus, silence_threshold)
    silences_indexes = list_char_duration_in_corpus >= silence_threshold;

    all_liaisons_labels = [];
    for i_char = 1:length(silences_indexes)
        if silences_indexes(i_char)
            all_liaisons_labels = [all_liaisons_labels; {'Liaison'}];
        else
            all_liaisons_labels = [all_liaisons_labels; {'Non-Liaison'}];
        end
    end
end
