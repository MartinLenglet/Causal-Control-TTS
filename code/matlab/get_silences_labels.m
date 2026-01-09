function all_silences_labels = get_silences_labels(list_char_duration_in_corpus, silence_threshold)
    silences_indexes = list_char_duration_in_corpus >= silence_threshold;

    all_silences_labels = [];
    for i_char = 1:length(silences_indexes)
        if silences_indexes(i_char)
            all_silences_labels = [all_silences_labels; {'Silence'}];
        else
            all_silences_labels = [all_silences_labels; {'Non-Silence'}];
        end
    end
end