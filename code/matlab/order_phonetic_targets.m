function ordered_phonetic_labels = order_phonetic_targets(list_phone_target_in_corpus)
    all_phonetic_labels_train_bias = [];
    all_phonetic_indexes_train_bias = [];
    for i_phon_label = 1:size(list_phone_target_in_corpus, 1)
        current_phon_label = list_phone_target_in_corpus{i_phon_label, 1};
        current_list_indexes = list_phone_target_in_corpus{i_phon_label, 2};
        nbr_current_labels = length(current_list_indexes);

        all_phonetic_labels_train_bias = [all_phonetic_labels_train_bias ; repmat({current_phon_label}, nbr_current_labels, 1)];
        all_phonetic_indexes_train_bias = [all_phonetic_indexes_train_bias ; current_list_indexes];
    end
    [~, re_order] = sort(all_phonetic_indexes_train_bias);
    ordered_phonetic_labels = all_phonetic_labels_train_bias(re_order);
end