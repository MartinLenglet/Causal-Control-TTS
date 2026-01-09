function [all_accuracy, phon_accuracy, weighted_all_f1, weighted_phone_f1, weighted_vowels_f1, weighted_consonants_f1, f1_by_class] = compute_classification_prediction_accuracy(gt_labels, pred_labels, is_bi_class)
    list_unique_classes = unique([gt_labels; pred_labels]);
    nbr_unique_classes = length(list_unique_classes);

    range_unique_classes = 1:nbr_unique_classes;
    list_vowels = {'a'; 'u'; 'i'; 'e'; 'e^'; 'y'; 'x'; 'x^'; 'o'; 'o^'; 'x~'; 'e~'; 'o~'; 'a~';};
    list_vowels_indexes = range_unique_classes(ismember(list_unique_classes, list_vowels));
    list_consonants = {'p'; 't'; 'k'; 'b'; 'd'; 'g';};
    list_consonants_indexes = range_unique_classes(ismember(list_unique_classes, list_consonants));
    list_exclude_phones = {'_'; '__'};
    list_phones_indexes = range_unique_classes(~ismember(list_unique_classes, list_exclude_phones));

    nbr_char = length(gt_labels);

    conf_mat = zeros(nbr_unique_classes, nbr_unique_classes);

    for i_char = 1:nbr_char
        index_gt_class = find(cellfun(@(subc) strcmp(gt_labels{i_char}, subc), list_unique_classes));
        index_pred_class = find(cellfun(@(subc) strcmp(pred_labels{i_char}, subc), list_unique_classes));

        conf_mat(index_pred_class, index_gt_class) = conf_mat(index_pred_class, index_gt_class) + 1;
    end

    % micro-averaged F1-score, or the micro-F1 = overall classes accuracy
    % and recall
    true_results = sum(diag(conf_mat));
    nbr_results = sum(sum(conf_mat));
    all_accuracy = true_results / nbr_results;
    
    % weighted F1 by class: table = Precision/recall/f1-score by phone
    f1_by_class = zeros(nbr_unique_classes, 3);
    true_positive = diag(conf_mat);
    for i_class = 1:nbr_unique_classes
        % Precision
        precision = true_positive(i_class)/sum(conf_mat(i_class, :));
        f1_by_class(i_class, 1) = precision;
        % Recall
        recall = true_positive(i_class)/sum(conf_mat(:, i_class));
        f1_by_class(i_class, 2) = recall;
        % F1-score
        f1_score = 2 * precision * recall / (precision + recall);
        f1_by_class(i_class, 3) = f1_score;
    end

    % ALL F1_score
    weighted_all_f1 = 0;
    for i_class = 1:nbr_unique_classes
        if ~isnan(f1_by_class(i_class, 3))
            weighted_all_f1 = weighted_all_f1 + f1_by_class(i_class, 3)*sum(conf_mat(:, i_class));
        end
    end
    weighted_all_f1 = weighted_all_f1 / sum(sum(conf_mat));

    if ~is_bi_class
        phon_accuracy = sum(diag(conf_mat(3:end, 3:end))) / sum(sum(conf_mat(3:end, 3:end)));

        % ALL but silences
        weighted_phone_f1 = 0;
        for i_class = list_phones_indexes
            if ~isnan(f1_by_class(i_class, 3))
                weighted_phone_f1 = weighted_phone_f1 + f1_by_class(i_class, 3)*sum(conf_mat(:, i_class));
            end
        end
        weighted_phone_f1 = weighted_phone_f1 / sum(sum(conf_mat(:, list_phones_indexes)));
    
        % only vowels
        weighted_vowels_f1 = 0;
        for i_class = list_vowels_indexes
            if ~isnan(f1_by_class(i_class, 3))
                weighted_vowels_f1 = weighted_vowels_f1 + f1_by_class(i_class, 3)*sum(conf_mat(:, i_class));
            end
        end
        weighted_vowels_f1 = weighted_vowels_f1 / sum(sum(conf_mat(:, list_vowels_indexes)));
    
        % only consonants
        weighted_consonants_f1 = 0;
        for i_class = list_consonants_indexes
            if ~isnan(f1_by_class(i_class, 3))
                weighted_consonants_f1 = weighted_consonants_f1 + f1_by_class(i_class, 3)*sum(conf_mat(:, i_class));
            end
        end
        weighted_consonants_f1 = weighted_consonants_f1 / sum(sum(conf_mat(:, list_consonants_indexes)));
    else
        phon_accuracy = 0;
        weighted_phone_f1 = 0;
        weighted_vowels_f1 = 0;
        weighted_consonants_f1 = 0;
    end
end