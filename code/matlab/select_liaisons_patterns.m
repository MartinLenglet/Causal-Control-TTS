function all_potential_liaisons_indexes = select_liaisons_patterns(path_corpus)
    end_of_word_pattern_liaisons = {
        'er';
        't';
        'n';
        'es';
    };
    nbr_end_of_word_patterns = length(end_of_word_pattern_liaisons);
    
    next_word_pattern_liaisons = {
        'a';
        'e';
        'i';
        'o';
        'u';
        'y';
        'é';
        'è';
        'ê';
        'à';
        'â';
        'ô';
        'ù';
        'û';
        'ô';
    };
    nbr_next_word_patterns = length(next_word_pattern_liaisons);
    
    pattern_to_find = cell(nbr_end_of_word_patterns*nbr_next_word_patterns, 1);
    for i_end_of_word = 1:nbr_end_of_word_patterns
        index_in_pattern = length(end_of_word_pattern_liaisons{i_end_of_word});
        for i_next_word = 1:nbr_next_word_patterns
            pattern_to_find{(i_end_of_word-1)*nbr_next_word_patterns+i_next_word, 1} = sprintf('%s %s', end_of_word_pattern_liaisons{i_end_of_word}, next_word_pattern_liaisons{i_next_word});
            pattern_to_find{(i_end_of_word-1)*nbr_next_word_patterns+i_next_word, 2} = index_in_pattern;
        end
    end

    all_potential_liaisons_indexes = [];
    for i_pattern = 1:length(pattern_to_find(:,1))
        all_pattern_indexes = load_pattern_index_from_corpus(path_corpus, pattern_to_find{i_pattern, 1});
        if ~isempty(all_pattern_indexes)
            all_potential_liaisons_indexes = [all_potential_liaisons_indexes; all_pattern_indexes(:, 3)-1+pattern_to_find{i_pattern, 2}];
        end
    end

    % Sort by ascent indexes in corpus
    all_potential_liaisons_indexes = sort(all_potential_liaisons_indexes);
end