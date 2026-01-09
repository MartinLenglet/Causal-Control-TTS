function list_char_duration_in_corpus = get_list_duration_phones_in_corpus(path_model, nbr_char_in_corpus, nbr_utt)
    list_char_duration_in_corpus = zeros(nbr_char_in_corpus, 1);
    frame_duration = 1000*256/22050; % (ms)

    index_char = 0;
    for i_utt = 1:nbr_utt
        if exist(sprintf('%sTEST%05d_syn_duration.mat', path_model, i_utt))
            load(sprintf('%sTEST%05d_syn_duration', path_model, i_utt)); %duration_mat
            for i_char = 1:length(duration_mat)
                index_char = index_char + 1;
                list_char_duration_in_corpus(index_char) = double(duration_mat(i_char));
            end
        else
            fid_seg = fopen(sprintf('%sTEST%05d_seg.csv', path_model, i_utt), 'r');
            S = textscan(fid_seg, '%s %f %f %f %f %f', 'Delimiter','\t', 'headerlines', 1);
            fclose(fid_seg);

            for i_char = 1:length(S{1})
                index_char = index_char + 1;
                list_char_duration_in_corpus(index_char) = round((double(S{3}(i_char))-double(S{2}(i_char)))/frame_duration);
            end
        end
    end
end