function all_enc_emb_mat = load_encoder_embeddings_by_layer(path_model, nbr_utt, nbr_char_in_corpus)

    name_emb_mat = sprintf('%sTEST%05d_syn_enc_emb_by_layer', path_model, 1);
    load(name_emb_mat); % variable name = enc_output_by_layer_mat

    dim_emb = size(enc_output_by_layer_mat, 1);
    nbr_layers = size(enc_output_by_layer_mat, 3);

    all_enc_emb_mat = single(zeros(nbr_char_in_corpus, dim_emb, nbr_layers));
    start_index_char = 1;
    end_index_char = 0;
    for i_utt = 1:nbr_utt
        fprintf('Encoder Embeddings | Utt %d/%d\n', i_utt, nbr_utt);
        name_emb_mat = sprintf('%sTEST%05d_syn_enc_emb_by_layer', path_model, i_utt);
        load(name_emb_mat); % variable name = enc_output_by_layer_mat

        nbr_char_in_utt = size(enc_output_by_layer_mat, 2);
        end_index_char = end_index_char + nbr_char_in_utt;

        all_enc_emb_mat(start_index_char:end_index_char, :, :) = permute(enc_output_by_layer_mat, [2, 1, 3]);

        start_index_char = end_index_char + 1;
    end
end