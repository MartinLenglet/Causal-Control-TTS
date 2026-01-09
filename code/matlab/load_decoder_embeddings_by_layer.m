function [all_dec_emb_mat, all_dec_emb_mat_residual, all_dec_emb_mat_context_vector] = load_decoder_embeddings_by_layer( ...
    path_model, ...
    model_type, ...
    nbr_utt, ...
    nbr_char_in_corpus)

    tacotron_residual_dim = [128, 1024, 1024];
    tacotron_context_vec_dim = [512, 512, 512];

    name_emb_mat = sprintf('%sTEST%05d_syn_dec_emb_by_layer', path_model, 1);
    load(name_emb_mat); % variable name = dec_output_by_layer_mat

    if strcmp(model_type, 'fastspeech')
        %dim_emb = size(dec_output_by_layer_mat, 1);
        nbr_layers = size(dec_output_by_layer_mat, 3);
    elseif strcmp(model_type, 'tacotron')
        %dim_emb = 512;
        nbr_layers  = length(dec_output_by_layer_mat);
    end

    %all_dec_emb_mat = single(zeros(nbr_char_in_corpus, dim_emb, nbr_layers));
    all_dec_emb_mat = cell(nbr_layers, 1);
    all_dec_emb_mat_residual = cell(nbr_layers, 1);
    all_dec_emb_mat_context_vector = cell(nbr_layers, 1);
%         for i_layer = 1:nbr_layers
%             dim_emb = size(dec_output_by_layer_mat{i_layer}, 1);
%             all_dec_emb_mat{i_layer} = single(zeros(nbr_char_in_corpus, dim_emb));
% 
% %             all_dec_emb_mat_residual{i_layer} = single(zeros(nbr_char_in_corpus, tacotron_residual_dim(i_layer)));
% %             all_dec_emb_mat_context_vector{i_layer} = single(zeros(nbr_char_in_corpus, tacotron_context_vec_dim(i_layer)));
%         end

    for i_layer = 1:nbr_layers
        if strcmp(model_type, 'fastspeech')
            dim_emb = size(dec_output_by_layer_mat, 1);
            all_dec_emb_mat_residual_current_layer = single(zeros(nbr_char_in_corpus, 256));
            all_dec_emb_mat_context_vector_current_layer = single(zeros(nbr_char_in_corpus, 256));
        elseif strcmp(model_type, 'tacotron')
            dim_emb = size(dec_output_by_layer_mat{i_layer}, 1);
            all_dec_emb_mat_residual_current_layer = single(zeros(nbr_char_in_corpus, tacotron_residual_dim(i_layer)));
            all_dec_emb_mat_context_vector_current_layer = single(zeros(nbr_char_in_corpus, tacotron_context_vec_dim(i_layer)));
        end

        all_dec_emb_mat_current_layer = single(zeros(nbr_char_in_corpus, dim_emb));

        index_char = 0;
        for i_utt = 1:nbr_utt
            fprintf('Decoder Embeddings | Layer: %d/%d | Utt %d/%d\n', i_layer, nbr_layers, i_utt, nbr_utt);

            if strcmp(model_type, 'fastspeech')
                name_emb_mat = sprintf('%sTEST%05d_syn_dec_emb_by_layer', path_model, i_utt);
                load(name_emb_mat); % variable name = dec_output_by_layer_mat

%                     temp_dec_output_by_layer_mat = cell(nbr_layers, 1);
%                     for i_layer = 1:nbr_layers
%                         temp_dec_output_by_layer_mat{i_layer} = dec_output_by_layer_mat(:, :, i_layer);
%                     end
%                     dec_output_by_layer_mat = temp_dec_output_by_layer_mat;

                dec_output_current_layer = dec_output_by_layer_mat(:, :, i_layer);
            elseif strcmp(model_type, 'tacotron')
                % Load last encoder stat
                name_emb_mat = sprintf('%sTEST%05d_syn_dec_emb_by_layer', path_model, i_utt);
                load(name_emb_mat); % variable name = dec_output_by_layer_mat

%                 dec_output_by_layer = cell(nbr_layers, 1);
%                     for i_layer = 1:nbr_layers
%                         dec_output_current_layer = kron(dec_output_by_layer_mat{i_layer}, ones(1,2));
%                         dec_output_by_layer_mat{i_layer} = dec_output_current_layer;
%                     end
                dec_output_current_layer = kron(dec_output_by_layer_mat{i_layer}, ones(1,2));
            end
%             nbr_frames = size(dec_output_by_layer_mat, 2);
            nbr_frames = size(dec_output_current_layer, 2);

            % Get mean_emb of frames by char
            name_seg_file = sprintf('%sTEST%05d_seg.csv', path_model, i_utt);
            fid = fopen(name_seg_file, 'r');
            S = textscan(fid, "%s %f %f %f %f %f", 'Delimiter','\t', 'headerlines', 1);
            fclose(fid);

            nbr_char_in_utt = length(S{1});
            for i_char = 1:nbr_char_in_utt
                index_char = index_char + 1;

                first_frame = max(round((22050/256)*S{2}(i_char)/1000)+1, 1);
                last_frame = min(round((22050/256)*S{3}(i_char)/1000), nbr_frames);

                first_third = min(first_frame + floor((last_frame - first_frame + 1)/3), nbr_frames);
                second_third = min(first_frame + ceil((last_frame - first_frame + 1)*2/3), nbr_frames);

                if second_third < first_third
                    continue;
                end

                permute_dec_output_by_layer_mat = dec_output_current_layer';
                frames_current_char = permute_dec_output_by_layer_mat(first_third:second_third, :);
                mean_frames_current_char = mean(frames_current_char, 1);

                all_dec_emb_mat_current_layer(index_char, :) = mean_frames_current_char;
%                     all_dec_emb_mat{i_layer} = all_dec_emb_mat_current_layer;

                if strcmp(model_type, 'tacotron')
%                         all_dec_emb_mat_residual_current_layer = all_dec_emb_mat_residual{i_layer};
%                         all_dec_emb_mat_context_vector_current_layer = all_dec_emb_mat_context_vector{i_layer};
% 
                    all_dec_emb_mat_residual_current_layer(index_char, :) = mean_frames_current_char(1:tacotron_residual_dim(i_layer));
%                         all_dec_emb_mat_residual{i_layer} = all_dec_emb_mat_residual_current_layer;
% 
                    all_dec_emb_mat_context_vector_current_layer(index_char, :) = mean_frames_current_char(tacotron_residual_dim(i_layer)+1:end);
%                         all_dec_emb_mat_context_vector{i_layer} = all_dec_emb_mat_context_vector_current_layer;
                end
            end
        end
        all_dec_emb_mat{i_layer} = all_dec_emb_mat_current_layer;
        all_dec_emb_mat_residual{i_layer} = all_dec_emb_mat_residual_current_layer;
        all_dec_emb_mat_context_vector{i_layer} = all_dec_emb_mat_context_vector_current_layer;
    end
end