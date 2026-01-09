text_file = "csv/save_NEB_txt/val_phon_mean_distrib_calib_FSE_test_by_layer.txt";
fid = fopen(text_file, 'r');
P = textscan(fid, "%s %s %s %s %s", 'Delimiter','|');
fclose(fid);

file_calib_control = "_NEB_Fastspeech2_120000_phonPred_predictor_embeddings_test_control/layer_6";
file_ref = "_NEB_Fastspeech2_120000_phonPred_predictor_embeddings_test_emb_by_layer_mean_distrib_calib";
% Mean pitch values
mean_pitch = 91.053;
std_pitch = 3.57;

% Mean energy values
mean_energy = 65.386;
std_energy = 8.36;

index_utt = 3;
phon_utt = P{3}{index_utt};
text_utt = P{4}{index_utt};
split_phon = phon_input_to_cell_array(phon_utt);

list_bias_f0 = [
    516;
    0;
    -516;
];
nbr_bias = length(list_bias_f0);
colors_by_style = distinguishable_colors(nbr_bias);

figure(1)
clf;

for i_bias = 1:nbr_bias
    current_bias = list_bias_f0(i_bias);
    if current_bias < 0
        label_bias = 'moins';
    else
        label_bias = 'plus';
    end
    absolute_value = abs(current_bias);
    if current_bias == 0
        load(sprintf("%s/TEST%05d_syn_pitch", file_ref, index_utt)); % pitch_prediction_mat
        load(sprintf("%s/TEST%05d_syn_energy", file_ref, index_utt)); % energy_prediction_mat
    else
        sub_folder = sprintf("pitch_control_bias_%s_%d", label_bias, absolute_value);
        load(sprintf("%s/%s/TEST%05d_syn_pitch", file_calib_control, sub_folder, index_utt)); pitch_prediction_mat
        load(sprintf("%s/%s/TEST%05d_syn_energy", file_calib_control, sub_folder, index_utt)); energy_prediction_mat
    end

    utt_length = length(pitch_prediction_mat);
    pitch_prediction_mat = std_pitch*pitch_prediction_mat + mean_pitch;
    energy_prediction_mat = std_energy*energy_prediction_mat + mean_energy;

    % Premier subplot pour la pitch prediction
    subplot(2, 1, 1)

    plot(1:utt_length, pitch_prediction_mat, 'color', colors_by_style(i_bias, :), 'LineWidth',3);
    hold on;
    plot([1,utt_length], mean(pitch_prediction_mat)*ones(1,2), '--', 'color', colors_by_style(i_bias, :), 'LineWidth',3, 'HandleVisibility', 'off');

    % Second subplot pour l'energy prediction
    subplot(2, 1, 2)
    
    plot(1:utt_length, energy_prediction_mat, 'color', colors_by_style(i_bias, :), 'LineWidth',3);
    hold on;
    plot([1,utt_length], mean(energy_prediction_mat)*ones(1,2), '--', 'color', colors_by_style(i_bias, :), 'LineWidth',3, 'HandleVisibility', 'off');
end

subplot(2, 1, 1)
grid on;
title(sprintf('Text: "%s"', text_utt));
xticks(1:utt_length);
xticklabels(split_phon);
xtickangle(0);
xlim([0, utt_length+1]);
xlabel('Phonetic Inputs');
ylabel("Pitch Prediction (Semitones)");
legend({'+2 std'; 'Unbiased'; '-2 std'});

subplot(2, 1, 2)
grid on;
title(sprintf('Text: "%s"', text_utt));
xticks(1:utt_length);
xticklabels(split_phon);
xtickangle(0);
xlim([0, utt_length+1]);
xlabel('Phonetic Inputs');
ylabel("Energy Prediction (dB)");
%legend({'+2 std'; 'Unbiased'; '-2 std'});